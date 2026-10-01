import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .alibi_wo_fa import ALiBiModelArgs, RMSNorm, FeedForward, repeat_kv
from .chunked_attn import plan_if_chunked


@dataclass
class DAPEALiBiModelArgs(ALiBiModelArgs):
    dape_mlp_width: int = 32  # Dimensão oculta do MLP (paper recomenda = n_heads)
    grad_checkpoint: bool = False  # recomputa ativações por camada no backward (troca compute por memória)
    # Bloco de linhas da atenção no forward de inferência. 0 = desligado
    # (matriz [b, h, T, T] inteira, comportamento histórico). Ver
    # models/chunked_attn.py. Não tem efeito sob autograd.
    attn_chunk: int = 0


# ---------------------------------------------------------------------------
# Módulo DAPE  (Appendix J do paper)
# ---------------------------------------------------------------------------

class DAPEModule(nn.Module):
    """
    Módulo DAPE multi-head (Appendix J).

    Recebe o produto QKᵀ e o bias ALiBi de todas as cabeças e devolve
    o termo de correção f(QKᵀ, B) com shape [bsz, n_heads, seqlen, seqlen].

    Implementa a Equação 3 do paper (variante com conexão residual):
        A = QKᵀ + B + f(QKᵀ, B)
    onde f(·) é um MLP de 2 camadas com LeakyReLU.
    """

    def __init__(self, n_heads: int, mlp_width: int):
        super().__init__()
        # Entrada: concatenação [QKᵀ, B] na dimensão das cabeças → 2 * n_heads features
        # Saída: correção por cabeça → n_heads features
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_heads, mlp_width),
            nn.LeakyReLU(),
            nn.Linear(mlp_width, n_heads),
        )

    def forward(self, qk_t: torch.Tensor, alibi_bias: torch.Tensor) -> torch.Tensor:
        """
        Args:
            qk_t      : [bsz, n_heads, seqlen, seqlen]  — QKᵀ / sqrt(d)
            alibi_bias: [1,   n_heads, seqlen, seqlen]  — bias ALiBi estático

        Returns:
            correction: [bsz, n_heads, seqlen, seqlen]
        """
        # Expande o bias estático para o tamanho do batch
        bias_tile = alibi_bias.expand(qk_t.shape[0], -1, -1, -1)

        # Concatena na dimensão das cabeças → [bsz, 2*n_heads, seqlen, seqlen]
        combined = torch.cat([qk_t, bias_tile], dim=1)

        # Rearrange para o MLP operar na última dimensão → [bsz, seqlen, seqlen, 2*n_heads]
        combined = rearrange(combined, 'b h q k -> b q k h')

        # MLP → [bsz, seqlen, seqlen, n_heads]
        correction = self.mlp(combined)

        # Restaura dimensão das cabeças para posição original
        return rearrange(correction, 'b q k h -> b h q k')


# ---------------------------------------------------------------------------
# Attention com DAPE
# ---------------------------------------------------------------------------

class DAPEALiBiAttention(nn.Module):
    """
    Variante com separação explícita do bias ALiBi e da máscara causal.
    O transformer passa dois tensores distintos:
      - mask      : máscara causal aditiva (-inf / 0),  [seqlen, seqlen] ou [bsz, 1, seqlen, seqlen]
      - alibi_bias: bias posicional ALiBi puro,          [1, n_heads, seqlen, seqlen]

    Isso permite que o DAPEModule receba exatamente B (sem -inf) conforme o paper.
    """

    def __init__(self, args: DAPEALiBiModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.dape = DAPEModule(self.n_local_heads, args.dape_mlp_width)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],        # máscara causal aditiva
        alibi_bias: Optional[torch.Tensor],  # [1, n_heads, seqlen, seqlen]
        plan=None,                           # ChunkedCausalPlan, só na inferência
    ) -> torch.Tensor:
        if plan is not None:
            return self._forward_chunked(x, plan)

        bsz, seqlen, _ = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        keys    = keys.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        values  = values.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        keys   = repeat_kv(keys, self.n_rep).transpose(1, 2)
        values = repeat_kv(values, self.n_rep).transpose(1, 2)

        # 1. Informação semântica: QKᵀ / sqrt(d)
        qk_t = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 2. Termo de correção adaptativa DAPE: f(QKᵀ, B)
        #    Somente quando há bias disponível (seqlen > 1)
        correction = self.dape(qk_t, alibi_bias) if alibi_bias is not None else 0

        # 3. Equação 3 do paper: scores = QKᵀ + B + f(QKᵀ, B)
        #    A máscara causal é somada junto ao bias
        scores = qk_t + correction
        if alibi_bias is not None:
            scores = scores + alibi_bias
        if mask is not None:
            scores = scores + mask  # soma a parte causal (-inf sobre a diagonal superior)

        scores = F.softmax(scores.float(), dim=-1).type_as(queries)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


    def _forward_chunked(self, x: torch.Tensor, plan) -> torch.Tensor:
        """Mesma matemática do forward acima, um bloco de linhas por vez.

        O MLP do DAPE é pontual em (i, j) — opera na dimensão das cabeças —
        e o softmax é por linha, então as linhas [i0, i1) podem ser
        processadas isoladamente. Por causalidade o bloco só precisa de
        K/V[0:i1].

        Ganho extra sobre o caminho original: o bias ALiBi deixa de ser um
        tensor [1, n_heads, T, T] construído uma vez e compartilhado pelas
        camadas (17 GiB em fp32 a 16k) e passa a ser uma fatia por bloco.
        """
        bsz, seqlen, _ = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        keys    = keys.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        values  = values.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        keys   = repeat_kv(keys, self.n_rep).transpose(1, 2)
        values = repeat_kv(values, self.n_rep).transpose(1, 2)

        out = None
        for i0, i1 in plan.chunks():
            q = queries[:, :, i0:i1]
            k = keys[:, :, :i1]
            v = values[:, :, :i1]

            alibi_bias = plan.alibi(i0, i1)
            mask = plan.mask(i0, i1)

            qk_t = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = qk_t + self.dape(qk_t, alibi_bias)
            scores = scores + alibi_bias
            scores = scores + mask

            scores = F.softmax(scores.float(), dim=-1).type_as(queries)
            chunk_out = torch.matmul(scores, v)

            if out is None:
                out = torch.empty(
                    bsz, self.n_local_heads, seqlen, self.head_dim,
                    dtype=chunk_out.dtype, device=chunk_out.device,
                )
            out[:, :, i0:i1] = chunk_out

            # Solta os tensores do bloco antes de alocar os do próximo, senão
            # o pico fica em dois blocos em vez de um.
            del qk_t, scores, chunk_out, alibi_bias, mask

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class DAPETransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: DAPEALiBiModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.attention = DAPEALiBiAttention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        alibi_bias: Optional[torch.Tensor],
        plan=None,
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), mask, alibi_bias, plan)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# ---------------------------------------------------------------------------
# Transformer principal
# ---------------------------------------------------------------------------

class DAPEALiBiTransformer(nn.Module):
    BLOCK_CLS = None

    def __init__(self, params: DAPEALiBiModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(self.BLOCK_CLS(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # Slopes registrados com o mesmo shape do ALiBiTransformer: [1, n_heads, 1, 1]
        slopes = self._get_slopes(params.n_heads)
        self.register_buffer(
            "slopes",
            torch.tensor(slopes).reshape(1, params.n_heads, 1, 1),
            persistent=False,
        )

        self.grad_checkpoint = getattr(params, "grad_checkpoint", False)
        # Sobrescrevível na instância carregada (model.attn_chunk = N) para
        # não depender do args.json de checkpoints já treinados.
        self.attn_chunk = getattr(params, "attn_chunk", 0)
        # 0 = so o bloco fixo acima decide; >0 = bloco automatico por comprimento.
        self.attn_ref_len = getattr(params, "attn_ref_len", 0)

    def forward(self, tokens: torch.Tensor, seq_codes: Optional[torch.Tensor] = None,
                return_hidden: bool = False):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # Na inferência com chunking, nem a máscara [T, T] nem o bias ALiBi
        # [1, n_heads, T, T] são materializados: o plano devolve fatias por bloco.
        plan = plan_if_chunked(self, seqlen, h.dtype, tokens.device, seq_codes, self.slopes)
        if plan is not None:
            for layer in self.layers:
                h = layer(h, None, None, plan)
            h = self.norm(h)
            if return_hidden:
                return h
            return self.output(h).float()

        mask = None
        alibi_bias = None

        if seqlen > 1:
            # --- Máscara causal aditiva ---
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

            if seq_codes is not None:
                # Document packing: bloqueia atenção entre seções diferentes
                mask = mask.unsqueeze(0).repeat(_bsz, 1, 1)
                section_mask = seq_codes.unsqueeze(-1) != seq_codes.unsqueeze(-2)
                mask[section_mask] = float("-inf")
                mask = mask.unsqueeze(-3)  # [bsz, 1, seqlen, seqlen]

            # --- Bias ALiBi puro: -(|i - j|) * slope ---
            positions = torch.arange(seqlen, device=tokens.device).float()
            alibi_bias = -(positions[None, :] - positions[:, None]).abs() * self.slopes
            # shape: [1, n_heads, seqlen, seqlen]

            mask = mask.type_as(h)
            alibi_bias = alibi_bias.type_as(h)

        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                # Recomputa as ativações da camada no backward em vez de
                # guardá-las. Matematicamente idêntico — é recomputação,
                # não aproximação. Necessário aqui porque o DAPE aplica um
                # MLP sobre tensores [b, h, T, T] e guarda os intermediários
                # de todas as 12 camadas.
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, mask, alibi_bias, use_reentrant=False
                )
            else:
                h = layer(h, mask, alibi_bias)

        h = self.norm(h)
        # return_hidden: devolve o estado escondido [b, T, dim] em vez dos
        # logits [b, T, vocab]. Em 512k os logits em fp32 sao 64 GiB (vocab
        # 32768 x 4 bytes por posicao) e nenhum forward cabe na placa; o
        # estado escondido nos mesmos 512k e 1,6 GiB. Quem chama projeta em
        # blocos de posicoes e reduz na hora (argmax no passkey, cross
        # entropy na perplexidade), ver eval_utils. Sem a flag, nada muda.
        if return_hidden:
            return h
        return self.output(h).float()

    # Reutiliza a lógica de slopes do ALiBiTransformer
    # o carregamento de checkpoints treinados com o ALiBi original.
    def _get_slopes(self, n: int):
        if math.log2(n).is_integer():
            return self._get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                self._get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    def _get_slopes_power_of_2(self, n: int):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * ratio ** i for i in range(n)]


DAPEALiBiTransformer.BLOCK_CLS = DAPETransformerBlock
