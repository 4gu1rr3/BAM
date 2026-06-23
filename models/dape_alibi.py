import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .alibi_wo_flex_attention import ALiBiModelArgs, RMSNorm, FeedForward, repeat_kv


@dataclass
class DAPEALiBiModelArgs(ALiBiModelArgs):
    dape_mlp_width: int = 32  # Dimensão oculta do MLP (paper recomenda = n_heads)


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
        bias_tile = rearrange(alibi_bias, '1 h q k -> 1 h q k').expand(qk_t.shape[0], -1, -1, -1)

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
    ) -> torch.Tensor:
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
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), mask, alibi_bias)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# ---------------------------------------------------------------------------
# Transformer principal
# ---------------------------------------------------------------------------

class DAPEALiBiTransformer(nn.Module):
    def __init__(self, params: DAPEALiBiModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(DAPETransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # Slopes registrados com o mesmo shape do ALiBiTransformer: [1, n_heads, 1, 1]
        slopes = self._get_slopes(params.n_heads)
        self.register_buffer(
            "slopes",
            torch.tensor(slopes).reshape(1, params.n_heads, 1, 1),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor, seq_codes: Optional[torch.Tensor] = None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

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
            h = layer(h, mask, alibi_bias)

        h = self.norm(h)
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