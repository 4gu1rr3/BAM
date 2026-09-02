import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .chunked_attn import plan_if_chunked


@dataclass
class CoPEModelArgs:
    dim: int = 1024
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32768
    multiple_of: int = 1  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 1024

    # Numero de embeddings de posicao (semantica da Listing 1, Appendix B): as posicoes
    # vao de 0 a cope_npos_max-1, entao o p_max efetivo da Secao 4 e cope_npos_max-1.
    # Paper usa 64 no Wikitext-103 com T=1024.
    cope_npos_max: int = 64
    # attn     : gates reusam q.k (CoPE puro, config headline do paper, param-matched)
    # sep_keys : gates com projecao W_g dedicada (Secao 4, "Computing gates"); melhor PPL
    #            na Table 8, ao custo de +1 projecao por camada
    cope_gate_mode: str = "attn"
    cope_share_layers: bool = True   # embeddings de posicao compartilhados entre camadas (default do paper)
    grad_checkpoint: bool = False    # recomputa ativacoes por camada no backward (troca compute por memoria)
    # Bloco de linhas da atencao no forward de inferencia. 0 = desligado
    # (matriz [b, h, T, T] inteira, comportamento historico). Ver
    # models/chunked_attn.py. Nao tem efeito sob autograd.
    attn_chunk: int = 0


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# ---------------------------------------------------------------------------
# Modulo CoPE  (Secao 4 + Appendix B do paper)
# ---------------------------------------------------------------------------

class CoPE(nn.Module):
    """
    Contextual Position Encoding.

    Recebe as queries e os logits de gate (ja mascarados) e devolve o termo
    posicional z_i[p_ij] que e somado aos logits de atencao.

    Equacoes do paper:
        g_ij      = sigma(q_i . k_j)                                  (Eq. 3)
        p_ij      = sum_{k=j}^{i} g_ik                                (Eq. 4)
        z_i[p]    = q_i . e[p]         para p inteiro em [0, p_max-1] (Eq. 7)
        z_i[p_ij] = (p - floor(p)) z_i[ceil(p)]
                    + (1 - p + floor(p)) z_i[floor(p)]                (Eq. 8)
        a_ij      = Softmax(q_i . k_j + z_i[p_ij])                    (Eq. 9)

    Os embeddings e[p] sao compartilhados entre as cabecas (Secao 4,
    "Multi-head attention"); o compartilhamento entre camadas e controlado
    por `cope_share_layers` no CoPETransformer.
    """

    def __init__(self, npos_max: int, head_dim: int):
        super().__init__()
        self.npos_max = npos_max
        # [1, head_dim, npos_max] -> broadcast sobre (bsz, n_heads) no matmul
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))

    def forward(self, queries: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries    : [bsz, n_heads, seqlen, head_dim]
            gate_logits: [bsz, n_heads, seqlen, seqlen] — ja contem a mascara causal
                         aditiva (-inf), de modo que sigmoid(-inf) = 0 e as posicoes
                         futuras (ou de outro documento) nao entram na contagem.

        Returns:
            [bsz, n_heads, seqlen, seqlen] — termo posicional a somar nos logits
        """
        # 1. Gates (Eq. 3). Em fp32: a cumsum abaixo percorre a sequencia inteira
        #    e acumularia erro relevante em fp16/bf16.
        gates = torch.sigmoid(gate_logits.float())

        # 2. Posicoes contextuais (Eq. 4): soma reversa acumulada de j ate i.
        #    Como g_ik = 0 para k > i (mascara), a soma de j ate T-1 equivale a
        #    soma de j ate i.
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)

        # 3. Interpolacao entre os embeddings inteiros vizinhos (Eqs. 7-8).
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()

        logits_int = torch.matmul(queries, self.pos_emb.type_as(queries))  # [b, h, T, npos_max]
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)

        w = (pos - pos_floor).type_as(logits_int)
        return logits_ceil * w + logits_floor * (1 - w)


# ---------------------------------------------------------------------------
# Attention com CoPE
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """
    Atencao com CoPE no lugar de qualquer PE baseada em contagem de tokens.

    O modulo CoPE pode ser proprio da camada (`cope_share_layers=False`) ou
    recebido do transformer no forward (default, embeddings compartilhados
    entre todas as camadas).
    """

    def __init__(self, args: CoPEModelArgs):
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

        self.gate_mode = args.cope_gate_mode
        if self.gate_mode not in {"attn", "sep_keys"}:
            raise ValueError(f"cope_gate_mode invalido: {self.gate_mode}")

        # sep-keys: projecao dedicada para os gates (Secao 4, "Computing gates"),
        # desacoplando "o que e contado" de "o que e atendido".
        if self.gate_mode == "sep_keys":
            self.wg = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        # Sem compartilhamento entre camadas, cada atencao tem seu proprio e[p].
        self.cope = None if args.cope_share_layers else CoPE(args.cope_npos_max, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        cope: Optional[CoPE],  # modulo compartilhado vindo do transformer
        plan=None,             # ChunkedCausalPlan, so na inferencia
    ) -> torch.Tensor:
        if plan is not None:
            return self._forward_chunked(x, plan, cope)

        bsz, seqlen, _ = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        values = values.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        keys = repeat_kv(keys, self.n_rep).transpose(1, 2)
        values = repeat_kv(values, self.n_rep).transpose(1, 2)

        # Enderecamento por contexto: q.k / sqrt(d)
        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        # Logits usados para os gates. No modo default reaproveitamos q.k, que ja
        # foi calculado (Secao 4, "Computation").
        cope_module = self.cope if self.cope is not None else cope
        if cope_module is not None:
            if self.gate_mode == "attn":
                gate_logits = scores
            else:  # sep_keys: g_ij = sigma(q_i . W_g h_j)
                gate_keys = self.wg(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
                gate_keys = repeat_kv(gate_keys, self.n_rep).transpose(1, 2)
                gate_logits = torch.matmul(queries, gate_keys.transpose(2, 3)) / math.sqrt(self.head_dim)
                if mask is not None:
                    gate_logits = gate_logits + mask

            # Eq. 9: soma do termo posicional. Posicoes mascaradas seguem -inf.
            scores = scores + cope_module(queries, gate_logits)

        scores = F.softmax(scores.float(), dim=-1).type_as(queries)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


    def _forward_chunked(self, x: torch.Tensor, plan, cope: Optional[CoPE]) -> torch.Tensor:
        """Mesma matematica do forward acima, um bloco de linhas por vez.

        Cada linha i e independente das outras: a cumsum reversa dos gates
        roda dentro da linha e o softmax tambem, entao processar as linhas
        [i0, i1) isoladamente da o mesmo resultado. Por causalidade o bloco
        so precisa de K/V[0:i1] -- as chaves acima de i1-1 sao -inf para
        todas as linhas do bloco e nao contribuem nem para o softmax nem
        para a contagem de posicoes (sigmoid(-inf) = 0).
        """
        bsz, seqlen, _ = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        values = values.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        keys = repeat_kv(keys, self.n_rep).transpose(1, 2)
        values = repeat_kv(values, self.n_rep).transpose(1, 2)

        cope_module = self.cope if self.cope is not None else cope

        gate_keys = None
        if cope_module is not None and self.gate_mode == "sep_keys":
            gate_keys = self.wg(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            gate_keys = repeat_kv(gate_keys, self.n_rep).transpose(1, 2)

        out = None
        for i0, i1 in plan.chunks():
            q = queries[:, :, i0:i1]
            k = keys[:, :, :i1]
            v = values[:, :, :i1]
            mask = plan.mask(i0, i1)

            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = scores + mask

            gate_logits = None
            if cope_module is not None:
                if self.gate_mode == "attn":
                    gate_logits = scores
                else:
                    gate_logits = torch.matmul(
                        q, gate_keys[:, :, :i1].transpose(2, 3)
                    ) / math.sqrt(self.head_dim)
                    gate_logits = gate_logits + mask

                scores = scores + cope_module(q, gate_logits)

            scores = F.softmax(scores.float(), dim=-1).type_as(queries)
            chunk_out = torch.matmul(scores, v)

            if out is None:
                out = torch.empty(
                    bsz, self.n_local_heads, seqlen, self.head_dim,
                    dtype=chunk_out.dtype, device=chunk_out.device,
                )
            out[:, :, i0:i1] = chunk_out

            # Solta a matriz do bloco antes de alocar a do proximo, senao o
            # pico fica em dois blocos em vez de um.
            del scores, chunk_out, mask, gate_logits

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: CoPEModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        cope: Optional[CoPE],
        plan=None,
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), mask, cope, plan)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# ---------------------------------------------------------------------------
# Transformer principal
# ---------------------------------------------------------------------------

class CoPETransformer(nn.Module):
    def __init__(self, params: CoPEModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # e[p] compartilhado entre camadas (default do paper, Appendix D Table 8).
        # Quando nao compartilhado, cada Attention cria o seu.
        head_dim = params.dim // params.n_heads
        self.cope = CoPE(params.cope_npos_max, head_dim) if params.cope_share_layers else None

        self.grad_checkpoint = getattr(params, "grad_checkpoint", False)
        # Sobrescrevivel na instancia carregada (model.attn_chunk = N) para
        # nao depender de args.json de checkpoints ja treinados.
        self.attn_chunk = getattr(params, "attn_chunk", 0)
        # 0 = so o bloco fixo acima decide; >0 = bloco automatico por comprimento.
        self.attn_ref_len = getattr(params, "attn_ref_len", 0)

    def forward(self, tokens: torch.Tensor, seq_codes: Optional[torch.Tensor] = None,
                return_hidden: bool = False):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # Na inferencia com chunking, a mascara [T, T] nunca e materializada:
        # o plano devolve fatias [Tc, i1] por bloco.
        plan = plan_if_chunked(self, seqlen, h.dtype, tokens.device, seq_codes)
        if plan is not None:
            for layer in self.layers:
                h = layer(h, None, self.cope, plan)
            h = self.norm(h)
            if return_hidden:
                return h
            return self.output(h).float()

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

            if seq_codes is not None:
                # Document packing: bloqueia atencao entre secoes diferentes.
                # Os gates do CoPE tambem zeram nessas posicoes, entao a contagem
                # nao atravessa a fronteira entre documentos.
                mask = mask.unsqueeze(0).repeat(_bsz, 1, 1)
                section_mask = seq_codes.unsqueeze(-1) != seq_codes.unsqueeze(-2)
                mask[section_mask] = float("-inf")
                mask = mask.unsqueeze(-3)  # [bsz, 1, seqlen, seqlen]

            mask = mask.type_as(h)

        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                # Recomputa as ativacoes da camada no backward em vez de
                # guarda-las. Matematicamente identico -- e recomputacao,
                # nao aproximacao. Necessario aqui porque o CoPE guarda
                # varios tensores [b, h, T, T] por camada, dos quais dois
                # sao int64 (os indices de gather de pos_ceil/pos_floor),
                # o que estoura os 24 GiB da GPU mesmo com batch pequeno.
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, mask, self.cope, use_reentrant=False
                )
            else:
                h = layer(h, mask, self.cope)

        h = self.norm(h)
        # return_hidden: devolve o estado escondido [b, T, dim] em vez dos
        # logits [b, T, vocab]. Em 512k os logits em fp32 sao 64 GiB (vocab
        # 32768 x 4 bytes por posicao) e nenhum forward cabe na placa; o
        # estado escondido nos mesmos 512k e 1,6 GiB. Quem chama projeta em
        # blocos de posicoes e reduz na hora (argmax no passkey, cross
        # entropy na perplexidade), ver eval_utils. Sem a flag, nada muda.
        if return_hidden:
            return h
        output = self.output(h).float()
        return output
