"""CoPE + SSMax.

CoPE (Golovneva et al., 2024) com Scalable-Softmax no lugar do softmax:

    a_ij = SSMax( q_i.k_j + z_i[p_ij] )
         = Softmax( (q_i.k_j + z_i[p_ij]) * s_h * log n_i )

onde n_i e o numero de chaves visiveis a query i (dentro da secao, sob a
mascara causal) e s_h e um escalar treinavel por cabeca, init 1.0.

Duas decisoes de projeto, ambas herdadas de convencoes ja existentes no
repo e nao inferidas do paper do CoPE:

 1. O bias posicional fica DENTRO do escalonamento: escala-se (q.k + z),
    nao so q.k. E a convencao do bam_ssmax/cabam_ssmax
    (`score = score + prior; return score * ssmax_mul`) e a leitura
    literal do SSMax, que define o escalonamento sobre a entrada do
    softmax. RoPE e sinusoidal fazem o mesmo de fato, ja que neles a
    posicao ja esta dentro de `score` quando ssmax_mul multiplica.

 2. Os GATES do CoPE nao sao escalados. g_ij = sigma(q_i.k_j) (Eq. 3) e a
    maquinaria de CONTAGEM de posicoes, nao uma entrada de softmax: o
    SSMax e uma correcao de temperatura do softmax da atencao e nao tem
    o que dizer sobre um sigmoid. Escalar os gates faria a contagem p_ij
    -- e portanto o significado dos embeddings e[p] -- variar com o
    comprimento da sequencia, que e exatamente o que o teto p_max da
    Secao 4 existe para evitar. Entao os gates veem q.k cru (mascarado).

Combinacao CoPE+SSMax nao vem de paper nenhum: e ablacao nossa, o braco
com SSMax do grid de cinco PEs. Ver models/cope.py para o CoPE puro, que
esta inalterado e continua sendo a reproducao do paper.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .chunked_attn import plan_if_chunked
from .cope import CoPE, CoPEModelArgs, RMSNorm, FeedForward, repeat_kv


@dataclass
class CoPESSMaxModelArgs(CoPEModelArgs):
    seq_scale: bool = True


def section_log_len_rows(
    seqlen: int,
    bsz: int,
    device: torch.device,
    seq_codes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """log(n_i) por linha de query, no formato [bsz, 1, seqlen, 1].

    n_i = numero de chaves que a query i enxerga. Sem document packing e
    simplesmente i+1 (mascara causal). Com packing, so conta as chaves da
    mesma secao -- o mesmo criterio que zera os gates do CoPE atraves da
    fronteira entre documentos, entao contagem e atencao concordam.

    O formato [b, 1, T, 1] difere do [b, 1, T] usado pelos modelos de
    flex_attention (alibi_ssmax, bam_ssmax) porque aqui a matriz de
    scores e materializada: [b, 1, T, 1] faz broadcast direto sobre
    [b, h, T, T] pela dimensao das chaves. Mesma quantidade, formato
    diferente. E o mesmo formato do rotary_ssmax_wo_fa.
    """
    if seq_codes is not None:
        same_section = seq_codes.unsqueeze(-1) == seq_codes.unsqueeze(-2)
        visible = torch.tril(same_section, diagonal=0)
        n = visible.sum(-1, keepdim=True).float()
        return n.log().unsqueeze(-3)

    n = torch.arange(1, seqlen + 1, device=device, dtype=torch.float32)
    return n.log().view(1, 1, seqlen, 1).expand(bsz, 1, seqlen, 1)


class Attention(nn.Module):
    """Atencao do CoPE com SSMax. Mesma estrutura de models/cope.py."""

    def __init__(self, args: CoPESSMaxModelArgs):
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

        if self.gate_mode == "sep_keys":
            self.wg = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        self.cope = None if args.cope_share_layers else CoPE(args.cope_npos_max, self.head_dim)

        seq_scale = torch.ones((1, args.n_heads, 1), dtype=torch.float)
        self.seq_scale = nn.Parameter(seq_scale, requires_grad=args.seq_scale)

    def _scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        mask: Optional[torch.Tensor],
        cope_module: Optional[CoPE],
        section_log_len: Optional[torch.Tensor],
        gate_keys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """q.k -> +z[p] -> *SSMax -> +mascara. Comum aos dois caminhos."""
        qk = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        scores = qk
        if cope_module is not None:
            if self.gate_mode == "attn":
                gate_logits = qk if mask is None else qk + mask
            else:
                gate_logits = torch.matmul(
                    queries, gate_keys.transpose(2, 3)
                ) / math.sqrt(self.head_dim)
                if mask is not None:
                    gate_logits = gate_logits + mask

            scores = qk + cope_module(queries, gate_logits)

        if section_log_len is not None:
            scores = scores * section_log_len * self.seq_scale.unsqueeze(-1)

        if mask is not None:
            scores = scores + mask
        return scores

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        cope: Optional[CoPE],
        section_log_len: Optional[torch.Tensor] = None,
        plan=None,
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

        cope_module = self.cope if self.cope is not None else cope

        gate_keys = None
        if cope_module is not None and self.gate_mode == "sep_keys":
            gate_keys = self.wg(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            gate_keys = repeat_kv(gate_keys, self.n_rep).transpose(1, 2)

        scores = self._scores(queries, keys, mask, cope_module, section_log_len, gate_keys)
        scores = F.softmax(scores.float(), dim=-1).type_as(queries)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

    def _forward_chunked(self, x: torch.Tensor, plan, cope: Optional[CoPE]) -> torch.Tensor:
        """Mesma matematica, um bloco de linhas por vez (so inferencia).

        Cada linha i e independente: a cumsum reversa dos gates, o log n_i
        do SSMax e o softmax rodam todos dentro da linha. Por causalidade o
        bloco so precisa de K/V[0:i1].

        n_i vem da propria mascara do bloco (entradas finitas por linha),
        que e a mesma contagem que section_log_len_rows faz no caminho
        completo -- inclusive sob document packing, ja que plan.mask()
        aplica o mesmo criterio de secao.
        """
        bsz, seqlen, _ = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        values = values.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        keys = repeat_kv(keys, self.n_rep).transpose(1, 2)
        values = repeat_kv(values, self.n_rep).transpose(1, 2)

        cope_module = self.cope if self.cope is not None else cope

        gate_keys_full = None
        if cope_module is not None and self.gate_mode == "sep_keys":
            gate_keys_full = self.wg(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            gate_keys_full = repeat_kv(gate_keys_full, self.n_rep).transpose(1, 2)

        out = None
        for i0, i1 in plan.chunks():
            q = queries[:, :, i0:i1]
            k = keys[:, :, :i1]
            v = values[:, :, :i1]
            mask = plan.mask(i0, i1)

            section_log_len = torch.isfinite(mask).sum(-1, keepdim=True).float().log()

            gk = None if gate_keys_full is None else gate_keys_full[:, :, :i1]
            scores = self._scores(q, k, mask, cope_module, section_log_len, gk)
            scores = F.softmax(scores.float(), dim=-1).type_as(queries)
            chunk_out = torch.matmul(scores, v)

            if out is None:
                out = torch.empty(
                    bsz, self.n_local_heads, seqlen, self.head_dim,
                    dtype=chunk_out.dtype, device=chunk_out.device,
                )
            out[:, :, i0:i1] = chunk_out

            del scores, chunk_out, mask, section_log_len

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: CoPESSMaxModelArgs):
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
        section_log_len: Optional[torch.Tensor] = None,
        plan=None,
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), mask, cope, section_log_len, plan)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class CoPESSMaxTransformer(nn.Module):
    def __init__(self, params: CoPESSMaxModelArgs):
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

        head_dim = params.dim // params.n_heads
        self.cope = CoPE(params.cope_npos_max, head_dim) if params.cope_share_layers else None

        self.grad_checkpoint = getattr(params, "grad_checkpoint", False)
        self.attn_chunk = getattr(params, "attn_chunk", 0)
        self.attn_ref_len = getattr(params, "attn_ref_len", 0)

    def forward(self, tokens: torch.Tensor, seq_codes: Optional[torch.Tensor] = None,
                return_hidden: bool = False):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        plan = plan_if_chunked(self, seqlen, h.dtype, tokens.device, seq_codes)
        if plan is not None:
            for layer in self.layers:
                h = layer(h, None, self.cope, None, plan)
            h = self.norm(h)
            if return_hidden:
                return h
            return self.output(h).float()

        mask = None
        section_log_len = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

            if seq_codes is not None:
                mask = mask.unsqueeze(0).repeat(_bsz, 1, 1)
                section_mask = seq_codes.unsqueeze(-1) != seq_codes.unsqueeze(-2)
                mask[section_mask] = float("-inf")
                mask = mask.unsqueeze(-3)

            mask = mask.type_as(h)
            section_log_len = section_log_len_rows(seqlen, _bsz, tokens.device, seq_codes)

        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, mask, self.cope, section_log_len, use_reentrant=False
                )
            else:
                h = layer(h, mask, self.cope, section_log_len)

        h = self.norm(h)
        if return_hidden:
            return h
        return self.output(h).float()
