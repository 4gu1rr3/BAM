import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .alibi_wo_fa import RMSNorm, FeedForward, repeat_kv
from .chunked_attn import plan_if_chunked
from .cope_ssmax import section_log_len_rows
from .dape_alibi import DAPEModule, DAPEALiBiModelArgs, DAPEALiBiTransformer


@dataclass
class DAPEALiBiSSMaxModelArgs(DAPEALiBiModelArgs):
    seq_scale: bool = True


class DAPEALiBiSSMaxAttention(nn.Module):
    """Atencao do DAPE-ALiBi com SSMax, Equacao 2 do paper (sem residual).

        scores = (QK^T + f(QK^T, B)) * log(n) * s

    Duas escolhas, ambas deliberadas:

    1. SEM conexao residual (Eq 2, `Concat`, e nao a Eq 3 `Concat_Residual`
       que o dape_alibi.py implementa).  O bias ALiBi entra so como ENTRADA
       do MLP f(), nunca somado direto ao score.  O paper diz que a Eq 2 e
       a mais efetiva quando "the bias matrix is underperforming but still
       conveys positional information", e caracteriza justamente o ALiBi
       assim.

    2. O escalonamento do SSMax multiplica a soma INTEIRA, que e a leitura
       literal do SSMax -- a mesma convencao de bam_ssmax, cabam_ssmax e
       cope_ssmax.

    Somar B direto ao score (Eq 3) com o SSMax por cima foi o que divergiu
    em 12/09 (jobs 954/955).  Medido: o B tem RMS 52 contra 0.33 do QK^T, o
    conteudo vira 0.6% do score, e o modelo tem que subir o QK^T duas ordens
    de grandeza para a atencao responder ao texto -- rampa que acelera
    sozinha ate a norma do gradiente transbordar.  Sem o termo residual o B
    so chega ao score filtrado pelo f(), numa escala muito menor.  Ver
    models/dape_alibi_ssmax_pein.py, que guarda a formula antiga para
    diagnostico.
    """

    def __init__(self, args: DAPEALiBiSSMaxModelArgs):
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

        seq_scale = torch.ones((1, args.n_heads, 1), dtype=torch.float)
        self.seq_scale = nn.Parameter(seq_scale, requires_grad=args.seq_scale)

    def _scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        mask: Optional[torch.Tensor],
        alibi_bias: Optional[torch.Tensor],
        section_log_len: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """QK^T +f(QK^T,B) -> *SSMax -> +mascara."""
        qk_t = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        correction = self.dape(qk_t, alibi_bias) if alibi_bias is not None else 0
        scores = qk_t + correction

        if section_log_len is not None:
            scores = scores * section_log_len * self.seq_scale.unsqueeze(-1)

        if mask is not None:
            scores = scores + mask
        return scores

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        alibi_bias: Optional[torch.Tensor],
        section_log_len: Optional[torch.Tensor] = None,
        plan=None,
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

        scores = self._scores(queries, keys, mask, alibi_bias, section_log_len)
        scores = F.softmax(scores.float(), dim=-1).type_as(queries)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

    def _forward_chunked(self, x: torch.Tensor, plan) -> torch.Tensor:
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

            section_log_len = torch.isfinite(mask).sum(-1, keepdim=True).float().log()

            scores = self._scores(q, k, mask, alibi_bias, section_log_len)
            scores = F.softmax(scores.float(), dim=-1).type_as(queries)
            chunk_out = torch.matmul(scores, v)

            if out is None:
                out = torch.empty(
                    bsz, self.n_local_heads, seqlen, self.head_dim,
                    dtype=chunk_out.dtype, device=chunk_out.device,
                )
            out[:, :, i0:i1] = chunk_out

            del scores, chunk_out, alibi_bias, mask, section_log_len

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class DAPESSMaxTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: DAPEALiBiSSMaxModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.attention = DAPEALiBiSSMaxAttention(args)
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
        section_log_len: Optional[torch.Tensor] = None,
        plan=None,
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), mask, alibi_bias, section_log_len, plan)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class DAPEALiBiSSMaxTransformer(DAPEALiBiTransformer):
    BLOCK_CLS = DAPESSMaxTransformerBlock

    def forward(self, tokens: torch.Tensor, seq_codes: Optional[torch.Tensor] = None,
                return_hidden: bool = False):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        plan = plan_if_chunked(self, seqlen, h.dtype, tokens.device, seq_codes, self.slopes)
        if plan is not None:
            for layer in self.layers:
                h = layer(h, None, None, None, plan)
            h = self.norm(h)
            if return_hidden:
                return h
            return self.output(h).float()

        mask = None
        alibi_bias = None
        section_log_len = None

        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

            if seq_codes is not None:
                mask = mask.unsqueeze(0).repeat(_bsz, 1, 1)
                section_mask = seq_codes.unsqueeze(-1) != seq_codes.unsqueeze(-2)
                mask[section_mask] = float("-inf")
                mask = mask.unsqueeze(-3)

            positions = torch.arange(seqlen, device=tokens.device).float()
            alibi_bias = -(positions[None, :] - positions[:, None]).abs() * self.slopes

            mask = mask.type_as(h)
            alibi_bias = alibi_bias.type_as(h)
            section_log_len = section_log_len_rows(seqlen, _bsz, tokens.device, seq_codes)

        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, mask, alibi_bias, section_log_len, use_reentrant=False
                )
            else:
                h = layer(h, mask, alibi_bias, section_log_len)

        h = self.norm(h)
        if return_hidden:
            return h
        return self.output(h).float()
