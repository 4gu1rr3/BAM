"""DAPE-ALiBi + SSMax com o bias posicional DENTRO do escalonamento.

Esta e a formula usada no run de 12/09 (jobs 954/955), que divergiu no step
88.  Existe so para diagnostico -- nao use em baseline novo.

    PE DENTRO (aqui):                 scores = (QK^T + B + f(QK^T,B)) * log(n) * s
    PE FORA   (dape_alibi_ssmax.py):  scores =  QK^T * log(n) * s + f(QK^T,B) + B

Herda tudo de dape_alibi_ssmax: a unica diferenca e o _scores.  Manter por
heranca (e nao por copia) garante que nenhuma outra diferenca se infiltre na
comparacao.

Traz tambem uma sonda (PROBE) que grava, por camada, as grandezas do forward
e do backward que o trace de parametros nao enxerga.  Ela so liga quando
train.py pede; com PROBE["on"] falso o caminho de calculo e identico ao
original.
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .alibi_wo_fa import RMSNorm, FeedForward
from .dape_alibi_ssmax import (
    DAPEALiBiSSMaxModelArgs,
    DAPEALiBiSSMaxAttention,
    DAPEALiBiSSMaxTransformer,
)


PROBE = {"on": False, "rec": None}


try:
    _nocompile = torch._dynamo.disable
except AttributeError:
    def _nocompile(f):
        return f


def _probe_slot(layer_id):
    if not PROBE["on"] or PROBE["rec"] is None:
        return None
    return PROBE["rec"].setdefault(layer_id, {})


def _sub(t):
    """Amostra [0:1, :, ::8, :] -- 1/(b*8) do tensor, estatistica equivalente
    a um custo de memoria desprezivel."""
    return t[0:1, :, ::8, :].detach().float() if t.dim() == 4 else t.detach().float()


def _rms(t):
    if t is None:
        return float("nan")
    return float(t.detach().float().pow(2).mean().sqrt())


def _rms_sub(t):
    if t is None:
        return float("nan")
    return float(_sub(t).pow(2).mean().sqrt())



@_nocompile
def _probe_scores(layer_id, qk_t, correction, pre, scores, has_bias):
    r = _probe_slot(layer_id)
    if r is None:
        return
    r["qk_rms"] = _rms_sub(qk_t)
    r["qk_absmax"] = float(_sub(qk_t).abs().max())
    if has_bias:
        r["f_rms"] = _rms_sub(correction)
        r["f_absmax"] = float(_sub(correction).abs().max())
        r["pre_rms"] = _rms_sub(pre)
    _fin = _sub(scores)
    _fin = _fin[torch.isfinite(_fin)]
    if _fin.numel():
        r["score_rms"] = float(_fin.pow(2).mean().sqrt())
        r["score_min"] = float(_fin.min())
        r["score_max"] = float(_fin.max())
        r["score_absmax"] = float(_fin.abs().max())
    else:
        r["score_rms"] = r["score_min"] = r["score_max"] = float("nan")
    p = torch.softmax(scores[0:1, :, -1, :].detach().float(), dim=-1)
    r["attn_width"] = float((-(p * (p + 1e-30).log()).sum(-1)).exp().mean())
    if scores.requires_grad:
        scores.register_hook(lambda g, r=r: r.__setitem__("dscore_rms", _rms_sub(g)))
    if has_bias and correction.requires_grad:
        correction.register_hook(lambda g, r=r: r.__setitem__("df_rms", _rms_sub(g)))


@_nocompile
def _probe_block(layer_id, x, attn, ffn, out):
    r = _probe_slot(layer_id)
    if r is None:
        return
    r["x_rms"] = _rms(x)
    r["attn_out_rms"] = _rms(attn)
    r["ffn_out_rms"] = _rms(ffn)
    r["out_rms"] = _rms(out)
    if x.requires_grad:
        x.register_hook(lambda g, r=r: r.__setitem__("dx_rms", _rms(g)))


@dataclass
class DAPEALiBiSSMaxPEInModelArgs(DAPEALiBiSSMaxModelArgs):
    pass


class DAPEALiBiSSMaxPEInAttention(DAPEALiBiSSMaxAttention):
    _probe_id = -1

    def _scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        mask: Optional[torch.Tensor],
        alibi_bias: Optional[torch.Tensor],
        section_log_len: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """QK^T +B +f(QK^T,B) -> *SSMax -> +mascara."""
        qk_t = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        correction = self.dape(qk_t, alibi_bias) if alibi_bias is not None else 0

        scores = qk_t + correction
        if alibi_bias is not None:
            scores = scores + alibi_bias
        pre = scores

        if section_log_len is not None:
            scores = scores * section_log_len * self.seq_scale.unsqueeze(-1)

        if mask is not None:
            scores = scores + mask

        if PROBE["on"]:
            _probe_scores(self._probe_id, qk_t, correction, pre, scores,
                          alibi_bias is not None)
        return scores


class DAPESSMaxPEInTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: DAPEALiBiSSMaxPEInModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.attention = DAPEALiBiSSMaxPEInAttention(args)
        self.attention._probe_id = layer_id
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, mask, alibi_bias, section_log_len=None, plan=None):
        attn = self.attention(self.attention_norm(x), mask, alibi_bias, section_log_len, plan)
        h = x + attn
        ffn = self.feed_forward(self.ffn_norm(h))
        out = h + ffn

        if PROBE["on"]:
            _probe_block(self.layer_id, x, attn, ffn, out)
        return out


class DAPEALiBiSSMaxPEInTransformer(DAPEALiBiSSMaxTransformer):
    BLOCK_CLS = DAPESSMaxPEInTransformerBlock
