import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class SSMaxBATModelArgs:
    dim: int = 1024
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32768
    multiple_of: int = 1
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 1024

    thata_beta_init: float | str = 0
    theta_alpha_init: float | str = 0
    theta_mu_init: float = 0

    train_theta_beta: bool = True
    train_theta_alpha: bool = True
    train_theta_mu: bool = False

    global_positional_encoding: bool = False
    seq_scale: bool = True

    mlp_width: int = 32


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


class AttentionPrior(nn.Module):
    """
    MLP contextual que produz (alpha, beta, mu) por token e por cabeça.

    Entrada : x  — [bs, seqlen, dim]  (saída da attention_norm, pré-atenção)
    Saída   : prior_params — [bs, n_heads, seqlen, 3]
                [..., 0] = alpha  (escala, sempre > 0  via exp)
                [..., 1] = beta   (forma)
                [..., 2] = mu     (localização, via sinh)
    """

    def __init__(self, n_heads: int, dim: int, hidden_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.n_heads = n_heads

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * n_heads, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [bs, seqlen, dim]
        bs, seqlen, _ = x.shape

        pos_emb = self.mlp(x)                                      # [bs, seqlen, 3*n_heads]
        pos_emb = pos_emb.view(bs, seqlen, self.n_heads, 3)        # [bs, seqlen, n_heads, 3]
        pos_emb = pos_emb.transpose(1, 2)                          # [bs, n_heads, seqlen, 3]

        # activations idênticas ao cabam com flex_attention
        pos_emb[..., 0] = pos_emb[..., 0].exp()                   # alpha > 0
        pos_emb[..., 2] = pos_emb[..., 2].exp() - pos_emb[..., 2].neg().exp()  # mu = sinh

        return pos_emb  # [bs, n_heads, seqlen, 3]

    def compute_bias(self, prior_params: torch.Tensor) -> torch.Tensor:
        """
        Materializa o bias posicional CABAM como tensor denso [bs, n_heads, T, T].

        prior_params: [bs, n_heads, seqlen, 3]
            [..., 0] = alpha_t   (escala)
            [..., 1] = beta_t    (forma)
            [..., 2] = mu_t      (localização)

        Equivalente vetorizado do score_mod do cabam com flex_attention:
            b_pos  = kv_idx - q_idx - mu_t[b, h, q_idx]
            prior  = -((|b_pos| + eps) ** beta_t) * alpha_t
        """
        bs, n_heads, seqlen, _ = prior_params.shape

        alpha = prior_params[..., 0]  # [bs, n_heads, seqlen]
        beta  = prior_params[..., 1]  # [bs, n_heads, seqlen]
        mu    = prior_params[..., 2]  # [bs, n_heads, seqlen]

        # posições relativas: kv_idx - q_idx, shape [seqlen, seqlen]
        idx = torch.arange(seqlen, device=prior_params.device)
        # b_pos[q, k] = k - q
        rel = (idx[None, :] - idx[:, None]).float()  # [T, T]

        # subtrai mu por query: [bs, n_heads, T, 1] - [bs, n_heads, T, T] → broadcast
        # mu é por q_idx, então expande na dimensão kv (última)
        b_pos = rel.unsqueeze(0).unsqueeze(0) - mu.unsqueeze(-1)  # [bs, n_heads, T, T]

        # prior = -((|b_pos| + eps)^beta) * alpha
        # alpha e beta são por q_idx → unsqueeze na dim kv
        bias = -((b_pos.abs() + self.eps) ** beta.unsqueeze(-1)) * alpha.unsqueeze(-1)
        # bias: [bs, n_heads, T, T]

        return bias


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class BayesianAttention(nn.Module):
    def __init__(self, args: SSMaxBATModelArgs):
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

        self.local_positional_encoding = not args.global_positional_encoding
        if self.local_positional_encoding:
            self.prior = AttentionPrior(
                n_heads=args.n_heads,
                dim=args.dim,
                hidden_dim=args.mlp_width,
            )

        # seq_scale: [1, n_heads, 1, 1] para broadcast direto sobre [bs, n_heads, T, T]
        seq_scale = torch.ones((1, args.n_heads, 1, 1), dtype=torch.float)
        self.seq_scale = nn.Parameter(seq_scale, requires_grad=args.seq_scale)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        global_prior: Optional[torch.Tensor] = None,
        section_log_len: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        keys    = keys.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        values  = values.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        keys   = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        queries = queries.transpose(1, 2)  # [bs, n_heads, T, head_dim]
        keys    = keys.transpose(1, 2)
        values  = values.transpose(1, 2)

        # scores QKᵀ / sqrt(d): [bs, n_heads, T, T]
        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # prior posicional CABAM
        if self.local_positional_encoding:
            prior_params = self.prior(x)                    # [bs, n_heads, T, 3]
            prior_bias   = self.prior.compute_bias(prior_params)  # [bs, n_heads, T, T]
        else:
            prior_bias = global_prior                       # [bs, n_heads, T, T] pré-computado

        scores = scores + prior_bias

        # SSMax: multiplica por log(posição_na_seção) * seq_scale
        # section_log_len: [bs, 1, T, 1] ou [bs, 1, T] → precisa ser [bs, n_heads, T, T]
        # unsqueeze(-1) para broadcast na dim kv
        ssmax_mul = section_log_len * self.seq_scale  # [bs, n_heads, T, 1]
        scores = scores * ssmax_mul

        # máscara causal + document packing
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(queries)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: SSMaxBATModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = BayesianAttention(args)
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
        global_prior: Optional[torch.Tensor] = None,
        section_log_len: Optional[torch.Tensor] = None,
    ):
        h = x + self.attention(self.attention_norm(x), mask, global_prior, section_log_len)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class SSMaxBATransformer(nn.Module):
    def __init__(self, params: SSMaxBATModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.global_positional_encoding = params.global_positional_encoding

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        if self.params.global_positional_encoding:
            # global prior não é contextual: mantém AttentionPrior estático
            # (sem MLP — use a versão do bam_ssmax se necessário)
            raise NotImplementedError(
                "global_positional_encoding não implementado no CABAM: "
                "o prior contextual é inerentemente local (depende de x)."
            )

    def forward(self, tokens: torch.Tensor, seq_codes: Optional[torch.Tensor] = None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        mask = None
        section_log_len = None

        if seqlen > 1:
            # máscara causal base
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

            if seq_codes is not None:
                # document packing: bloqueia atenção entre seções diferentes
                mask = mask.unsqueeze(0).repeat(_bsz, 1, 1)
                section_mask = seq_codes.unsqueeze(-1) != seq_codes.unsqueeze(-2)
                mask[section_mask] = float("-inf")
                mask = mask.unsqueeze(-3)  # [bs, 1, T, T]

                # log do comprimento acumulado dentro da seção, por token
                section_log_len = (
                    torch.tril(~section_mask, diagonal=0)
                    .sum(-1, keepdim=True)
                    .log()
                    .unsqueeze(-3)
                )  # [bs, 1, T, 1]
            else:
                section_log_len = (
                    torch.tril(torch.ones((1, 1, seqlen, seqlen)), diagonal=0)
                    .sum(-1, keepdim=True)
                    .log()
                    .to(tokens.device)
                )  # [1, 1, T, 1]

            mask = mask.type_as(h)

        for layer in self.layers:
            h = layer(h, mask, None, section_log_len)

        h = self.norm(h)
        return self.output(h).float()