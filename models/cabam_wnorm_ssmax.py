import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import create_block_mask


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
    theta_mu_init:   float = 0

    train_theta_beta: bool = True
    train_theta_alpha: bool = True
    train_theta_mu:   bool = False

    global_positional_encoding: bool = False
    seq_scale: bool = True
    
    mlp_width: int = 32

    prior_gain_init: float = 0.0
    prior_dir_init: str = "zeros"

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output if self.weight is None else output * self.weight


class NormLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = False, gain_init: float = 1.0,
                 dir_init: str = "zeros"):
        super().__init__()
        if dir_init == "zeros":
            w = torch.zeros(d_out, d_in)
        elif dir_init == "randn":
            w = torch.randn(d_out, d_in) / d_in**0.5
        elif dir_init == "orthogonal":
            w = torch.empty(d_out, d_in)
            nn.init.orthogonal_(w)
            w = w / w.norm(dim=1, keepdim=True)
        else:
            raise ValueError(f"dir_init desconhecido: {dir_init!r} (zeros | randn | orthogonal)")
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None
        self.gain = nn.Parameter(torch.full((d_out,), float(gain_init)))

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        return self.gain * (y / self.weight.norm(dim=1).clamp_min(1e-6))

class AttentionPrior(nn.Module):
    def __init__(self, n_heads: int, dim: int, hidden_dim: int, train_theta_alpha: bool = True, train_theta_beta: bool = True, train_theta_mu: bool = False,
                 prior_gain_init: float = 0.0, prior_dir_init: str = "zeros"):
        super().__init__()
        self.eps = 1e-5
        self.train_theta_alpha = train_theta_alpha
        self.train_theta_beta = train_theta_beta
        self.train_theta_mu = train_theta_mu
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.mlp = nn.Sequential(
            NormLinear(dim, 3*n_heads, bias=False, gain_init=prior_gain_init, dir_init=prior_dir_init),
        )
    
    @torch.no_grad()
    def prior_ceiling(self):
        g = self.mlp[-1].gain.detach().float().view(self.n_heads, 3).abs()
        y = g * self.dim**0.5
        return F.softplus(y[:, 0]), y[:, 1], 2.0 * torch.sinh(y[:, 2])

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        bs, seqlen, dim = x.shape

        pos_emb = self.mlp(x)
        pos_emb = pos_emb.view(bs, seqlen, self.n_heads, 3).transpose(1,2)

        alpha_raw = pos_emb[..., 0] if self.train_theta_alpha else pos_emb[..., 0].detach()
        # alpha = alpha_raw.exp()
        alpha = F.softplus(alpha_raw)
        beta = pos_emb[..., 1] if self.train_theta_beta else pos_emb[..., 1].detach()
        mu_raw = pos_emb[..., 2] if self.train_theta_mu else pos_emb[..., 2].detach()
        mu = mu_raw.exp() - mu_raw.neg().exp()

        return (alpha, beta, mu)

def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]
    
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2**math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]


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

@torch.compiler.disable
def _materialize(*tensors):
    return tuple(t.contiguous() for t in tensors)

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
            self.prior = AttentionPrior(n_heads=args.n_heads, dim=args.dim, hidden_dim=args.mlp_width, train_theta_alpha=args.train_theta_alpha, train_theta_beta=args.train_theta_beta, train_theta_mu=args.train_theta_mu, prior_gain_init=args.prior_gain_init, prior_dir_init=args.prior_dir_init)

        seq_scale =  torch.ones((1, args.n_heads, 1), dtype=torch.float)
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
        keys = keys.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        values = values.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        ssmax_mul = section_log_len * self.seq_scale
        alpha_t, beta_t, mu_t = self.prior(x) if self.local_positional_encoding else global_prior
        
        alpha_t, beta_t, mu_t, ssmax_mul = _materialize(alpha_t, beta_t, mu_t, ssmax_mul)

        def score_mod(score, b, h, q_idx, kv_idx):
            alpha = alpha_t[b, h, q_idx]
            beta = beta_t[b, h, q_idx]
            mu = mu_t[b, h, q_idx]

            b_pos = kv_idx - q_idx - mu
            prior = -((b_pos.abs() + self.prior.eps) ** beta) * alpha
            
            prior = torch.nan_to_num(prior, nan=-50.0, posinf=-50.0, neginf=-50.0)

            score = score + prior
            return score * ssmax_mul[b, h, q_idx]


        output = flex_attention(queries, keys, values, score_mod=score_mod, block_mask=mask)

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
            self.prior = AttentionPrior(params)

    def forward(self, tokens: torch.Tensor, seq_codes: Optional[torch.Tensor] = None,
                return_hidden: bool = False):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        if seq_codes is not None:
            correction = torch.zeros_like(seq_codes, device=tokens.device)
            positions = torch.arange(seq_codes.size(-1), device=tokens.device).unsqueeze(0)
            positions = positions.repeat(seq_codes.size(0), 1)

            first_tokens = seq_codes.diff(dim=-1) != 0
            correction[:,1:][first_tokens] = positions[:,1:][first_tokens]

            correction = correction.cummax(dim=-1).values
            positions = positions - correction + 1
            section_log_len = positions.log().unsqueeze(1)
        else:
            section_log_len = torch.arange(1, seqlen+1).log().unsqueeze(0).unsqueeze(0).to(tokens.device).repeat(bsz, 1, 1)
            seq_codes = torch.zeros_like(tokens, device=tokens.device)

        def mask_mod(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            seq_mask = seq_codes[b, q_idx] == seq_codes[b, kv_idx]
            return causal_mask & seq_mask
        mask = create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seqlen, KV_LEN=seqlen, device=tokens.device, BLOCK_SIZE=128)

        global_prior = None
        if self.global_positional_encoding:
            global_prior = self.prior(seqlen)


        for layer in self.layers:
            h = layer(h, mask, global_prior, section_log_len)
        h = self.norm(h)
        if return_hidden:
            return h
        output = self.output(h).float()
        return output