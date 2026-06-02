import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# Importamos utilitários do seu ALiBi original para evitar duplicação
from .alibi import ALiBiModelArgs, RMSNorm, repeat_kv

@dataclass
class DAPEALiBiModelArgs(ALiBiModelArgs):
    dape_mlp_width: int = 32 # Dimensão oculta do MLP sugerida pelo paper

class DAPEModule(nn.Module):
    """Módulo DAPE que processa todas as cabeças simultaneamente."""
    def __init__(self, n_heads: int, mlp_width: int):
        super().__init__()
        # Entrada: 2 * n_heads (QK^T de todas as cabeças + Bias de todas as cabeças)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_heads, mlp_width),
            nn.LeakyReLU(),
            nn.Linear(mlp_width, n_heads)
        )

    def forward(self, qk_t: torch.Tensor, alibi_bias: torch.Tensor):
        # qk_t: [bsz, n_heads, seqlen, seqlen]
        # alibi_bias: [1, n_heads, seqlen, seqlen]
        bsz = qk_t.shape[0]
        
        # Expande o bias estático para o tamanho do batch
        bias_tile = alibi_bias.expand(bsz, -1, -1, -1)
        
        # Concatena a atenção e o bias na dimensão das cabeças
        combined = torch.cat([qk_t, bias_tile], dim=1) # [bsz, 2*n_heads, seqlen, seqlen]
        
        # Move a dimensão das cabeças para o final para passar pelo Linear (MLP)
        combined = rearrange(combined, 'b h q k -> b q k h')
        
        # Aplica o MLP
        correction = self.mlp(combined)
        
        # Devolve a dimensão das cabeças para o lugar original
        return rearrange(correction, 'b q k h -> b h q k')


class DAPEALiBiAttention(nn.Module):
    def __init__(self, args: DAPEALiBiModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_local_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_local_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_local_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_local_heads * self.head_dim, args.dim, bias=False)
        
        self.dape = DAPEModule(self.n_local_heads, args.dape_mlp_width)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor], slopes: torch.Tensor):
        bsz, seqlen, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        k = repeat_kv(k, self.n_rep).transpose(1, 2)
        v = repeat_kv(v, self.n_rep).transpose(1, 2)

        # 1. Informação Semântica (QK^T)
        qk_t = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 2. Informação Posicional Estática (ALiBi Bias Matrix)
        q_idx = torch.arange(seqlen, device=x.device).view(seqlen, 1)
        kv_idx = torch.arange(seqlen, device=x.device).view(1, seqlen)
        relative_dis = kv_idx - q_idx # Distância relativa
        
        # [n_heads, 1, 1] * [1, seqlen, seqlen] -> [n_heads, seqlen, seqlen]
        alibi_bias = slopes.view(-1, 1, 1) * relative_dis.view(1, seqlen, seqlen)
        alibi_bias = alibi_bias.unsqueeze(0) # [1, n_heads, seqlen, seqlen]

        # 3. Termo de Correção Adaptativa (DAPE)
        correction = self.dape(qk_t, alibi_bias)

        # 4. Equação final de atenção: QK^T + Bias + f(QK^T, Bias)
        scores = qk_t + alibi_bias + correction

        # 5. Aplicação da Máscara (Causal + Seq Codes se houver)
        if mask is not None:
            # Expandir máscara para [bsz, 1, seqlen, seqlen]
            mask_expanded = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask_expanded, float('-inf'))
        else:
            # Máscara causal padrão se nenhuma for passada
            causal_mask = torch.ones((seqlen, seqlen), dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~causal_mask.view(1, 1, seqlen, seqlen), float('-inf'))

        # 6. Softmax e multiplicação por V
        probs = F.softmax(scores, dim=-1)
        output = probs @ v
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

# --------------------------------------------------------------------------------
# Classes de bloco que usam a nova Attention
# --------------------------------------------------------------------------------

class DAPETransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: DAPEALiBiModelArgs):
        super().__init__()
        from .alibi import FeedForward # Importado aqui para evitar circularidade pesada
        self.layer_id = layer_id
        self.attention = DAPEALiBiAttention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=args.dim, multiple_of=args.multiple_of, ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor], slopes: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), mask, slopes)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

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

        # Lógica de Slopes original do ALiBi
        slopes = self._get_slopes(params.n_heads)
        self.register_buffer("slopes", torch.tensor(slopes).reshape(params.n_heads), persistent=False)

    def forward(self, tokens: torch.Tensor, seq_codes: Optional[torch.Tensor] = None):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        # Criação da máscara Booleana explícita (sem flex_attention)
        causal_mask = torch.ones((seqlen, seqlen), dtype=torch.bool, device=tokens.device).tril()
        mask = causal_mask.unsqueeze(0).expand(bsz, -1, -1) # [bsz, seqlen, seqlen]
        
        if seq_codes is not None:
            # Máscara de documento (document packing)
            seq_mask = seq_codes.unsqueeze(-1) == seq_codes.unsqueeze(1)
            mask = mask & seq_mask

        for layer in self.layers:
            h = layer(h, mask, self.slopes)
        h = self.norm(h)
        return self.output(h).float()
    
    # Repete as funções de cálculo de slopes do seu ALiBi
    def _get_slopes(self, n):
        if math.log2(n).is_integer():
            return self._get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return self._get_slopes_power_of_2(closest_power_of_2) + self._get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

    def _get_slopes_power_of_2(self, n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]