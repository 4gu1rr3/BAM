from typing import Optional

import torch

class ChunkedCausalPlan:
    def __init__(
        self,
        seqlen: int,
        chunk: int,
        dtype: torch.dtype,
        device: torch.device,
        seq_codes: Optional[torch.Tensor] = None,
        slopes: Optional[torch.Tensor] = None,
    ):
        self.seqlen = seqlen
        self.chunk = chunk
        self.dtype = dtype
        self.device = device
        self.seq_codes = seq_codes
        self.slopes = slopes

    def chunks(self):
        for i0 in range(0, self.seqlen, self.chunk):
            yield i0, min(i0 + self.chunk, self.seqlen)

    def mask(self, i0: int, i1: int) -> torch.Tensor:
        """Mascara causal aditiva das linhas [i0, i1) sobre as colunas [0, i1).

        Sem seq_codes: [i1-i0, i1]. Com document packing: [bsz, 1, i1-i0, i1],
        mesmo formato que o caminho nao-chunkado produz.
        """
        q = torch.arange(i0, i1, device=self.device).unsqueeze(-1)
        k = torch.arange(i1, device=self.device).unsqueeze(0)
        m = torch.zeros(i1 - i0, i1, dtype=self.dtype, device=self.device)
        m.masked_fill_(k > q, float("-inf"))

        if self.seq_codes is not None:
            section = (self.seq_codes[:, i0:i1].unsqueeze(-1)
                       != self.seq_codes[:, :i1].unsqueeze(-2))
            m = m.unsqueeze(0).masked_fill(section, float("-inf")).unsqueeze(1)

        return m

    def alibi(self, i0: int, i1: int) -> torch.Tensor:
        """Fatia do bias ALiBi puro: [1, n_heads, i1-i0, i1]."""
        pq = torch.arange(i0, i1, device=self.device, dtype=torch.float32).unsqueeze(-1)
        pk = torch.arange(i1, device=self.device, dtype=torch.float32).unsqueeze(0)
        return (-(pk - pq).abs() * self.slopes).to(self.dtype)


def chunk_for_length(seqlen: int, ref_len: int) -> int:
    if ref_len <= 0 or seqlen <= ref_len:
        return 0
    return max(64, (ref_len * ref_len) // seqlen)


def plan_if_chunked(
    module,
    seqlen: int,
    dtype: torch.dtype,
    device: torch.device,
    seq_codes: Optional[torch.Tensor] = None,
    slopes: Optional[torch.Tensor] = None,
) -> Optional[ChunkedCausalPlan]:

    chunk = getattr(module, "attn_chunk", 0)
    if chunk <= 0:
        # Modo automatico: bloco derivado do comprimento (ver chunk_for_length).
        chunk = chunk_for_length(seqlen, getattr(module, "attn_ref_len", 0))

    if chunk <= 0 or chunk >= seqlen or torch.is_grad_enabled():
        return None
    return ChunkedCausalPlan(seqlen, chunk, dtype, device, seq_codes, slopes)
