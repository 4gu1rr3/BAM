import time
import torch
from torch.nn.attention.flex_attention import flex_attention

from models.bam_ssmax import SSMaxBATransformer, SSMaxBATModelArgs
from models.cabam    import SSMaxBATransformer as CABAMTransformer, SSMaxBATModelArgs as CABAMModelArgs
from models.dape_alibi import DAPEALiBiTransformer, DAPEALiBiModelArgs

# Hiperparâmetros ────────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 4
SEQ_LEN      = 512
N_WARMUP     = 5
N_RUNS       = 20
DEVICE       = "cuda"

MODEL_KWARGS = dict(
    dim=1024,
    n_layers=32,
    n_heads=32,
    ffn_dim_multiplier=None,
    max_seq_len=SEQ_LEN,
    max_batch_size=BATCH_SIZE,
)

# Funções auxiliares ────────────────────────────────────────────────────────────────────────
def make_inputs(batch_size, seq_len, vocab_size, device):
    """Cria tokens e seq_codes aleatórios — não precisa de dataset real."""
    tokens    = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    seq_codes = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    return tokens, seq_codes

def measure(model, tokens, seq_codes, n_warmup, n_runs):
    """Mede latência (ms) e pico de VRAM (MB) do forward pass."""
    model.eval()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize()
    
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(tokens, seq_codes=seq_codes)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.synchronize()

        # Medição
        torch.cuda.reset_peak_memory_stats(DEVICE)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(tokens, seq_codes=seq_codes)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    elapsed_ms  = (t1 - t0) / n_runs * 1000          # ms por forward pass
    peak_mb     = torch.cuda.max_memory_allocated(DEVICE) / 1024**2  # MB

    return elapsed_ms, peak_mb

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for p in model.tok_embeddings.parameters())
    return total, total - embed

# Main ────────────────────────────────────────────────────────────────────────
def main():
    assert torch.cuda.is_available(), "CUDA não encontrado."
    print(f"\nDevice : {torch.cuda.get_device_name(0)}")
    print(f"Batch  : {BATCH_SIZE: 2d}  |  SeqLen : {SEQ_LEN}")
    print(f"Warmup : {N_WARMUP}  |  Runs   : {N_RUNS}\n")

    # BAM SSMax ────────────────────────────────────────────────
    bam_args = SSMaxBATModelArgs(**MODEL_KWARGS)
    bam_model = SSMaxBATransformer(bam_args).to(DEVICE).to(torch.bfloat16)

    tokens, seq_codes = make_inputs(BATCH_SIZE, SEQ_LEN, bam_args.vocab_size, DEVICE)

    bam_time, bam_mem = measure(bam_model, tokens, seq_codes, N_WARMUP, N_RUNS)
    bam_total, bam_nonembed = count_params(bam_model)

    print("── BAM SSMax ──────────────────────────────")
    print(f"  Parâmetros totais     : {bam_total:>12,}")
    print(f"  Parâmetros (sem emb.) : {bam_nonembed:>12,}")
    print(f"  Latência forward      : {bam_time:>10.2f} ms")
    print(f"  Pico de VRAM          : {bam_mem:>10.1f} MB")

    del bam_model # Tirar da memória
    torch.cuda.empty_cache() # Limpar cache

    # CABAM ────────────────────────────────────────────────
    cabam_args = CABAMModelArgs(**MODEL_KWARGS)
    cabam_model = CABAMTransformer(cabam_args).to(DEVICE).to(torch.bfloat16)

    tokens, seq_codes = make_inputs(BATCH_SIZE, SEQ_LEN, cabam_args.vocab_size, DEVICE)

    cabam_time, cabam_mem = measure(cabam_model, tokens, seq_codes, N_WARMUP, N_RUNS)
    cabam_total, cabam_nonembed = count_params(cabam_model)

    print("\n── CABAM ─────────────────────────────────")
    print(f"  Parâmetros totais     : {cabam_total:>12,}")
    print(f"  Parâmetros (sem emb.) : {cabam_nonembed:>12,}")
    print(f"  Latência forward      : {cabam_time:>10.2f} ms")
    print(f"  Pico de VRAM          : {cabam_mem:>10.1f} MB")

    del cabam_model # Tirar da memória
    torch.cuda.empty_cache() # Limpar cache
    
    # DAPE ALiBi ────────────────────────────────────────────────
    dape_alibi_args = DAPEALiBiModelArgs(**MODEL_KWARGS)
    dape_alibi_model = DAPEALiBiTransformer(dape_alibi_args).to(DEVICE).to(torch.bfloat16)
    
    tokens, seq_codes = make_inputs(BATCH_SIZE, SEQ_LEN, dape_alibi_args.vocab_size, DEVICE)
    
    dape_alibi_time, dape_alibi_mem = measure(dape_alibi_model, tokens, seq_codes, N_WARMUP, N_RUNS)
    dape_alibi_total, dape_alibi_nonembed = count_params(dape_alibi_model)
    print("\n── DAPE ALiBi ───────────────────────────────")
    print(f"  Parâmetros totais     : {dape_alibi_total:>12,}")
    print(f"  Parâmetros (sem emb.) : {dape_alibi_nonembed:>12,}")
    print(f"  Latência forward      : {dape_alibi_time:>10.2f} ms")
    print(f"  Pico de VRAM          : {dape_alibi_mem:>10.1f} MB")
    
    del dape_alibi_model # Tirar da memória
    torch.cuda.empty_cache() # Limpar cache
    
    # Comparação ────────────────────────────────────────────────
    delta_time = cabam_time - bam_time
    ratio_time = cabam_time / bam_time
    
    delta_mem  = cabam_mem  - bam_mem
    ratio_mem  = cabam_mem  / bam_mem
    
    delta_params = cabam_total - bam_total
    
    delta_time_dape = dape_alibi_time - bam_time
    ratio_time_dape = dape_alibi_time / bam_time
    
    delta_mem_dape  = dape_alibi_mem  - bam_mem
    ratio_mem_dape  = dape_alibi_mem  / bam_mem
    
    delta_params_dape = dape_alibi_total - bam_total

    print("\n── CA-BAM vs BAM SSMax ───────────")
    print(f"  Diferença de Parâmetros          : {delta_params:>+12,}")
    print(f"  Diferença de Latência            : {delta_time:>+10.2f} ms  ({ratio_time:.3f}×)")
    print(f"  Diferença de VRAM                : {delta_mem:>+10.1f} MB  ({ratio_mem:.3f}×)")
    print()

    print("── DAPE-ALiBi vs BAM SSMax ───────────")
    print(f"  Diferença de Parâmetros          : {delta_params_dape:>+12,}")
    print(f"  Diferença de Latência            : {delta_time_dape:>+10.2f} ms  ({ratio_time_dape:.3f}×)")
    print(f"  Diferença de VRAM                : {delta_mem_dape:>+10.1f} MB  ({ratio_mem_dape:.3f}×)")
    print()
    
if __name__ == "__main__":
    main()