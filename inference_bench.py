import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="índice da GPU a usar")
parser.add_argument("--models", nargs="+", default=["bam_wo_fa", "cabam_wo_fa", "dape"],
                    choices=["bam", "cabam", "dape", "bam_wo_fa", "cabam_wo_fa"],
                    help="modelos a benchmarkar (ex: --models bam cabam)")
parser.add_argument("--seq_len", type=int, default=4096)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--n_warmup", type=int, default=5)
parser.add_argument("--n_runs", type=int, default=20)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import time
import torch

DEVICE     = "cuda:0"
SEQ_LEN    = args.seq_len
BATCH_SIZE = args.batch_size
N_WARMUP   = args.n_warmup
N_RUNS     = args.n_runs

MODEL_KWARGS = dict(
    dim=1024,
    n_layers=32,
    n_heads=32,
    ffn_dim_multiplier=None,
    max_seq_len=SEQ_LEN,
    max_batch_size=BATCH_SIZE,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_inputs(batch_size, seq_len, vocab_size, device):
    tokens    = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    seq_codes = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    return tokens, seq_codes

def measure(model, tokens, seq_codes, n_warmup, n_runs):
    model.eval()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(tokens, seq_codes=seq_codes)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(tokens, seq_codes=seq_codes)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) / n_runs * 1000
    peak_mb    = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
    return elapsed_ms, peak_mb

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for p in model.tok_embeddings.parameters())
    return total, total - embed

def bench_model(label, ModelArgs, ModelClass, kwargs):
    model_args  = ModelArgs(**kwargs)
    model       = ModelClass(model_args).to(DEVICE).to(torch.bfloat16)
    model = torch.compile(model)
    tokens, seq_codes = make_inputs(BATCH_SIZE, SEQ_LEN, model_args.vocab_size, DEVICE)
    t_ms, mem_mb      = measure(model, tokens, seq_codes, N_WARMUP, N_RUNS)
    total, nonembed   = count_params(model)
    print(f"── {label} {'─'*(42 - len(label))}")
    print(f"  Parâmetros totais     : {total:>12,}")
    print(f"  Parâmetros (sem emb.) : {nonembed:>12,}")
    print(f"  Latência forward      : {t_ms:>10.2f} ms")
    print(f"  Pico de VRAM          : {mem_mb:>10.1f} MB")
    del model
    torch.cuda.empty_cache()
    return {"label": label, "time": t_ms, "mem": mem_mb, "params": total}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"CUDA_VISIBLE_DEVICES : {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"GPU física em uso     : {torch.cuda.get_device_properties(0).name}")
    assert torch.cuda.is_available(), "CUDA não encontrado."
    print(f"\nDevice : {torch.cuda.get_device_name(0)}")
    print(f"Batch  : {BATCH_SIZE: 2d}  |  SeqLen : {SEQ_LEN}")
    print(f"Warmup : {N_WARMUP}  |  Runs   : {N_RUNS}")
    print(f"Modelos: {args.models}\n")

    registry = {}
    if "bam" in args.models:
        from models.bam_ssmax import SSMaxBATransformer, SSMaxBATModelArgs
        registry["bam"] = ("BAM SSMax", SSMaxBATModelArgs, SSMaxBATransformer)
    if "cabam" in args.models:
        from models.cabam import SSMaxBATransformer as CABAMTransformer, SSMaxBATModelArgs as CABAMModelArgs
        registry["cabam"] = ("CABAM", CABAMModelArgs, CABAMTransformer)
    if "dape" in args.models:
        from models.dape_alibi import DAPEALiBiTransformer, DAPEALiBiModelArgs
        registry["dape"] = ("DAPE ALiBi", DAPEALiBiModelArgs, DAPEALiBiTransformer)
    if "bam_wo_fa" in args.models:
        from models.bam_ssmax_wo_fa import SSMaxBATransformer, SSMaxBATModelArgs
        registry["bam_wo_fa"] = ("BAM SSMax (sem FA)", SSMaxBATModelArgs, SSMaxBATransformer)
    if "cabam_wo_fa" in args.models:
        from models.cabam_wo_fa import SSMaxBATransformer as CABAMTransformer, SSMaxBATModelArgs as CABAMModelArgs
        registry["cabam_wo_fa"] = ("CABAM (sem FA)", CABAMModelArgs, CABAMTransformer)

    results = {}
    for key in args.models:
        label, ModelArgs, ModelClass = registry[key]
        results[key] = bench_model(label, ModelArgs, ModelClass, MODEL_KWARGS)
        print()

    if len(results) > 1:
        baseline_key = args.models[0]
        baseline     = results[baseline_key]
        print(f"── Comparação vs {baseline['label']} {'─'*(30 - len(baseline['label']))}")
        for key in args.models[1:]:
            r = results[key]
            delta_time   = r["time"]   - baseline["time"]
            delta_mem    = r["mem"]    - baseline["mem"]
            delta_params = r["params"] - baseline["params"]
            ratio_time   = r["time"]   / baseline["time"]
            ratio_mem    = r["mem"]    / baseline["mem"]
            print(f"  {r['label']} vs {baseline['label']}:")
            print(f"    Δ Parâmetros : {delta_params:>+12,}")
            print(f"    Δ Latência   : {delta_time:>+10.2f} ms  ({ratio_time:.3f}×)")
            print(f"    Δ VRAM       : {delta_mem:>+10.1f} MB  ({ratio_mem:.3f}×)")
            print()

if __name__ == "__main__":
    main()