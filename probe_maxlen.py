import torch
from eval_utils import Evaluator

device = 'cuda:0'
ev = Evaluator(device=device, compile=False, dtype='bfloat16', perplexity_dataset_dirs=[], passkey_samplings=[])
model = ev.load_model('logs/l12/cabam/version_04/')
model = torch.compile(model)
model.to(device)
model.eval()

BLOCK = 128

def try_len(n):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.inference_mode():
            tokens = torch.randint(0, model.vocab_size, (1, n), device=device)
            out = model(tokens)
            torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        del out, tokens
        torch.cuda.empty_cache()
        return True, peak
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, None

lo, hi = 94208, 155648  # lo: known OK from job 790, hi: known OOM from job 790
print(f"lo={lo} OK (known from previous run), hi={hi} OOM (known from previous run)", flush=True)

for _ in range(7):
    mid = ((lo + hi) // 2 // BLOCK) * BLOCK
    if mid <= lo:
        break
    ok, peak = try_len(mid)
    if ok:
        print(f"seq_len={mid} -> OK, peak={peak:.2f} GiB", flush=True)
        lo = mid
    else:
        print(f"seq_len={mid} -> OOM", flush=True)
        hi = mid

print(f"RESULT: max feasible ~{lo} tokens ({lo/512:.1f}x train ctx), smallest failing ~{hi} tokens", flush=True)
