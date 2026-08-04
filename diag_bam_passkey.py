"""Diagnostic for the BAM-vs-paper passkey gap.

Part 1 -- equivalence: at lengths that fit, run the SAME prompt through the
non-chunked forward() (the code path that produced the paper's Figure 3) and
through forward_chunk() (the KV-cache path used by job 804's sweep), for all
20 equidistant depths. If they disagree, the chunked path is the culprit; if
they agree and both are low, the model itself is.

Part 2 -- paper protocol: chunked sweep at the paper's Figure 3 lengths
(n=20, equidistant, 5 digits), where the paper reports ~1.00 for BAM SSMax.

Logits are only ever materialized for the last few positions (logits_tail),
otherwise the (1, L, 32768) fp32 projection alone OOMs past ~8k.
"""
import argparse
import json

import torch
import torch._dynamo

from eval_utils import Evaluator, PromptGenerator

torch._dynamo.config.cache_size_limit = 128

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='logs/l12/bam_ssmax/version_02/')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dtype', type=str, default='bfloat16')
parser.add_argument('--sample_size', type=int, default=20)
parser.add_argument('--pred_digits', type=int, default=5)
parser.add_argument('--chunk_size', type=int, default=16384)
parser.add_argument('--equiv_lengths', type=int, nargs='+', default=[2048, 4096, 8192])
parser.add_argument('--sweep_lengths', type=int, nargs='+',
                    default=[512, 1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768])
parser.add_argument('--tag', type=str, default='BAM')
args = parser.parse_args()

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]

ev = Evaluator(device=args.device, compile=False, dtype=args.dtype,
               perplexity_dataset_dirs=[], passkey_samplings=[])
model = ev.load_model(args.log_dir)
model.to(args.device)
model.eval()
n_layers = model.n_layers

compiled_chunk = torch.compile(model.forward_chunk)
# Uncompiled flex_attention materializes the whole (B, H, L, L) score matrix and
# OOMs past ~4k, which is the only reason the non-chunked path can't be run at
# the lengths we care about -- compiling the model gives the fused kernel.
compiled_full = torch.compile(model)
generator = PromptGenerator(digits=args.pred_digits)
logits_tail = args.pred_digits + 2


@torch.inference_mode()
def run_chunked(total_ids, chunk_size):
    kv_caches = [None] * n_layers
    n = len(total_ids)
    start = 0
    logits = None
    while start < n:
        end = min(start + chunk_size, n)
        chunk = torch.tensor(total_ids[start:end], device=args.device).unsqueeze(0)
        is_last = (end == n)
        with torch.autocast(device_type='cuda', dtype=ptdtype):
            out, kv_caches = compiled_chunk(chunk, start, kv_caches, is_last,
                                            logits_tail if is_last else None)
        if is_last:
            logits = out
        start = end
    del kv_caches
    return logits


@torch.inference_mode()
def run_full(total_ids):
    """Plain non-chunked model(tokens) -- the exact path behind the paper's
    Figure 3. Only the tail is kept so the comparison matches run_chunked()."""
    ids = torch.tensor(total_ids, device=args.device).unsqueeze(0)
    with torch.autocast(device_type='cuda', dtype=ptdtype):
        out = compiled_full(ids)
    return out[:, -logits_tail:].float()


def is_correct(logits, pass_key):
    pred = logits.argmax(-1)
    pred_pass_key = list(pred[0, -args.pred_digits - 1:-1].cpu())
    return pred_pass_key == pass_key[1:]


print(f"=== [{args.tag}] PART 1: non-chunked forward() vs chunked forward_chunk(), "
      f"all {args.sample_size} equidistant depths ===", flush=True)
print(f"{'len':>7}  {'full':>6}  {'chunk':>6}  {'agree':>6}   full-hits / chunk-hits", flush=True)
for req_len in args.equiv_lengths:
    prompts, passkeys = generator(req_len, args.sample_size, 'equidistant')
    f_hits, c_hits, agree = [], [], 0
    for prompt, pass_key in zip(prompts, passkeys):
        total_ids = prompt + pass_key
        lf = run_full(total_ids)
        lc = run_chunked(total_ids, args.chunk_size)
        cf, cc = is_correct(lf, pass_key), is_correct(lc, pass_key)
        f_hits.append(int(cf))
        c_hits.append(int(cc))
        agree += int(lf.argmax(-1).equal(lc.argmax(-1)))
        del lf, lc
    print(f"{len(prompts[0]):>7}  {sum(f_hits)/len(f_hits)*100:>5.0f}%  "
          f"{sum(c_hits)/len(c_hits)*100:>5.0f}%  {agree:>3}/{args.sample_size}   "
          f"{''.join(map(str, f_hits))} / {''.join(map(str, c_hits))}", flush=True)

print(f"\n=== [{args.tag}] PART 2: paper Figure-3 protocol (chunked, n={args.sample_size}, "
      f"equidistant, 5 digits) ===", flush=True)
print(f"{'req_len':>9} {'act_len':>9} {'acc':>7}   per-sample (depth 0 -> deepest)", flush=True)
summary = {}
per_depth = [0] * args.sample_size
per_depth_n = [0] * args.sample_size
for req_len in args.sweep_lengths:
    prompts, passkeys = generator(req_len, args.sample_size, 'equidistant')
    hits = []
    for i, (prompt, pass_key) in enumerate(zip(prompts, passkeys)):
        logits = run_chunked(prompt + pass_key, args.chunk_size)
        ok = is_correct(logits, pass_key)
        hits.append(int(ok))
        per_depth[i] += int(ok)
        per_depth_n[i] += 1
        del logits
    acc = sum(hits) / len(hits)
    summary[req_len] = acc
    print(f"{req_len:>9} {len(prompts[0]):>9} {acc*100:>6.1f}%   {''.join(map(str, hits))}", flush=True)

print(f"\n=== [{args.tag}] accuracy by passkey depth (all sweep lengths pooled) ===", flush=True)
for i in range(args.sample_size):
    print(f"  depth {i/(args.sample_size-1)*100:5.1f}%: {per_depth[i]}/{per_depth_n[i]}", flush=True)

print("\nDIAG_RESULT_START")
print(json.dumps({"tag": args.tag, "log_dir": args.log_dir, "acc_by_len": summary}))
print("DIAG_RESULT_END")
