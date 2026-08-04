"""Extends the passkey curve past 32k with per-depth detail, and confirms the
chunked path is chunk-size invariant in the regime where accuracy actually
drops (job 804's sweep only validated chunking at depth 0, at 8k/32k).
"""
import argparse
import json
import random
from contextlib import nullcontext

import torch
import torch._dynamo

from eval_utils import Evaluator, PromptGenerator

torch._dynamo.config.cache_size_limit = 256

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='logs/l12/bam_ssmax/version_02/')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dtype', type=str, default='bfloat16')
parser.add_argument('--sample_size', type=int, default=20)
parser.add_argument('--pred_digits', type=int, default=5)
parser.add_argument('--chunk_size', type=int, default=16384)
parser.add_argument('--invariance_len', type=int, default=51200)
parser.add_argument('--invariance_chunks', type=int, nargs='+', default=[8192, 16384, 32768])
parser.add_argument('--sweep_lengths', type=int, nargs='+', default=[51200, 102400, 204800])
parser.add_argument('--tag', type=str, default='BAM')
parser.add_argument('--prompt_seed', type=int, default=None,
                    help='seed PromptGenerator so two runs see byte-identical prompts/passkeys')
args = parser.parse_args()

if args.prompt_seed is not None:
    random.seed(args.prompt_seed)

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
# autocast only accepts a reduced-precision dtype; float32 means "run as-is".
def amp():
    return nullcontext() if ptdtype is torch.float32 else torch.autocast(device_type='cuda', dtype=ptdtype)

ev = Evaluator(device=args.device, compile=False, dtype=args.dtype,
               perplexity_dataset_dirs=[], passkey_samplings=[])
model = ev.load_model(args.log_dir)
model.to(args.device)
model.eval()
n_layers = model.n_layers

compiled_chunk = torch.compile(model.forward_chunk)
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
        with amp():
            out, kv_caches = compiled_chunk(chunk, start, kv_caches, is_last,
                                            logits_tail if is_last else None)
        if is_last:
            logits = out
        start = end
    del kv_caches
    torch.cuda.empty_cache()
    return logits


def is_correct(logits, pass_key):
    pred = logits.argmax(-1)
    return list(pred[0, -args.pred_digits - 1:-1].cpu()) == pass_key[1:]


print(f"=== [{args.tag}] chunk-size invariance at ~{args.invariance_len} tokens, "
      f"all {args.sample_size} depths ===", flush=True)
prompts, passkeys = generator(args.invariance_len, args.sample_size, 'equidistant')
per_chunk = {}
for cs in args.invariance_chunks:
    hits = [int(is_correct(run_chunked(p + k, cs), k)) for p, k in zip(prompts, passkeys)]
    per_chunk[cs] = hits
    print(f"  chunk_size={cs:>6}: {sum(hits)/len(hits)*100:>5.1f}%  {''.join(map(str, hits))}", flush=True)
ref = per_chunk[args.invariance_chunks[0]]
print(f"  identical across chunk sizes: {all(v == ref for v in per_chunk.values())}", flush=True)

print(f"\n=== [{args.tag}] long sweep (chunked, n={args.sample_size}, equidistant, "
      f"chunk_size={args.chunk_size}) ===", flush=True)
print(f"{'req_len':>9} {'act_len':>9} {'acc':>7}   per-sample (depth 0 -> deepest)", flush=True)
summary = {}
for req_len in args.sweep_lengths:
    prompts, passkeys = generator(req_len, args.sample_size, 'equidistant')
    hits = []
    for prompt, pass_key in zip(prompts, passkeys):
        hits.append(int(is_correct(run_chunked(prompt + pass_key, args.chunk_size), pass_key)))
    summary[req_len] = sum(hits) / len(hits)
    print(f"{req_len:>9} {len(prompts[0]):>9} {summary[req_len]*100:>6.1f}%   "
          f"{''.join(map(str, hits))}", flush=True)

print("\nDIAG_RESULT_START")
print(json.dumps({"tag": args.tag, "log_dir": args.log_dir,
                  "invariance": {str(k): v for k, v in per_chunk.items()},
                  "acc_by_len": summary}))
print("DIAG_RESULT_END")
