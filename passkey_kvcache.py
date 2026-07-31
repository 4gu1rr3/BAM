import argparse
import time
import torch
import torch._dynamo

from eval_utils import Evaluator, PromptGenerator

# The KV-cache grows every chunk, so each chunk has a distinct (Q_LEN, KV_LEN)
# shape -> a distinct compiled graph. Sweeping up to 512k tokens in 16k
# chunks means dozens of distinct shapes; the default cache_size_limit (8)
# would silently stop compiling new ones and fall back to eager flex_attention,
# which is exactly the unfused/OOM-prone path we're trying to avoid.
torch._dynamo.config.cache_size_limit = 128

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='logs/l12/cabam/version_04/')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dtype', type=str, default='bfloat16')
parser.add_argument('--chunk_size', type=int, default=16384)
parser.add_argument('--pred_digits', type=int, default=5)
parser.add_argument('--sample_size', type=int, default=5)
parser.add_argument('--lengths', type=int, nargs='+', default=None,
                     help='explicit target sequence lengths to test; overrides --linspace_end/--linspace_steps')
parser.add_argument('--linspace_end', type=int, default=512000)
parser.add_argument('--linspace_steps', type=int, default=11)
parser.add_argument('--validate_only', action=argparse.BooleanOptionalAction, default=False,
                     help='only run the correctness check against a known-good length, skip the sweep')
parser.add_argument('--shard_id', type=int, default=0,
                     help='when running multiple GPUs in parallel, this process handles samples[shard_id::num_shards] of each length')
parser.add_argument('--num_shards', type=int, default=1)
args = parser.parse_args()

device = args.device
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]

ev = Evaluator(device=device, compile=False, dtype=args.dtype, perplexity_dataset_dirs=[], passkey_samplings=[])
model = ev.load_model(args.log_dir)
model.to(device)
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
        chunk = torch.tensor(total_ids[start:end], device=device).unsqueeze(0)
        is_last = (end == n)
        with torch.autocast(device_type='cuda', dtype=ptdtype):
            out, kv_caches = compiled_chunk(chunk, start, kv_caches, is_last, logits_tail if is_last else None)
        if is_last:
            logits = out
        start = end
    return logits


def check_prompt(prompt, pass_key, chunk_size, verbose=True):
    total_ids = prompt + pass_key
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    logits = run_chunked(total_ids, chunk_size)
    torch.cuda.synchronize()
    dt = time.time() - t0
    pred = logits.argmax(-1)
    pred_pass_key = list(pred[0, -args.pred_digits - 1:-1].cpu())
    correct = (pred_pass_key == pass_key[1:])
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    if verbose:
        print(f"seq_len={len(prompt)} chunk_size={chunk_size} -> {'CORRECT' if correct else 'WRONG'} "
              f"({dt:.2f}s, peak={peak:.2f} GiB)", flush=True)
    return correct, peak


# --- Step 1: correctness validation against a length we already trust from
# the non-chunked eval (job 792: 100% accuracy at seq_len=8178, sample_size=20).
# If chunking/caching has a position-offset bug, this is where it would show up.
# Only shard 0 bothers -- both shards load the identical checkpoint/code, so a
# pass on shard 0 is as good as a pass on every shard.
if args.shard_id == 0:
    print("=== Validating chunked+cached path against known-good non-chunked results ===", flush=True)
    for seq_len, chunk_size in [(8178, 4096), (8178, 2048), (32754, 16384)]:
        prompts, passkeys = generator(seq_len, 1, 'equidistant')
        check_prompt(prompts[0], passkeys[0], chunk_size)

if args.validate_only:
    raise SystemExit(0)

print(f"\n=== Passkey retrieval sweep (chunked KV-cache) shard {args.shard_id}/{args.num_shards} ===", flush=True)
if args.lengths:
    target_lengths = args.lengths
else:
    target_lengths = torch.linspace(0, args.linspace_end, args.linspace_steps).int().tolist()[1:]

for length in target_lengths:
    prompts, passkeys = generator(length, args.sample_size, 'equidistant')
    my_idx = list(range(args.shard_id, args.sample_size, args.num_shards))
    correct_count = 0
    n_done = 0
    peak_last = None
    oom = False
    for i in my_idx:
        prompt, pass_key = prompts[i], passkeys[i]
        try:
            correct, peak = check_prompt(prompt, pass_key, args.chunk_size, verbose=False)
            correct_count += int(correct)
            n_done += 1
            peak_last = peak
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"seq_len~{length} shard={args.shard_id} sample_idx={i} -> OOM", flush=True)
            oom = True
            break
    if oom:
        print(f"RESULT_PARTIAL seq_len~{length} shard={args.shard_id}/{args.num_shards}: OOM, stopping sweep", flush=True)
        break
    print(f"RESULT_PARTIAL seq_len~{length} shard={args.shard_id}/{args.num_shards}: correct={correct_count} total={n_done} peak={peak_last:.2f}", flush=True)

print("DONE", flush=True)
