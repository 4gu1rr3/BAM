import argparse
import json
import torch

from eval_utils import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='logs/l12/cabam/version_04/')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='wikipedia', choices=['wikipedia', '10B'])
parser.add_argument('--seq_len', type=int, default=32768)
parser.add_argument('--window_size', type=int, default=1024)
parser.add_argument('--wiki_articles', type=int, default=32)
parser.add_argument('--ntokens', type=int, default=3932160)
parser.add_argument('--dtype', type=str, default='bfloat16')
parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

ev = Evaluator(
    device=args.device,
    compile=args.compile,
    dtype=args.dtype,
    passkey_samplings=[],
    perplexity_dataset_dirs=[args.dataset],
    perplexity_seq_len=args.seq_len,
    perplexity_window_size=args.window_size,
    perplexity_wiki_articles=args.wiki_articles,
    perplexity_ntokens=args.ntokens,
)

results = ev.evaluate(args.log_dir, evals=['perplexity'])
perp = list(results['perplexity'].values())[0]

print("PERPLEXITY_RESULT_START")
print(json.dumps({
    "dataset": args.dataset,
    "seq_len": args.seq_len,
    "window_size": args.window_size,
    "positions": perp["positions"],
    "perplexity": perp["perplexity"],
}))
print("PERPLEXITY_RESULT_END")
