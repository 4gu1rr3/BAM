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
parser.add_argument('--logits_chunk', type=int, default=0,
                    help='projeta a saida em blocos de N posicoes (return_hidden). '
                         'Necessario acima de ~128k: os logits [b,T,32768] em fp32 '
                         'sao 64 GiB em 512k. 0 = logits inteiros de uma vez')
parser.add_argument('--attn_chunk', type=int, default=0,
                    help='CoPE/DAPE: linhas da matriz de atencao por bloco no '
                         'forward de inferencia. 0 = matriz inteira (comportamento '
                         'historico). Ver models/chunked_attn.py')
parser.add_argument('--attn_ref_len', type=int, default=0,
                    help='CoPE/DAPE: em vez de um bloco fixo, escolhe o bloco para '
                         'manter o pico de memoria no nivel deste comprimento. '
                         'Comprimentos <= a ele nao sao chunkados, entao saem '
                         'identicos aos ja medidos. 0 = desligado')
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
    attn_chunk=args.attn_chunk,
    attn_ref_len=args.attn_ref_len,
    logits_chunk=args.logits_chunk,
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
