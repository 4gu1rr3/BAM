"""Perplexidade pelo mesmo caminho que o paper do BAM usou:
Evaluator.evaluate(evals=['perplexity']).

Diferenca central em relacao ao que a gente vinha fazendo com o
eval_baseline_ppl.sh: o PerplexityEvaluator roda UM UNICO seq_len e
reporta a perplexidade por janela dentro dele
(positions = arange(nwindows)*window_size + window_size). Ou seja, o eixo
"Sequence length" das Figuras 8 e 9 e POSICAO dentro de uma sequencia de
32768 -- nao avaliacoes separadas em T diferentes.

Isso importa porque o pool da Wikipedia e filtrado por
`len(artigo) >= seq_len+1`: rodando T diferentes em jobs separados, cada
T via um conjunto de artigos diferente e as curvas nao eram comparaveis
entre si (foi o que produziu ALiBi 22.7 em T=16384 contra 12.1 em
T=32768). Com um seq_len so, o pool e um so.

Os defaults do Evaluator sao os do paper: seq_len=32768,
window_size=1024, ntokens=3_932_160, wiki_articles=256. Note que 256 e
8x o que o eval_baseline_ppl.sh usava (32), o que reduz bastante a
variancia da linha de Wikipedia.
"""

import argparse
import json

import torch

from eval_utils import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, required=True)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dtype', type=str, default='bfloat16')
parser.add_argument('--seq_len', type=int, default=32768)
parser.add_argument('--window_size', type=int, default=1024)
parser.add_argument('--ntokens', type=int, default=3_932_160)
parser.add_argument('--wiki_articles', type=int, default=256)
parser.add_argument('--datasets', type=str, nargs='+',
                    default=['10B', 'wikipedia'])
parser.add_argument('--compile', action=argparse.BooleanOptionalAction,
                    default=False)
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

log_dir = args.log_dir if args.log_dir.endswith('/') else args.log_dir + '/'

ev = Evaluator(
    device=args.device,
    dtype=args.dtype,
    compile=args.compile,
    passkey_samplings=[],           # so perplexidade
    perplexity_dataset_dirs=args.datasets,
    perplexity_seq_len=args.seq_len,
    perplexity_window_size=args.window_size,
    perplexity_ntokens=args.ntokens,
    perplexity_wiki_articles=args.wiki_articles,
    attn_chunk=args.attn_chunk,
    attn_ref_len=args.attn_ref_len,
    logits_chunk=args.logits_chunk,
)

print(f'=== perplexidade via Evaluator.evaluate | {log_dir} | {args.device} | '
      f'compile={args.compile} | seq_len={args.seq_len} | '
      f'window={args.window_size} | wiki_articles={args.wiki_articles} ===',
      flush=True)

out = ev.evaluate(log_dir, evals=['perplexity'])

for ds, res in out['perplexity'].items():
    nome = ds.replace('data/', '')
    print(f'RESULT dataset={nome} seq_len={res["seq_len"]} '
          f'ppl_inicio={res["perplexity"][0]:.2f} '
          f'ppl_fim={res["perplexity"][-1]:.2f}', flush=True)

print('PERPLEXITY_RESULT_START', flush=True)
print(json.dumps({'log_dir': log_dir,
                  'compiled': args.compile,
                  'via': 'Evaluator.evaluate',
                  'results': {k.replace('data/', ''): v
                              for k, v in out['perplexity'].items()}}),
      flush=True)
print('PERPLEXITY_RESULT_END', flush=True)
