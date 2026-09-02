"""PassKey Retrieval em comprimentos curtos, via forward unico.

Motivo de existir separado do passkey_kvcache.py: aquele usava prefill
de KV-cache em chunks e por isso exigia model.forward_chunk(), que so
bam_ssmax.py e cabam_ssmax.py implementavam -- nenhum dos cinco
baselines tinha. Esse caminho inteiro saiu em 02/09/2026 (o script, o
forward_chunk e os forward_cached de BAM e CABAM), porque nenhum numero
do paper passou por ele: todo eval sempre usou o forward unico. O
PasskeyEvaluator nativo do eval_utils avalia com model(input) e
portanto funciona com qualquer classe de modelo, sem tocar em codigo de
modelo nenhum.

O preco e o teto de comprimento: um forward unico precisa da sequencia
inteira na memoria. Para ALiBi/NoPE/RoPE, que passam pelo
flex_attention, isso escala em O(T). Para CoPE e DAPE nao: os dois
calculam os scores de atencao explicitamente para aplicar gates e MLP,
entao a matriz [b, h, T, T] e materializada e a memoria vira O(T^2).

Por isso o sweep e CRESCENTE, com o OOM tratado por comprimento em vez
de derrubar a corrida: cada modelo vai ate onde couber e reporta tudo
que mediu. O teto que cada um atinge e, ele proprio, um resultado.

Contexto de treino = 512 tokens, entao 16384 e 32x o comprimento visto
no treino -- e o paper do CoPE avalia em 1024.
"""

import argparse
import json
import random
import sys

import torch

import eval_utils as ev_mod
from eval_utils import Evaluator, PasskeyEvaluator

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, required=True)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dtype', type=str, default='bfloat16')
parser.add_argument('--lengths', type=int, nargs='+',
                    default=[1024, 2048, 4096, 8192, 16384])
parser.add_argument('--sample_size', type=int, default=20)
parser.add_argument('--pred_digits', type=int, default=5)
parser.add_argument('--compile', action=argparse.BooleanOptionalAction,
                    default=False)
parser.add_argument('--seed', type=int, default=1337)
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

# load_model concatena o path com 'args.json' sem separador, entao a
# barra final e obrigatoria. Ja custou uma rodada inteira de jobs.
log_dir = args.log_dir if args.log_dir.endswith('/') else args.log_dir + '/'

ptdtype = {'float32': torch.float32,
           'bfloat16': torch.bfloat16,
           'float16': torch.float16}[args.dtype]

# Sem torch.compile, flex_attention cai no fallback eager, que
# materializa [b, h, T, T] -- ou seja, ALiBi/NoPE/RoPE ficam O(T^2) e
# estouram em 16k, exatamente o que o compile evita. O default segue
# False porque CoPE e DAPE nao ganham nada (a matriz T x T e explicita
# no forward deles) e compilar so adiciona custo.
#
# O sweep passa por varios comprimentos e cada um dispara um recompile.
# O limite default do dynamo e 8: passando disso ele desiste e volta
# pro eager em silencio, o que traria o O(T^2) de volta sem aviso
# nenhum no log. Por isso o limite sobe junto com o compile.
if args.compile:
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = max(
        64, 4 * len(args.lengths))

ev = Evaluator(device=args.device, compile=args.compile, dtype=args.dtype,
               perplexity_dataset_dirs=[], passkey_samplings=[],
               attn_chunk=args.attn_chunk, attn_ref_len=args.attn_ref_len,
               logits_chunk=args.logits_chunk)
model = ev.load_model(log_dir)
model.eval()

# O Evaluator guarda self.compile mas so aplica dentro do proprio
# .evaluate(), que este script nao usa -- ele chama load_model() direto.
# Entao passar compile=True para o construtor nao faz nada aqui; o
# torch.compile tem que ser explicito.
# dynamic=False e obrigatorio, nao preferencia: com shapes dinamicos o
# inductor quebra o lowering do flex_attention no RoPE
# (InductorError: CantSplit ... (s27 + 127)//128), porque nao consegue
# dividir a grade de blocos com o simbolo. Estatico compila uma vez por
# comprimento -- mais lento, e por isso o cache_size_limit acima.
if args.compile:
    model = torch.compile(model, dynamic=False)

print(f'=== passkey curto | {log_dir} | device {args.device} | '
      f'compile={args.compile} | seed={args.seed} ===', flush=True)

rows = []
for seq_len in sorted(args.lengths):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)
    try:
        evaluator = PasskeyEvaluator(
            seq_lens=[seq_len],
            device=args.device,
            pred_digits=args.pred_digits,
            sample_size=args.sample_size,
            # Este script instancia o PasskeyEvaluator direto, entao o
            # logits_chunk que foi para o Evaluator NAO chega aqui sozinho.
            # Sem esta linha a flag vira no-op silencioso e o eval materializa
            # os logits inteiros -- 64 GiB em 512k.
            logits_chunk=args.logits_chunk,
        )
        # O gerador sorteia a passkey com random.randint sem seed
        # (eval_utils.py), entao rodadas diferentes usam numeros
        # diferentes e a acurácia varia de 2 a 3 amostras em 20. As
        # POSICOES ja eram deterministicas (linspace equidistante); o que
        # faltava era o valor.
        #
        # A seed depende do comprimento para que cada seq_len tenha seu
        # conjunto fixo de passkeys, independente de quais outros
        # comprimentos vieram antes no sweep. Assim rodar um subconjunto
        # da grade reproduz exatamente os mesmos numeros.
        #
        # Semeado aqui e nao antes do PasskeyEvaluator porque o __init__
        # dele ja consome um randint ao medir o comprimento real.
        random.seed(args.seed + seq_len)
        torch.manual_seed(args.seed + seq_len)

        with torch.autocast(device_type='cuda', dtype=ptdtype):
            _, result = evaluator.evaluate(model, verbose=False)
        acc = result['accs'][0]
        peak = torch.cuda.max_memory_allocated(args.device) / 2**30
        rows.append({'seq_len': seq_len, 'acc': acc, 'peak_GiB': round(peak, 2)})
        print(f'RESULT seq_len={seq_len} acc={acc*100:.1f}% '
              f'peak={peak:.2f}GiB', flush=True)
    except Exception as exc:
        # Com torch.compile o OOM sobe embrulhado por dynamo/inductor,
        # entao checar o tipo nao basta -- tem que varrer a cadeia de
        # causas. Sem isso um teto de memoria seria rotulado ERRO e
        # passaria por bug de codigo em vez de resultado.
        chain, e = [], exc
        while e is not None and e not in chain:
            chain.append(e); e = e.__cause__ or e.__context__
        oom = any(isinstance(e, torch.cuda.OutOfMemoryError)
                  or 'out of memory' in str(e).lower() for e in chain)
        if not oom:
            print(f'ERRO seq_len={seq_len}: {type(exc).__name__}: {exc}',
                  flush=True)
            break
        # Nao adianta tentar os maiores: a memoria cresce monotonicamente
        # com o comprimento. Para aqui e mantem o que ja foi medido.
        print(f'OOM seq_len={seq_len} -- teto deste modelo atingido, '
              f'parando o sweep', flush=True)
        model.to('cpu')
        torch.cuda.empty_cache()
        break

print('PASSKEY_RESULT_START', flush=True)
print(json.dumps({'log_dir': log_dir,
                  'sample_size': args.sample_size,
                  'seed': args.seed,
                  'compiled': args.compile,
                  'results': rows}), flush=True)
print('PASSKEY_RESULT_END', flush=True)
