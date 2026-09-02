import os
import json
import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from time import time
from contextlib import nullcontext

from models.sinusoidal import SinusoidalModelArgs, SinusoidalTransformer
from models.sinusoidal_ssmax import SinusoidalSSMaxModelArgs, SinusoidalSSMaxTransformer
from models.rotary import RotaryModelArgs, RotaryTransformer
from models.rotary_local import LocalRotaryTransformer, LocalRotaryModelArgs
from models.rotary_ssmax import RotarySSMaxModelArgs, RotarySSMaxTransformer
from models.alibi import ALiBiModelArgs, ALiBiTransformer
from models.alibi_ssmax import ALiBiSSMaxModelArgs, ALiBiSSMaxTransformer
from models.bam import BATransformer, BATModelArgs
from models.bam_ssmax import SSMaxBATransformer, SSMaxBATModelArgs
from models.nope import NoPEModelArgs, NoPETransformer
from models.nope_ssmax import NoPESSMaxModelArgs, NoPESSMaxTransformer
from models.cabam_ssmax import SSMaxBATransformer as CABAMTransformer, SSMaxBATModelArgs as CABAMModelArgs
from models.dape_alibi import DAPEALiBiTransformer, DAPEALiBiModelArgs
from models.cope import CoPETransformer, CoPEModelArgs


class PasskeyEvaluator:
    def __init__(self, seq_lens, device='cpu', pred_digits=5, preffix_digits=0, sampling='equidistant', patience=float('inf'), sample_size=10, seq_batch_size=None, seed=None, logits_chunk=0):
        self.generator = PromptGenerator(digits=pred_digits+preffix_digits)
        # seed=None preserva o comportamento historico (passkey sorteada
        # sem seed a cada rodada), que e o que o paper do BAM usou.
        self.seed = seed
        self.seq_lens = [len(self.generator(seq_len)[0][0]) for seq_len in seq_lens]
        self.device = device
        self.pred_digits = pred_digits
        self.preffix_digits = preffix_digits
        self.sampling = sampling
        self.patience = patience
        self.sample_size = sample_size
        self.seq_batch_size = seq_batch_size
        # >0 projeta a saida em blocos de posicoes (models: return_hidden).
        # 0 = comportamento historico, logits inteiros de uma vez.
        self.logits_chunk = logits_chunk
    
    @torch.inference_mode()
    def evaluate(self, model, verbose=True, patience=None, prev_results=None):
        # A seed PRECISA entrar na chave: sem ela, reavaliar o mesmo
        # checkpoint com outra seed bate no curto-circuito abaixo e
        # devolve os numeros antigos em cache, sem rodar nada e sem
        # nada no log denunciando. Quando seed=None a chave fica
        # identica a original, entao results.json ja existentes
        # continuam validos.
        result_config = str((self.sampling, self.sample_size, self.pred_digits, self.preffix_digits))
        if self.seed is not None:
            result_config = str((self.sampling, self.sample_size, self.pred_digits, self.preffix_digits, self.seed))
        if prev_results is not None:
            if result_config in prev_results:
                if set(self.seq_lens).issubset(set(prev_results[result_config]['seq_lens'])):
                    return prev_results, prev_results[result_config]
            results = prev_results
            results[result_config] = {}
        else:
            results = {}
            results[result_config] = {}
        results[result_config]['seq_lens'] = []
        results[result_config]['accs'] = []

        model.to(self.device)
        patience = patience or self.patience
        results[result_config].setdefault('peak_GiB', [])
        for seq_len in self.seq_lens:
            print(f"                0/0 correct", end='\r')
            correct = 0
            # Semeia por comprimento (e nao uma vez no inicio) para que
            # cada seq_len tenha seu conjunto fixo de passkeys, qualquer
            # que seja a ordem ou o subconjunto avaliado.
            if self.seed is not None:
                random.seed(self.seed + seq_len)
                torch.manual_seed(self.seed + seq_len)
            prompts, passkeys = self.generator(seq_len, self.sample_size, self.sampling)
            start = time()
            if 'cuda' in str(self.device):
                torch.cuda.reset_peak_memory_stats(self.device)
            try:
                for i, (prompt, pass_key) in enumerate(zip(prompts, passkeys)):
                    if not len(prompt) == seq_len:
                        raise ValueError(f"Prompt length {len(prompt)} does not match expected length {seq_len}")
                    model_input = torch.tensor(prompt+pass_key).unsqueeze(0).to(self.device)
                    if self.logits_chunk:
                        # O passkey le so as ultimas pred_digits+1 posicoes, entao
                        # projetar a sequencia inteira e desperdicio puro: em 512k
                        # sao 64 GiB de logits para usar 6 linhas. Com o estado
                        # escondido em maos, projeta-se so a cauda.
                        h = model(model_input, return_hidden=True)
                        tail = model.output(h[:, -self.pred_digits-1:]).float()
                        pred_pass_key = tail[0, :-1].argmax(-1).cpu()
                        del h, tail
                    else:
                        output = model(model_input).argmax(-1)
                        pred_pass_key = output[0, -self.pred_digits-1:-1].cpu()
                    if (list(pred_pass_key) == pass_key[self.preffix_digits+1:]):
                        correct += 1
                    end = time()
                    print(f"                seq_len: {len(prompt)}, acc: {correct}/{i+1} of {self.sample_size} took {(end-start)/(i+1):.2f}s", end='\r')
            except Exception as exc:
                # A memoria cresce monotonicamente com o comprimento, entao
                # um OOM aqui e o teto do modelo: para o sweep e devolve o
                # que ja foi medido, em vez de perder a corrida inteira.
                # Sob torch.compile o OOM sobe embrulhado por
                # dynamo/inductor, por isso a varredura da cadeia de causas.
                chain, e = [], exc
                while e is not None and e not in chain:
                    chain.append(e); e = e.__cause__ or e.__context__
                if not any(isinstance(e, torch.cuda.OutOfMemoryError)
                           or 'out of memory' in str(e).lower() for e in chain):
                    raise
                print(f"OOM em seq_len={seq_len} -- teto do modelo, parando o sweep"
                      f"                                        ")
                torch.cuda.empty_cache()
                break
            peak = (torch.cuda.max_memory_allocated(self.device) / 2**30
                    if 'cuda' in str(self.device) else 0.0)
            results[result_config]['seq_lens'].append(seq_len)
            results[result_config]['accs'].append(correct/self.sample_size)
            results[result_config]['peak_GiB'].append(round(peak, 2))
            if verbose:
                print(f"seq_len: {len(prompt)}, acc: {correct/self.sample_size*100:04.1f}%                                                                                        ")
            if correct == 0:
                patience -= 1
            else:
                patience = self.patience
            if patience == 0:
                print(f"Early stopping at seq_len: {seq_len}")
                break
        model.to('cpu')
        return results, results[result_config]
        


class PromptGenerator:
    def __init__(self, digits=8, tokenizer='mistralai/Mistral-7B-Instruct-v0.3'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.task_description   = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
        self.garbage_inf        = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
        self.information_line   = "The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
        self.final_question     = "What is the pass key? The pass key is"

        self.task_description_tokens    = self.tokenizer(self.task_description, add_special_tokens=False)['input_ids']
        self.garbage_inf_tokens         = self.tokenizer(self.garbage_inf, add_special_tokens=False)['input_ids']
        self.final_question_tokens      = self.tokenizer(self.final_question, add_special_tokens=False)['input_ids']

        self.task_description_len   = len(self.task_description_tokens)
        self.garbage_inf_len        = len(self.garbage_inf_tokens)
        self.information_line_len   = len(self.tokenizer(self.information_line.format(pass_key=10**digits-1), add_special_tokens=False)['input_ids'])
        self.final_question_len     = len(self.final_question_tokens)
        self.n_digits               = digits

    def __generate_prompt(self, n_garbage, n_garbage_prefix):
        n_garbage_suffix = n_garbage - n_garbage_prefix

        pass_key = random.randint(10**(self.n_digits-1), 10**self.n_digits-1)
        information_line = self.information_line.format(pass_key=pass_key)

        information_tokens  = self.tokenizer(information_line, add_special_tokens=False)['input_ids']
        passkey_tokens      = self.tokenizer(' ' + str(pass_key), add_special_tokens=False)['input_ids']

        garbage_prefix = self.garbage_inf_tokens * n_garbage_prefix
        garbage_suffix = self.garbage_inf_tokens * n_garbage_suffix

        prompt = self.task_description_tokens + garbage_prefix + information_tokens + garbage_suffix + self.final_question_tokens
        return prompt, passkey_tokens
    
    def generate_prompt(self, length, sample_size=1, sampling='random'):
        prompts = []
        passkeys = []
        n_garbage = (length -self.task_description_len -self.information_line_len -self.final_question_len) // self.garbage_inf_len
        n_garbage = max(n_garbage, 0)
        if sampling == 'random':
            passkey_positions = random.choices(range(n_garbage+1), k=sample_size)
        elif sampling == 'equidistant':
            passkey_positions = torch.linspace(0, n_garbage, sample_size).long().tolist()
        elif sampling == 'beginning':
            passkey_positions = [0] * sample_size
        elif sampling == 'end':
            passkey_positions = [n_garbage] * sample_size
        else:
            raise ValueError(f"Unknown sampling method: {sampling}")
        for passkey_position in passkey_positions:
            prompt, passkey_tokens = self.__generate_prompt(n_garbage, passkey_position)
            prompts.append(prompt)
            passkeys.append(passkey_tokens)
        return prompts, passkeys
    
    def __call__(self, length, sample_size=1, sampling='random'):
        return self.generate_prompt(length, sample_size, sampling)


    




def _project_logits_chunked(model, h, chunk):
    """Projeta o estado escondido em blocos de posicoes, devolvendo argmax.

    Existe porque os logits [b, T, vocab] em fp32 sao 64 GiB em T=512k com
    vocab 32768 -- maiores que qualquer placa disponivel -- enquanto o
    estado escondido nos mesmos 512k e 1,6 GiB. Projetando um bloco de
    posicoes por vez e reduzindo na hora, o pico vira chunk x vocab.

    A projecao e linear e o argmax e por posicao, entao cortar em posicoes
    e exato: cada linha da saida depende so da linha correspondente de h.
    """
    out = None
    T = h.shape[1]
    for i0 in range(0, T, chunk):
        i1 = min(i0 + chunk, T)
        ids = model.output(h[:, i0:i1]).float().argmax(-1)
        if out is None:
            out = torch.empty(h.shape[0], T, dtype=ids.dtype, device=ids.device)
        out[:, i0:i1] = ids
        del ids
    return out


def _ce_chunked(model, h, targets, cross_entropy, chunk):
    """Cross entropy por token, projetando em blocos de posicoes.

    Mesmo motivo do _project_logits_chunked. `targets` sao os tokens
    deslocados de um, ja no device de h; devolve a loss por posicao na CPU,
    que e o que o chamador acumula.
    """
    T = targets.shape[0]
    losses = torch.empty(T, dtype=torch.float32)
    for i0 in range(0, T, chunk):
        i1 = min(i0 + chunk, T)
        logits = model.output(h[0, i0:i1]).float()
        losses[i0:i1] = cross_entropy(logits, targets[i0:i1]).cpu()
        del logits
    return losses


class PerplexityEvaluator:
    def __init__(self, dataset_dir, seq_len, ntokens, window_size=512, device='cpu', wiki_articles=512, seq_batch_size=None, logits_chunk=0):
        self.dataset_dir    = os.path.join('data', dataset_dir)
        self.seq_len        = seq_len
        self.ntokens        = ntokens
        self.window_size    = window_size
        self.nwindows       = int(seq_len // window_size)
        self.cross_entropy  = torch.nn.CrossEntropyLoss(reduction='none')
        self.logits_chunk   = logits_chunk
        self.seq_batch_size = seq_batch_size
        self.device         = device

        assert seq_len % window_size == 0, f"seq_len {seq_len} must be divisible by window_size {window_size}"

        if dataset_dir == 'wikipedia':
            tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
            # Streaming avoids materializing/downloading the full ~20GB split
            # just to find a handful of long-enough articles.
            dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
            tokens = []
            batch = []
            scanned = 0
            def flush(batch):
                if not batch:
                    return
                input_ids = tokenizer(batch, add_special_tokens=True)['input_ids']
                for input_id in input_ids:
                    if len(input_id) >= seq_len+1:
                        tokens.append(input_id[:seq_len+1])
                        print(f"Loaded {len(tokens)}/{wiki_articles} articles >= {seq_len+1} tokens (scanned {scanned})", flush=True)
                    if len(tokens) >= wiki_articles:
                        break
            for example in dataset:
                batch.append(example['text'])
                scanned += 1
                if scanned % 2000 == 0:
                    print(f"...scanned {scanned} wikipedia articles, found {len(tokens)}/{wiki_articles} long enough", flush=True)
                if len(batch) >= 64:
                    flush(batch)
                    batch = []
                if len(tokens) >= wiki_articles:
                    break
            if len(tokens) < wiki_articles:
                flush(batch)
            self.tokens = torch.tensor(tokens).long()

                
        else:
            files = sorted(os.listdir(self.dataset_dir))
            assert len(files) > 0, f"did not find any files that match the pattern {self.dataset_dir}"
            filename = os.path.join(self.dataset_dir, files[0])

            dataset = torch.load(filename)
            tokens     = dataset['input_ids'][:ntokens+1].long()

            # Identify EOS and BOS tokens 
            tokenizer      = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
            eos_token_id   = tokenizer.eos_token_id
            bos_token_id   = tokenizer.bos_token_id

            # Remove EOS and BOS tokens from the dataset
            eos_idxs = tokens == eos_token_id
            bos_idxs = tokens == bos_token_id
            idxs = eos_idxs | bos_idxs
            tokens = tokens[~idxs]

            nbatches = len(tokens) // (seq_len+1)
            tokens = tokens[:nbatches * (seq_len + 1)]
            self.tokens = tokens.reshape(nbatches, seq_len+1)

    @torch.inference_mode()
    def evaluate(self, model, prev_results=None):
        model.to(self.device)
        if prev_results is not None:
            for result in prev_results:
                if result['window_size'] == self.window_size and result['seq_len'] == self.seq_len and result['ntokens'] == self.ntokens and result['dataset_dir'] == self.dataset_dir:
                    return prev_results, result
            results = prev_results
            results.append({})
        else:
            results = [{}]
            

        entropies = torch.zeros(self.nwindows)
        entropies_sqrd = torch.zeros(self.nwindows)
        for i, tokens in enumerate(self.tokens):
            print(f'{i+1}/{len(self.tokens)}   {(entropies.mean()/(i+1)).exp()}')

            if self.logits_chunk:
                h = model(tokens.unsqueeze(0).to(self.device), return_hidden=True)
                targets = tokens[1:].to(self.device)
                loss = _ce_chunked(model, h[:, :-1], targets, self.cross_entropy,
                                   self.logits_chunk).reshape(-1, self.window_size)
                del h, targets
            else:
                output = model(tokens.unsqueeze(0).to(self.device)).cpu()
                loss = self.cross_entropy(output[0, :-1, :], tokens[1:]).reshape(-1, self.window_size)

            entropies += loss.mean(dim=-1).cpu().detach()
            entropies_sqrd += (loss**2).mean(dim=-1).cpu().detach()
        entropies = entropies/len(self.tokens)
        entropies_sqrd = entropies_sqrd/len(self.tokens)
        entropies_std = torch.sqrt(entropies_sqrd - entropies**2)
        perplexity = torch.exp(entropies)

        positions = torch.arange(self.nwindows) * self.window_size + self.window_size
        
        results[-1]['window_size'] = self.window_size
        results[-1]['seq_len'] = self.seq_len
        results[-1]['ntokens'] = self.ntokens
        results[-1]['dataset_dir'] = self.dataset_dir
        results[-1]['perplexity'] = perplexity.tolist()
        results[-1]['positions'] = positions.tolist()
        model.to('cpu')
        return results, results[-1]
    
class Evaluator:
    def __init__(self,
                 passkey_seq_lens=None,
                 passkey_sample_size=20,
                 passkey_pred_digits=5,
                 passkey_preffix_digits=0,
                 passkey_samplings=['equidistant'],
                 passkey_patience=float('inf'),
                 passkey_seed=None,
                 perplexity_dataset_dirs=['10B', 'wikipedia'],
                 perplexity_seq_len=32_768,
                 perplexity_ntokens=3_932_160,
                 perplexity_window_size=1024,
                 perplexity_wiki_articles=256,
                 seq_batch_size=512,
                 device='cpu',
                 compile=False,
                 dtype='bfloat16',
                 # Atencao por blocos de linhas no CoPE/DAPE (models/chunked_attn.py).
                 # Os outros modelos ignoram: passam pelo flex_attention e ja sao O(T).
                 attn_chunk=0,      # bloco fixo de linhas; 0 = desligado
                 attn_ref_len=0,    # bloco automatico: mantem o pico no nivel deste comprimento
                 # Projecao de saida em blocos de posicoes. Necessaria acima de
                 # ~128k: os logits [b, T, 32768] em fp32 sao 64 GiB em 512k.
                 logits_chunk=0,
                 ):
        self.passkey_seq_lens = passkey_seq_lens or [0, 128, 256, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768]
        self.passkey_evaluators = []
        self.device = device
        self.compile = compile
        self.attn_chunk = attn_chunk
        self.attn_ref_len = attn_ref_len
        self.logits_chunk = logits_chunk

        # set up a context manager following the desired dtype and device
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        self.ctx = torch.autocast(device_type=device_type, dtype=ptdtype) if (device_type == "cuda") else nullcontext()

        for sampling in passkey_samplings:
            self.passkey_evaluators.append(PasskeyEvaluator(
                seq_lens=self.passkey_seq_lens,
                pred_digits=passkey_pred_digits,
                preffix_digits=passkey_preffix_digits,
                sampling=sampling,
                patience=passkey_patience,
                sample_size=passkey_sample_size,
                logits_chunk=logits_chunk,
                seq_batch_size=seq_batch_size,
                device=device,
                seed=passkey_seed,
            ))

        self.perplexity_evaluators = []
        for perplexity_dataset_dir in perplexity_dataset_dirs:
            self.perplexity_evaluators.append(PerplexityEvaluator(
                dataset_dir=perplexity_dataset_dir,
                seq_len=perplexity_seq_len,
                ntokens=perplexity_ntokens,
                window_size=perplexity_window_size,
                wiki_articles=perplexity_wiki_articles,
                logits_chunk=logits_chunk,
                seq_batch_size=seq_batch_size,
                device=device,
            ))

    @torch.inference_mode()        
    def evaluate(self, model_dir, evals=['passkey', 'perplexity']):
        model = self.load_model(model_dir)
        if self.compile:
            # dynamic=False e obrigatorio: com shape simbolico o inductor
            # quebra o lowering do flex_attention no RoPE
            # (InductorError: CantSplit). Estatico compila uma vez por
            # comprimento, por isso o cache_size_limit precisa subir junto.
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit,
                4 * len(self.passkey_seq_lens))
            model = torch.compile(model, dynamic=False)
        results = self.load_results(model_dir)


        with self.ctx:
            if 'passkey' in evals:
                passkey_results = {}
                for evaluator in self.passkey_evaluators:
                    results['passkey'], passkey_result = evaluator.evaluate(model, prev_results=results['passkey'])
                    passkey_results[evaluator.sampling] = passkey_result
                # Grava antes de comecar a perplexidade: sem isso, uma
                # falha la joga fora o passkey inteiro que ja rodou.
                with open(os.path.join(model_dir, 'results.json'), 'w') as f:
                    json.dump(results, f, indent=4)
            else:
                passkey_results = None

            if 'perplexity' in evals:
                perplexity_results = {}
                for evaluator in self.perplexity_evaluators:
                    results['perplexity'], perplexity_result = evaluator.evaluate(model, prev_results=results['perplexity'])
                    perplexity_results[evaluator.dataset_dir] = perplexity_result
            else:
                perplexity_results = None

        # Save the results
        with open(os.path.join(model_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        return {
            'passkey': passkey_results,
            'perplexity': perplexity_results,
        }
        


    def load_model(self, dir):
        import warnings
        with open(dir+'args.json') as f:
            args = json.load(f)

        ModelArgs, Transformer = {
            "rotary":       (RotaryModelArgs,       RotaryTransformer       ),
            "rotary_local": (LocalRotaryModelArgs,  LocalRotaryTransformer  ),
            "rotary_ssmax": (RotarySSMaxModelArgs,  RotarySSMaxTransformer  ),
            "sinusoidal":   (SinusoidalModelArgs,   SinusoidalTransformer   ),
            "sinusoidal_ssmax": (SinusoidalSSMaxModelArgs, SinusoidalSSMaxTransformer),
            "alibi":        (ALiBiModelArgs,        ALiBiTransformer        ),
            "alibi_ssmax":  (ALiBiSSMaxModelArgs,   ALiBiSSMaxTransformer   ),
            "bam":          (BATModelArgs,          BATransformer           ),
            "bam_ssmax":    (SSMaxBATModelArgs,     SSMaxBATransformer      ),
            "nope":         (NoPEModelArgs,         NoPETransformer         ),
            "nope_ssmax":   (NoPESSMaxModelArgs,    NoPESSMaxTransformer    ),
            "cabam":        (CABAMModelArgs,        CABAMTransformer        ),
            "dape_alibi":   (DAPEALiBiModelArgs,    DAPEALiBiTransformer    ),
            "cope":         (CoPEModelArgs,         CoPETransformer         ),
        }[args['args']['position_encoding']]
        model_dict = torch.load(os.path.join(dir, f'model.pt'))
        model = Transformer(ModelArgs(**args['model_args']))
        warn = False
        for k, v in model_dict.items():
            if (v.ndim==4) and ('attention.prior.theta' in k):
                model_dict[k] = v[0,:,0]
                warn = True
            elif (v.ndim==4) and ('attention.seq_scale' in k):
                model_dict[k] = v[:,:,0]
                warn = True
            if 'module.' in k or '_orig_mod.' in k:
                warn = True
        model_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in model_dict.items()}
        if warn:

            warnings.warn("The loaded model was trained on a previous version of the code. Some parameters have been adjusted to match the current version.")
        model.load_state_dict(model_dict)

        # Setado na instancia, nao no ModelArgs: os args.json dos checkpoints ja
        # treinados nao tem esses campos, e nenhum peso e afetado. Modelos que
        # nao leem os atributos simplesmente os ignoram.
        model.attn_chunk = self.attn_chunk
        model.attn_ref_len = self.attn_ref_len

        return model
    
    def load_results(self, dir):
        if os.path.exists(os.path.join(dir, 'results.json')):
            with open(os.path.join(dir, 'results.json')) as f:
                results = json.load(f)
        else:
            results = {
                'passkey': None,
                'perplexity': None,
            }
        return results
