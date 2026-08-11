"""Coleta a distribuicao dos parametros do prior do CABAM em funcao do
comprimento da sequencia.

No CABAM, alpha/beta saem de uma MLP sobre o conteudo do token, entao cada head
tem uma distribuicao de valores em vez de um escalar. Este script roda o modelo
em contextos de tamanhos crescentes (prefill em chunks + KV-cache) e acumula,
por (comprimento, camada, head):

  * histograma de beta e de log(alpha)  -- para os plots de densidade
  * quantis, media, desvio
  * fracao de tokens com beta < 0 e beta < -0.6
  * media de beta e fracao negativa por faixa de posicao dentro da sequencia

Os valores brutos nao cabem em disco em 100k+ tokens (192 heads x L amostras),
entao guardamos histogramas com bins fixos, que sao suficientes para densidade,
mais uma subamostra pequena por head para diagramas de dispersao.

Os resultados sao salvos de forma incremental: cada comprimento e gravado assim
que termina, entao um job que estoure o tempo ou a memoria ainda deixa dados
uteis dos comprimentos menores.

Uso:
    python collect_prior_stats.py --log_dir logs/l12/cabam/version_04 \
        --lengths 512 1024 2048 4096 8192 --out prior_stats.pt
"""

import argparse
import os
import time
import traceback

import torch

BETA_BINS = torch.linspace(-8.0, 8.0, 321)
LOGA_BINS = torch.linspace(-16.0, 8.0, 241)
POS_BINS = 64
SUBSAMPLE_PER_HEAD = 300
ALPHA_FLOOR = 1e-8


class Accumulator:
    """Acumula estatisticas por (camada, head) ao longo dos chunks."""

    def __init__(self, n_layers, n_heads, total_len, n_segments=12):
        self.n_layers, self.n_heads, self.total_len = n_layers, n_heads, total_len
        # Histograma de beta por segmento do contexto. POS_BINS (64) e fino
        # demais para estimar quartis com confianca em cada faixa, entao os
        # segmentos sao uma particao mais grossa, feita para caber num grafico
        # de quartis: cada head vira n_segments distribuicoes.
        self.n_segments = n_segments
        self.seg_hist = torch.zeros(n_layers, n_heads, n_segments,
                                    len(BETA_BINS) - 1, dtype=torch.float64)
        z = lambda *s: torch.zeros(*s, dtype=torch.float64)
        self.beta_hist = z(n_layers, n_heads, len(BETA_BINS) - 1)
        self.loga_hist = z(n_layers, n_heads, len(LOGA_BINS) - 1)
        self.count = z(n_layers, n_heads)
        self.beta_sum, self.beta_sq = z(n_layers, n_heads), z(n_layers, n_heads)
        self.loga_sum, self.loga_sq = z(n_layers, n_heads), z(n_layers, n_heads)
        self.neg = z(n_layers, n_heads)
        self.aggressive = z(n_layers, n_heads)
        self.pos_sum = z(n_layers, n_heads, POS_BINS)
        self.pos_neg = z(n_layers, n_heads, POS_BINS)
        self.pos_count = z(n_layers, n_heads, POS_BINS)
        self.sub = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def add(self, layer, alpha, beta, start_pos):
        """alpha/beta: (bs, n_heads, chunk_len), comecando em start_pos."""
        alpha = alpha.reshape(alpha.shape[0], self.n_heads, -1).double()
        beta = beta.reshape(beta.shape[0], self.n_heads, -1).double()
        loga = alpha.clamp_min(ALPHA_FLOOR).log()
        chunk = beta.shape[-1]

        # faixa de posicao de cada token dentro da sequencia inteira
        pos = torch.arange(start_pos, start_pos + chunk, dtype=torch.float64)
        bin_idx = (pos / max(self.total_len, 1) * POS_BINS).long().clamp(0, POS_BINS - 1)
        seg_idx = (pos / max(self.total_len, 1) * self.n_segments).long().clamp(0, self.n_segments - 1)

        for h in range(self.n_heads):
            b, a = beta[:, h].reshape(-1), loga[:, h].reshape(-1)
            self.beta_hist[layer, h] += torch.histogram(b.float(), BETA_BINS).hist.double()
            self.loga_hist[layer, h] += torch.histogram(a.float(), LOGA_BINS).hist.double()
            self.count[layer, h] += b.numel()
            self.beta_sum[layer, h] += b.sum();  self.beta_sq[layer, h] += (b * b).sum()
            self.loga_sum[layer, h] += a.sum();  self.loga_sq[layer, h] += (a * a).sum()
            self.neg[layer, h] += (b < 0).sum()
            self.aggressive[layer, h] += (b < -0.6).sum()

            bb = beta[:, h]                                   # (bs, chunk)
            for tgt in bin_idx.unique():
                sel = bin_idx == tgt
                self.pos_sum[layer, h, tgt] += bb[:, sel].sum()
                self.pos_neg[layer, h, tgt] += (bb[:, sel] < 0).sum()
                self.pos_count[layer, h, tgt] += bb[:, sel].numel()
            for tgt in seg_idx.unique():
                vals = bb[:, seg_idx == tgt].reshape(-1).float()
                self.seg_hist[layer, h, tgt] += torch.histogram(vals, BETA_BINS).hist.double()

            keep = SUBSAMPLE_PER_HEAD - sum(s.shape[0] for s in self.sub[layer][h])
            if keep > 0:
                n = min(keep, b.numel())
                idx = torch.randperm(b.numel())[:n]
                self.sub[layer][h].append(torch.stack([b[idx], a[idx]], dim=1).float())

    def result(self):
        c = self.count.clamp_min(1)
        sub = torch.stack([torch.stack([torch.cat(self.sub[l][h])[:SUBSAMPLE_PER_HEAD]
                                        for h in range(self.n_heads)])
                           for l in range(self.n_layers)])
        return {
            "beta_hist": self.beta_hist.float(), "loga_hist": self.loga_hist.float(),
            "count": self.count.float(),
            "beta_mean": (self.beta_sum / c).float(),
            "beta_std": ((self.beta_sq / c) - (self.beta_sum / c) ** 2).clamp_min(0).sqrt().float(),
            "loga_mean": (self.loga_sum / c).float(),
            "loga_std": ((self.loga_sq / c) - (self.loga_sum / c) ** 2).clamp_min(0).sqrt().float(),
            "frac_neg": (self.neg / c).float(),
            "frac_aggressive": (self.aggressive / c).float(),
            "pos_beta_mean": (self.pos_sum / self.pos_count.clamp_min(1)).float(),
            "pos_frac_neg": (self.pos_neg / self.pos_count.clamp_min(1)).float(),
            "pos_count": self.pos_count.float(),
            "seg_hist": self.seg_hist.float(),
            "n_segments": self.n_segments,
            "subsample": sub,
        }


def quantiles_from_hist(hist, edges, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    """Quantis aproximados a partir do histograma (interpolacao linear no bin)."""
    centers = (edges[:-1] + edges[1:]) / 2
    cdf = hist.cumsum(-1)
    total = cdf[..., -1:].clamp_min(1e-9)
    cdf = cdf / total
    out = []
    for q in qs:
        idx = (cdf < q).sum(-1).clamp(max=hist.shape[-1] - 1)
        out.append(centers[idx])
    return torch.stack(out, dim=-1)


def build_input(kind, length, tokenizer_dir, dataset_dir):
    """Devolve uma lista de token ids de comprimento ~length."""
    if kind == "passkey":
        from eval_utils import PromptGenerator
        gen = PromptGenerator(digits=5)
        prompts, keys = gen(length, sample_size=1, sampling="equidistant")
        return prompts[0] + keys[0]
    # texto natural: tokens de validacao do shard 0 (nao vistos no treino)
    d = torch.load(os.path.join("data", dataset_dir, "sample_000000.pt"))
    ids = d["input_ids"]
    if len(ids) < length:
        raise RuntimeError(f"shard tem {len(ids)} tokens, {length} pedidos")
    return ids[:length].long().tolist()


def run_length(model, priors, tokens, chunk_size, device, n_layers, n_heads, n_segments=12):
    acc = Accumulator(n_layers, n_heads, len(tokens), n_segments)
    state = {"start": 0}
    handles = []
    for idx, mod in enumerate(priors):
        def hook(_m, _i, out, idx=idx):
            alpha, beta, _mu = out
            acc.add(idx, alpha.detach().float().cpu(), beta.detach().float().cpu(), state["start"])
        handles.append(mod.register_forward_hook(hook))

    kv = [None] * n_layers
    try:
        with torch.no_grad():
            pos = 0
            while pos < len(tokens):
                end = min(pos + chunk_size, len(tokens))
                state["start"] = pos
                chunk = torch.tensor(tokens[pos:end], device=device).unsqueeze(0)
                _, kv = model.forward_chunk(chunk, pos, kv, False, None)
                pos = end
    finally:
        for h in handles:
            h.remove()
    return acc.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs/l12/cabam/version_04")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    ap.add_argument("--kinds", nargs="+", default=["text", "passkey"],
                    choices=["text", "passkey"])
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--segments", type=int, default=12,
                    help="em quantas partes o contexto e dividido para o grafico de quartis")
    ap.add_argument("--dataset", default="10B")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="prior_stats.pt")
    ap.add_argument("--time-budget", type=float, default=1e9,
                    help="segundos; para de tentar comprimentos maiores ao estourar")
    args = ap.parse_args()

    import json
    from models.cabam_ssmax import (SSMaxBATransformer as Transformer,
                                    SSMaxBATModelArgs as ModelArgs, AttentionPrior)
    cfg = json.load(open(os.path.join(args.log_dir, "args.json")))
    model = Transformer(ModelArgs(**cfg["model_args"]))
    sd = torch.load(os.path.join(args.log_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval().to(args.device)
    priors = [m for _, m in model.named_modules() if isinstance(m, AttentionPrior)]
    n_layers, n_heads = len(priors), cfg["model_args"]["n_heads"]
    print(f"modelo: {n_layers} camadas x {n_heads} heads, device={args.device}", flush=True)

    results = {"meta": {"log_dir": args.log_dir, "n_layers": n_layers, "n_heads": n_heads,
                        "beta_bins": BETA_BINS, "loga_bins": LOGA_BINS, "pos_bins": POS_BINS,
                        "chunk_size": args.chunk_size}, "runs": {}}
    t_start = time.time()
    for kind in args.kinds:
        for length in sorted(args.lengths):
            elapsed = time.time() - t_start
            if elapsed > args.time_budget:
                print(f"[{kind} {length}] orcamento de tempo estourado, parando", flush=True)
                break
            try:
                tokens = build_input(kind, length, None, args.dataset)
            except Exception as e:
                print(f"[{kind} {length}] entrada indisponivel: {e}", flush=True)
                continue
            t0 = time.time()
            try:
                res = run_length(model, priors, tokens, args.chunk_size,
                                 args.device, n_layers, n_heads, args.segments)
            except (torch.OutOfMemoryError, MemoryError, RuntimeError) as e:
                print(f"[{kind} {length}] FALHOU: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                break
            dt = time.time() - t0
            res["actual_len"] = len(tokens)
            res["seconds"] = dt
            results["runs"][(kind, length)] = res
            torch.save(results, args.out)   # salva incremental
            fn = res["frac_neg"].mean().item() * 100
            fa = res["frac_aggressive"].mean().item() * 100
            bm = res["beta_mean"].mean().item()
            print(f"[{kind:8s} L={len(tokens):7d}] {dt:7.1f}s  "
                  f"beta_medio {bm:+.3f}  beta<0 {fn:5.2f}%  beta<-0.6 {fa:5.2f}%", flush=True)

    print(f"\n-> {args.out}  ({len(results['runs'])} execucoes)")


if __name__ == "__main__":
    main()
