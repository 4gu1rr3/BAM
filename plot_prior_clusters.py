"""Plota os parametros do prior GGD (theta_beta x theta_alpha) no estilo da
Figura 6 do paper do BAM.

Suporta os dois modelos, que parametrizam o mesmo prior de formas diferentes:

  BAM    prior = -((|j-i| + eps) ** theta_beta) * exp(theta_alpha)
         theta_alpha/theta_beta sao dois escalares treinados por head, entao
         cada head vira exatamente um ponto (12 camadas x 16 heads = 192).

  CABAM  prior = -((|j-i| + eps) ** beta) * alpha
         alpha/beta saem de uma MLP sobre o conteudo do token, entao existe um
         par (alpha, beta) por (head, token) -- nao um escalar por head. Para
         cair no mesmo eixo do BAM usamos beta no x e log(alpha) no y, ja que
         o alpha do CABAM ocupa o lugar do exp(theta_alpha) do BAM.

         Como cada head vira uma nuvem e nao um ponto, o grafico mostra as
         duas coisas: a nuvem por token (fraca, subamostrada) e a mediana por
         head (marcador cheio) -- a mediana e o analogo direto do ponto do
         BAM, e a nuvem mostra a dispersao que a mediana esconde, que e
         justamente o que o CABAM adiciona.

Uso:
    python plot_prior_clusters.py \
        --checkpoint BAM=logs/l12/bam_ssmax/version_02 \
        --checkpoint CABAM=logs/l12/cabam/version_04 \
        --out prior_clusters.pdf
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

# Limite entre o cluster de retrieval e o de retrieval agressivo (Secao 3.3).
AGGRESSIVE = -0.6

# Cores e marcadores iguais aos da Figura 6.
COLORS = {"aggressive": "#5a108b", "retrieval": "#ffb10a", "local": "#17b570"}
CLUSTER_LABELS = [
    ("aggressive", r"$\theta_\beta < -0.6$"),
    ("retrieval",  r"$-0.6 \leq \theta_\beta < 0$"),
    ("local",      r"$\theta_\beta > 0$"),
]
MARKERS = ["s", "o", "^", "D", "v", "P"]

ALPHA_FLOOR = 1e-8   # alpha=softplus(.) pode chegar a zero; log(0) = -inf


def classify(theta_beta):
    if theta_beta < AGGRESSIVE:
        return "aggressive"
    if theta_beta < 0:
        return "retrieval"
    return "local"


def _state_dict(log_dir):
    path = log_dir if log_dir.endswith(".pt") else os.path.join(log_dir, "model.pt")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if not any("theta_beta" in k or "prior" in k for k in sd):
        sd = sd.get("model", sd)
    return sd


def read_bam(log_dir):
    """Um ponto por head, lido direto dos parametros."""
    sd = _state_dict(log_dir)
    layers = {}
    for key, val in sd.items():
        m = re.search(r"layers\.(\d+)\.attention\.prior\.theta_(alpha|beta)", key)
        if m:
            layers.setdefault(int(m.group(1)), {})[m.group(2)] = val.flatten().float()
    beta = torch.cat([layers[l]["beta"] for l in sorted(layers)])
    alpha = torch.cat([layers[l]["alpha"] for l in sorted(layers)])
    return beta, alpha, None   # theta_alpha ja esta em escala log


def load_val_tokens(dataset_dir, n_seqs, seq_len):
    """Primeiros tokens do shard 0 -- a regiao que o treino reserva para validacao
    (DistributedShardedDataset.reset pula val_tokens_padding + val_tokens antes
    de comecar a treinar), entao esses tokens nao foram vistos no treino."""
    shard = os.path.join("data", dataset_dir, "sample_000000.pt")
    d = torch.load(shard)
    n = n_seqs * seq_len
    ids = d["input_ids"][:n].long().view(n_seqs, seq_len)
    codes = d["seq_codes"][:n].long().view(n_seqs, seq_len)
    return ids, codes


def read_cabam(log_dir, n_seqs, batch_size, dataset_dir, device):
    """Uma nuvem por head: roda o modelo e captura as saidas da MLP do prior."""
    from models.cabam_ssmax import (SSMaxBATransformer as Transformer,
                                    SSMaxBATModelArgs as ModelArgs,
                                    AttentionPrior)

    cfg = json.load(open(os.path.join(log_dir, "args.json")))
    model = Transformer(ModelArgs(**cfg["model_args"]))
    model.load_state_dict(_state_dict(log_dir), strict=False)
    model.eval().to(device)

    priors = [m for _, m in model.named_modules() if isinstance(m, AttentionPrior)]
    captured = {}   # layer -> lista de (alpha, beta) por batch
    handles = []
    for idx, mod in enumerate(priors):
        def hook(_m, _inp, out, idx=idx):
            alpha, beta, _mu = out
            captured.setdefault(idx, []).append(
                (alpha.detach().float().cpu(), beta.detach().float().cpu()))
        handles.append(mod.register_forward_hook(hook))

    seq_len = cfg["model_args"]["max_seq_len"]
    ids, codes = load_val_tokens(dataset_dir, n_seqs, seq_len)
    with torch.no_grad():
        for i in range(0, n_seqs, batch_size):
            model(ids[i:i + batch_size].to(device), seq_codes=codes[i:i + batch_size].to(device))
    for h in handles:
        h.remove()

    # (layer, head) -> vetor de amostras sobre (batch, token)
    beta_cloud, alpha_cloud = [], []
    for layer in sorted(captured):
        alpha = torch.cat([a for a, _ in captured[layer]], dim=0)   # (N, heads, seqlen)
        beta = torch.cat([b for _, b in captured[layer]], dim=0)
        n_heads = alpha.shape[1]
        for h in range(n_heads):
            alpha_cloud.append(alpha[:, h].reshape(-1))
            beta_cloud.append(beta[:, h].reshape(-1))

    log_alpha = [a.clamp_min(ALPHA_FLOOR).log() for a in alpha_cloud]
    n_floor = sum(int((a <= ALPHA_FLOOR).sum()) for a in alpha_cloud)
    if n_floor:
        total = sum(a.numel() for a in alpha_cloud)
        print(f"  aviso: {n_floor}/{total} amostras com alpha <= {ALPHA_FLOOR:g} "
              f"foram truncadas antes do log")

    beta_med = torch.tensor([b.median() for b in beta_cloud])
    alpha_med = torch.tensor([a.median() for a in log_alpha])
    return beta_med, alpha_med, (beta_cloud, log_alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True, metavar="NOME=CAMINHO")
    ap.add_argument("--out", default="prior_clusters.pdf")
    ap.add_argument("--figsize", type=float, nargs=2, default=(9.0, 4.0))
    ap.add_argument("--dataset", default="10B", help="subpasta de data/ para o CABAM")
    ap.add_argument("--n-seqs", type=int, default=16, help="sequencias de validacao (CABAM)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--cloud-per-head", type=int, default=60,
                    help="amostras por head desenhadas na nuvem (CABAM)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    ax.grid(True, linestyle="--", linewidth=0.8, color="#b0b0b0", alpha=0.6)
    ax.set_axisbelow(True)

    marker_handles = []
    generator = torch.Generator().manual_seed(0)
    for i, spec in enumerate(args.checkpoint):
        name, _, path = spec.partition("=")
        pe = json.load(open(os.path.join(path, "args.json")))["args"]["position_encoding"]
        print(f"{name} ({pe})")
        if "cabam" in pe:
            beta, alpha, cloud = read_cabam(path, args.n_seqs, args.batch_size,
                                            args.dataset, args.device)
        else:
            beta, alpha, cloud = read_bam(path)

        marker = MARKERS[i % len(MARKERS)]
        if cloud is not None:
            bc, ac = cloud
            for b, a in zip(bc, ac):
                k = min(args.cloud_per_head, b.numel())
                sel = torch.randperm(b.numel(), generator=generator)[:k]
                ax.scatter(b[sel], a[sel], s=3, marker=marker, alpha=0.06,
                           c=COLORS[classify(float(b.median()))],
                           edgecolors="none", zorder=2, rasterized=True)

        colors = [COLORS[classify(float(b))] for b in beta]
        ax.scatter(beta, alpha, c=colors, marker=marker, s=50,
                   edgecolors="none", zorder=3)
        marker_handles.append(Line2D([], [], marker=marker, linestyle="none",
                                     color="#808080", markersize=7, label=name))

        n = len(beta)
        agg = int((beta < AGGRESSIVE).sum())
        neg = int((beta < 0).sum())
        print(f"  n={n:4d}  theta_beta<-0.6: {agg:3d}  theta_beta<0: {neg:3d}  "
              f"theta_beta>=0: {n - neg:3d}")

    ax.set_xlabel(r"$\theta_\beta$", fontsize=13)
    ax.set_ylabel(r"$\theta_\alpha$", fontsize=13)

    color_handles = [Line2D([], [], marker="o", linestyle="none", color=COLORS[k],
                            markersize=7, label=lab) for k, lab in CLUSTER_LABELS]
    first = ax.legend(handles=color_handles, loc="upper left", frameon=True, fontsize=10)
    ax.add_artist(first)
    if len(marker_handles) > 1:
        ax.legend(handles=marker_handles, loc="lower right", frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    root, ext = os.path.splitext(args.out)
    if ext.lower() != ".png":
        fig.savefig(root + ".png", dpi=200, bbox_inches="tight")
        print(f"\n-> {args.out} e {root}.png")
    else:
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
