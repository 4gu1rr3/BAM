"""Plota o que o collect_prior_stats.py coletou:

  1) densidade de beta por head (heatmap: uma linha por head, 192 no total)
  2) como as estatisticas mudam com o comprimento da sequencia
  3) beta em funcao da posicao relativa dentro da sequencia

Uso:
    python plot_prior_stats.py --stats prior_stats.pt --kind text --prefix prior
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

CLUSTER_C = {"aggressive": "#5a108b", "retrieval": "#ffb10a", "local": "#17b570"}


def load(path, kind):
    d = torch.load(path, weights_only=False)
    runs = {L: r for (k, L), r in d["runs"].items() if k == kind}
    if not runs:
        raise SystemExit(f"nenhuma execucao com kind={kind} em {path}")
    return d["meta"], dict(sorted(runs.items()))


def quantiles_from_hist(hist, edges, qs=(0.5,)):
    """Quantis aproximados a partir do histograma (o bruto nao cabe em disco)."""
    centers = (edges[:-1] + edges[1:]) / 2
    cdf = hist.cumsum(-1)
    cdf = cdf / cdf[..., -1:].clamp_min(1e-9)
    return torch.stack([centers[(cdf < q).sum(-1).clamp(max=hist.shape[-1] - 1)]
                        for q in qs], dim=-1)


def _cluster_of(median_beta):
    return ("aggressive" if median_beta < -0.6
            else "retrieval" if median_beta < 0 else "local")


def density_by_head(meta, runs, length, out, var="beta", xlim=None):
    """Heatmap: 192 heads no eixo y, o parametro no x, cor = densidade dentro da head.

    var='beta' usa theta_beta; var='loga' usa log(alpha), que e a grandeza
    comparavel ao theta_alpha do BAM (la o prior multiplica por exp(theta_alpha),
    aqui multiplica por alpha direto).
    """
    spec = {
        "beta": dict(key="beta_hist", bins="beta_bins", label=r"$\theta_\beta$",
                     xlim=(-3, 3), vlines=[(0, "-"), (-0.6, "--")]),
        "loga": dict(key="loga_hist", bins="loga_bins", label=r"$\log \alpha$",
                     xlim=(-2.5, 1.5), vlines=[(0, "-")]),
    }[var]
    if xlim is None:
        xlim = spec["xlim"]

    res = runs[length]
    edges = meta[spec["bins"]].numpy()
    centers = (edges[:-1] + edges[1:]) / 2
    L, H = meta["n_layers"], meta["n_heads"]

    hist = res[spec["key"]].reshape(L * H, -1).numpy()
    hist = hist / np.maximum(hist.sum(1, keepdims=True), 1e-9)   # densidade por head

    lo, hi = np.searchsorted(centers, xlim[0]), np.searchsorted(centers, xlim[1])
    hist, centers = hist[:, lo:hi], centers[lo:hi]

    # cluster de cada head, sempre pela mediana de beta -- assim a faixa lateral
    # significa a mesma coisa nos dois graficos
    med = quantiles_from_hist(res["beta_hist"], meta["beta_bins"], qs=(0.5,))
    med = med.reshape(L * H).numpy()
    strip = np.array([matplotlib.colors.to_rgb(CLUSTER_C[_cluster_of(m)]) for m in med])

    fig, (sax, ax, cax) = plt.subplots(
        1, 3, figsize=(9.4, 8), gridspec_kw={"width_ratios": [1, 44, 1.2]})

    sax.imshow(strip[:, None, :], aspect="auto", origin="lower", interpolation="nearest")
    sax.set_xticks([])
    sax.set_yticks([(l + 0.5) * H - 0.5 for l in range(L)])
    sax.set_yticklabels([f"L{l}" for l in range(L)], fontsize=9)
    sax.set_ylabel("camada (16 heads cada)", fontsize=11)
    sax.set_title("cluster", fontsize=8)

    vmin = max(hist[hist > 0].min(), 1e-5)
    im = ax.imshow(hist, aspect="auto", origin="lower", cmap="magma_r",
                   norm=LogNorm(vmin=vmin, vmax=hist.max()),
                   extent=[centers[0], centers[-1], -0.5, L * H - 0.5],
                   interpolation="nearest")
    for l in range(1, L):
        ax.axhline(l * H - 0.5, color="#ffffff", lw=0.8, alpha=0.6)
    for v, ls in spec["vlines"]:
        ax.axvline(v, color="#0a0a0a", lw=1.2 if ls == "-" else 1.0, ls=ls)
    ax.set_yticks([])
    ax.set_xlabel(spec["label"], fontsize=12)
    ax.set_title(f"Densidade de {spec['label']} por head — contexto "
                 f"{res['actual_len']:,} tokens".replace(",", "."), fontsize=12)
    fig.colorbar(im, cax=cax, label="densidade (normalizada por head)")

    handles = [plt.Line2D([], [], marker="s", ls="none", color=CLUSTER_C[k], ms=7, label=lab)
               for k, lab in [("aggressive", r"mediana $\theta_\beta<-0.6$"),
                              ("retrieval", r"$-0.6\leq$ mediana $\theta_\beta<0$"),
                              ("local", r"mediana $\theta_\beta\geq 0$")]]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def versus_length(meta, runs, out):
    """Como as estatisticas se movem conforme o contexto cresce."""
    lengths = sorted(runs)
    x = [runs[L]["actual_len"] for L in lengths]
    L_, H = meta["n_layers"], meta["n_heads"]
    cmap = plt.get_cmap("viridis")

    # layout="constrained" em vez de tight_layout: a colorbar de camada e
    # ancorada em todos os eixos, o que impede o tight_layout de rodar e faz
    # os rotulos do eixo x de cima colidirem com os titulos de baixo.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")

    # (a) fracao de tokens com beta<0, por camada
    ax = axes[0, 0]
    for l in range(L_):
        y = [runs[k]["frac_neg"][l].mean().item() * 100 for k in lengths]
        ax.plot(x, y, marker="o", ms=3.5, lw=1.4, color=cmap(l / max(L_ - 1, 1)))
    y = [runs[k]["frac_neg"].mean().item() * 100 for k in lengths]
    ax.plot(x, y, marker="s", ms=5, lw=2.4, color="#c1121f", label="todas as camadas", zorder=5)
    ax.set_ylabel(r"tokens com $\theta_\beta<0$  (%)")
    ax.set_title("(a) quanto o modo retrieval é acionado", fontsize=11)
    ax.legend(fontsize=9)

    # (b) beta medio, por camada
    ax = axes[0, 1]
    for l in range(L_):
        y = [runs[k]["beta_mean"][l].mean().item() for k in lengths]
        ax.plot(x, y, marker="o", ms=3.5, lw=1.4, color=cmap(l / max(L_ - 1, 1)))
    y = [runs[k]["beta_mean"].mean().item() for k in lengths]
    ax.plot(x, y, marker="s", ms=5, lw=2.4, color="#c1121f", zorder=5)
    ax.axhline(0, color="#666", lw=1, ls="--")
    ax.set_ylabel(r"$\theta_\beta$ médio")
    ax.set_title("(b) deslocamento do centro da distribuição", fontsize=11)

    # (c) dispersao dentro da head
    ax = axes[1, 0]
    for l in range(L_):
        y = [runs[k]["beta_std"][l].mean().item() for k in lengths]
        ax.plot(x, y, marker="o", ms=3.5, lw=1.4, color=cmap(l / max(L_ - 1, 1)))
    y = [runs[k]["beta_std"].mean().item() for k in lengths]
    ax.plot(x, y, marker="s", ms=5, lw=2.4, color="#c1121f", zorder=5)
    ax.set_ylabel(r"desvio de $\theta_\beta$ dentro da head")
    ax.set_title("(c) o quanto cada head varia com o conteúdo", fontsize=11)

    # (d) quantas heads sao majoritariamente negativas
    ax = axes[1, 1]
    for thr, lab, c in [(0.5, "mediana < 0 (>50% dos tokens)", CLUSTER_C["retrieval"]),
                        (0.25, ">25% dos tokens", "#7a7a7a"),
                        (0.05, ">5% dos tokens", CLUSTER_C["local"])]:
        y = [int((runs[k]["frac_neg"] > thr).sum()) for k in lengths]
        ax.plot(x, y, marker="o", ms=4, lw=1.8, color=c, label=lab)
    ax.set_ylabel("heads (de 192)")
    ax.set_title(r"(d) heads que acionam $\theta_\beta<0$", fontsize=11)
    ax.legend(fontsize=8.5)

    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("comprimento do contexto (tokens)")
        ax.grid(True, ls="--", lw=0.7, color="#b0b0b0", alpha=0.5)
        ax.set_axisbelow(True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, L_ - 1))
    fig.colorbar(sm, ax=axes, label="camada", fraction=0.02, pad=0.02)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def versus_position(meta, runs, out):
    """beta medio em funcao da posicao relativa, uma curva por comprimento."""
    lengths = sorted(runs)
    cmap = plt.get_cmap("plasma")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rel = (np.arange(meta["pos_bins"]) + 0.5) / meta["pos_bins"]
    for i, k in enumerate(lengths):
        c = cmap(i / max(len(lengths) - 1, 1))
        lab = f"{runs[k]['actual_len']:,}".replace(",", ".")
        w = runs[k]["pos_count"].sum((0, 1))
        m = (runs[k]["pos_beta_mean"] * runs[k]["pos_count"]).sum((0, 1)) / w.clamp_min(1)
        f = (runs[k]["pos_frac_neg"] * runs[k]["pos_count"]).sum((0, 1)) / w.clamp_min(1)
        axes[0].plot(rel, m.numpy(), lw=1.6, color=c, label=lab)
        axes[1].plot(rel, f.numpy() * 100, lw=1.6, color=c, label=lab)
    axes[0].set_ylabel(r"$\theta_\beta$ médio")
    axes[0].axhline(0, color="#666", lw=1, ls="--")
    axes[1].set_ylabel(r"tokens com $\theta_\beta<0$ (%)")
    for ax in axes:
        ax.set_xlabel("posição relativa na sequência")
        ax.grid(True, ls="--", lw=0.7, color="#b0b0b0", alpha=0.5)
        ax.set_axisbelow(True)
    axes[1].legend(fontsize=8, title="contexto", ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="prior_stats.pt")
    ap.add_argument("--kind", default="text", choices=["text", "passkey"])
    ap.add_argument("--prefix", default="prior")
    ap.add_argument("--density-length", type=int, default=None,
                    help="comprimento usado no heatmap (padrao: o maior disponivel)")
    args = ap.parse_args()

    meta, runs = load(args.stats, args.kind)
    print(f"comprimentos disponiveis ({args.kind}): "
          + ", ".join(str(runs[k]['actual_len']) for k in sorted(runs)))

    dl = args.density_length or max(runs)
    for var, tag in [("beta", "beta"), ("loga", "logalpha")]:
        density_by_head(meta, runs, dl, var=var,
                        out=f"{args.prefix}_density_{tag}_{args.kind}.pdf")
    if len(runs) > 1:
        versus_length(meta, runs, f"{args.prefix}_vs_length_{args.kind}.pdf")
        versus_position(meta, runs, f"{args.prefix}_vs_position_{args.kind}.pdf")


if __name__ == "__main__":
    main()
