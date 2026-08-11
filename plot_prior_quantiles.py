"""Distribuicao de theta_beta por head ao longo das partes de um contexto longo.

O contexto e dividido em segmentos iguais e, para cada (head, segmento), os
quartis de theta_beta sao estimados a partir do histograma coletado pelo
collect_prior_stats.py --segments N. Cada painel e um "leque": mediana no meio,
faixa escura = quartis (p25-p75), faixa clara = p10-p90.

Isso responde uma coisa que a media por posicao nao responde: se uma head fica
mais negativa no fim do contexto porque *toda* a distribuicao desliza, ou
porque ela passa a alternar mais entre dois modos (a mediana mal se move mas os
quartis abrem).

Uso:
    python plot_prior_quantiles.py --stats prior_seg16k.pt --kind text
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from collect_prior_stats import quantiles_from_hist

QS = (0.10, 0.25, 0.50, 0.75, 0.90)
C_MED = "#5a108b"
C_IQR = "#5a108b"
C_OUT = "#a06fc4"


def fan(ax, x, q, title, ylim=None):
    """q: (n_segmentos, 5) com os quantis de QS."""
    ax.fill_between(x, q[:, 0], q[:, 4], color=C_OUT, alpha=0.28, lw=0)
    ax.fill_between(x, q[:, 1], q[:, 3], color=C_IQR, alpha=0.38, lw=0)
    ax.plot(x, q[:, 2], color=C_MED, lw=2, marker="o", ms=3)
    ax.axhline(0, color="#0a0a0a", lw=1.1)
    ax.axhline(-0.6, color="#0a0a0a", lw=0.9, ls="--")
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.grid(True, ls="--", lw=0.6, color="#bbb", alpha=0.5)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)


def legend_handles():
    import matplotlib.patches as mp
    return [plt.Line2D([], [], color=C_MED, lw=2, label="mediana"),
            mp.Patch(color=C_IQR, alpha=0.38, label="quartis (p25–p75)"),
            mp.Patch(color=C_OUT, alpha=0.28, label="p10–p90"),
            plt.Line2D([], [], color="#0a0a0a", lw=1.1, label=r"$\theta_\beta=0$"),
            plt.Line2D([], [], color="#0a0a0a", lw=0.9, ls="--", label=r"$\theta_\beta=-0.6$")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="prior_seg16k.pt")
    ap.add_argument("--kind", default="text", choices=["text", "passkey"])
    ap.add_argument("--length", type=int, default=None)
    ap.add_argument("--prefix", default="cabam_quantis")
    ap.add_argument("--select", default="retrieval", choices=["retrieval", "movimento"],
                    help="quais heads detalhar: as de retrieval (mediana global<0) "
                         "ou as que mais mudam de mediana entre segmentos")
    ap.add_argument("--n-heads-detail", type=int, default=12)
    args = ap.parse_args()

    d = torch.load(args.stats, weights_only=False)
    runs = {L: r for (k, L), r in d["runs"].items() if k == args.kind}
    if not runs:
        raise SystemExit(f"nenhuma execucao com kind={args.kind} em {args.stats}")
    length = args.length or max(runs)
    res = runs[length]
    if "seg_hist" not in res:
        raise SystemExit("esse arquivo foi coletado antes do --segments; "
                         "rode o collect_prior_stats.py de novo")

    bins = d["meta"]["beta_bins"]
    seg = res["seg_hist"]                       # (L, H, S, bins)
    L, H, S, _ = seg.shape
    x = (np.arange(S) + 0.5) / S * res["actual_len"]
    nice = f"{res['actual_len']:,}".replace(",", ".")

    # ---- por camada (16 heads somadas) ----
    ql = quantiles_from_hist(seg.sum(1), bins, QS).numpy()      # (L, S, 5)
    qall = quantiles_from_hist(seg.sum((0, 1)), bins, QS).numpy()
    ylim = (min(ql[..., 0].min(), -0.9) - 0.1, max(ql[..., 4].max(), 0.9) + 0.1)

    fig, axes = plt.subplots(3, 4, figsize=(15, 8.5), sharex=True, sharey=True,
                             layout="constrained")
    for l in range(L):
        fan(axes.flat[l], x, ql[l], f"camada {l}", ylim)
    for ax in axes[-1]:
        ax.set_xlabel("posição no contexto (tokens)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\theta_\beta$")
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=5, fontsize=9)
    fig.suptitle(f"Distribuição de $\\theta_\\beta$ por parte do contexto — {args.kind}, "
                 f"{nice} tokens (16 heads somadas por camada)", fontsize=12.5)
    out = f"{args.prefix}_por_camada_{args.kind}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")

    # ---- heads individuais ----
    qh = quantiles_from_hist(seg, bins, QS).numpy()             # (L, H, S, 5)
    med_global = quantiles_from_hist(seg.sum(2), bins, (0.5,)).numpy()[..., 0]
    if args.select == "retrieval":
        cand = [(l, h) for l in range(L) for h in range(H) if med_global[l, h] < 0]
        cand.sort(key=lambda lh: med_global[lh])
        subtitle = "heads de retrieval (mediana global < 0)"
    else:
        span = qh[..., 2].max(-1) - qh[..., 2].min(-1)
        cand = sorted(((l, h) for l in range(L) for h in range(H)),
                      key=lambda lh: -span[lh])
        subtitle = "heads cuja mediana mais se move entre as partes"
    cand = cand[:args.n_heads_detail]
    if not cand:
        print("nenhuma head selecionada"); return

    ncol = 4
    nrow = int(np.ceil(len(cand) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.9 * nrow + 1),
                             sharex=True, squeeze=False, layout="constrained")
    for ax in axes.flat[len(cand):]:
        ax.axis("off")
    for ax, (l, h) in zip(axes.flat, cand):
        fan(ax, x, qh[l, h], f"L{l}H{h}   (mediana global {med_global[l, h]:+.2f})")
    for ax in axes[-1]:
        ax.set_xlabel("posição no contexto (tokens)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\theta_\beta$")
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=5, fontsize=9)
    fig.suptitle(f"Distribuição de $\\theta_\\beta$ por parte do contexto — {subtitle}\n"
                 f"{args.kind}, {nice} tokens", fontsize=12.5)
    out = f"{args.prefix}_por_head_{args.kind}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")

    # ---- resumo numerico ----
    print(f"\ncontexto {nice} tokens, {S} partes de {res['actual_len']//S} tokens")
    print("parte   p10     p25   mediana   p75     p90    (todas as heads)")
    for s in range(S):
        q = qall[s]
        print(f"  {s+1:2d}  {q[0]:+.3f}  {q[1]:+.3f}  {q[2]:+.3f}  {q[3]:+.3f}  {q[4]:+.3f}")


if __name__ == "__main__":
    main()
