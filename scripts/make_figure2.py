"""Figure 2 for Analysis 2 (Idea 2): fixed-duration composition, duration sweep,
selection policy and saturation, low-quality references.

Usage: python make_figure2.py RESULTS_JSON OUT_PREFIX
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#2f9e6e", "#b7b6ae", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.linewidth": 0.6, "font.family": "serif"})


def pt(ax, x, ci, color, **kw):
    m, lo, hi = ci[:3]
    ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt="o", ms=4, color=color, ecolor=color,
                elinewidth=1.2, capsize=0, zorder=3, mec="white", mew=0.6, **kw)


def pdiff(C, a, b, n=10000):
    """Participant-paired difference a - b with a bootstrap 95% interval."""
    da, db = dict(zip(C[a]["participants"], C[a]["delta"])), dict(zip(C[b]["participants"], C[b]["delta"]))
    x = np.array([da[p] - db[p] for p in sorted(set(da) & set(db))])
    m = x[np.random.default_rng(0).integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main(path, out):
    r = json.load(open(path))
    C = r["conditions"]
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.1), gridspec_kw=dict(width_ratios=[1.35, 1, 1.1, 1.35]))

    # (a) fixed D = 20 s: paired change vs one continuous 20 s clip from one recording
    ax = axs[0]
    P = r["paired"]
    rows = [("2x10s_1rec", "2×10 s, 1 rec.", ORANGE), ("4x5s_1rec", "4×5 s, 1 rec.", ORANGE),
            ("2x10s_2rec", "2×10 s, 2 rec.", BLUE), ("4x5s_2rec", "4×5 s, 2 rec.", BLUE),
            ("4x5s_3rec", "4×5 s, 3 rec.", GREEN)]
    for i, (n, lab, col) in enumerate(rows):
        m_, lo, hi = pdiff(C, n, "1x20s_1rec")
        ax.plot([lo, hi], [i, i], color=col, lw=1.2)
        ax.plot([m_], [i], "o", color=col, ms=3.5, mec="white", mew=0.5)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(rows)), [l for _, l, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("ΔΔ vs 1×20 s, 1 rec.")
    ax.set_title("(a) fixed 20 s total", loc="left", fontsize=7.5)

    # (b) duration sweep at fixed count
    ax = axs[1]
    for names, col, lab in ((["1x5s_1rec", "1x10s_1rec", "1x20s_1rec"], ORANGE, "1 clip"),
                            (["2x5s_2rec", "2x10s_2rec", "2x20s_2rec"], BLUE, "2 clips, 2 rec.")):
        ys = [C[n]["delta_ci"][0] for n in names]
        ax.plot([0, 1, 2], ys, color=col, lw=0.8)
        for x, n in enumerate(names):
            pt(ax, x, C[n]["delta_ci"], col, label=lab if x == 0 else None)
    ax.set_xticks([0, 1, 2], ["5 s", "10 s", "20 s"])
    ax.set_xlabel("duration per clip")
    ax.legend(frameon=False, fontsize=6, loc="upper left", handletextpad=0.1)
    ax.set_title("(b) duration sweep", loc="left", fontsize=7.5)

    # (c) saturation (random) and coverage selection
    ax = axs[2]
    ks = [1, 2, 4, 8, 16]
    ax.plot(range(5), [C[f"rand_k{k}"]["delta_ci"][0] for k in ks], color=BLUE, lw=0.8)
    for i, k in enumerate(ks):
        pt(ax, i, C[f"rand_k{k}"]["delta_ci"], BLUE, label="random" if i == 0 else None)
    for i, k in ((1, 2), (2, 4), (3, 8)):
        pt(ax, i + 0.18, C[f"cov_k{k}"]["delta_ci"], ORANGE, label="coverage" if k == 2 else None)
    ax.set_xticks(range(5), [str(k) for k in ks])
    ax.set_xlabel("number of 5 s references")
    ax.legend(frameon=False, fontsize=6, loc="upper left", handletextpad=0.1)
    ax.set_title("(c) count and selection", loc="left", fontsize=7.5)

    # (d) low-quality references: paired change vs 4 good blocks
    ax = axs[3]
    typs = [("good", "good"), ("jit_lo", "jitter σ=.02"), ("jit_hi", "jitter σ=.05"),
            ("degraded", "degraded track."), ("wrong_person", "wrong person")]
    for i, (t, lab) in enumerate(typs):
        for m, col, dy in ((2, BLUE, -0.15), (4, ORANGE, 0.15)):
            key = f"lq_{t}_m{m} - lq_base4"
            if key in P:
                m_, lo, hi = P[key]["delta"][:3]
                ax.plot([lo, hi], [i + dy, i + dy], color=col, lw=1.2)
                ax.plot([m_], [i + dy], "o", color=col, ms=3.5, mec="white", mew=0.5,
                        label=f"+{m}" if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(typs)), [l for _, l in typs])
    ax.invert_yaxis()
    ax.set_xlabel("ΔΔ vs 4 good clips")
    ax.legend(frameon=False, fontsize=6, loc="lower right", handletextpad=0.1)
    ax.set_title("(d) added clips", loc="left", fontsize=7.5)

    axs[1].set_ylabel("Δ = d(diff.) − d(same)")
    for a in axs[1:3]:
        a.axhline(0, color=MUTED, lw=0.6)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=200)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
