"""Figure 1 for Analysis 1: coverage, cross- vs same-recording distance differences,
per-descriptor differences, and reference-selection comparison.

Usage: python make_figure.py RESULTS_JSON OUT_PREFIX
"""
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#b7b6ae", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.linewidth": 0.6, "font.family": "serif"})


def ci_bar(ax, x, ci, color):
    m, lo, hi = ci[:3]
    ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt="o", ms=5, color=color,
                ecolor=color, elinewidth=1.4, capsize=0, zorder=3, mec="white", mew=0.8)


def main(path, out):
    r = json.load(open(path))
    cov = r["coverage"]
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.0), gridspec_kw=dict(width_ratios=[1.5, 1, 1.3, 1]))

    # (a) coverage: valid windows per participant, split by recording
    ax = axs[0]
    wpp = cov["windows_per_participant"]
    parts = sorted(wpp, key=lambda p: (-len(wpp[p]), -sum(wpp[p].values())))
    for i, p in enumerate(parts):
        bottom = 0
        for j, (rid, n) in enumerate(sorted(wpp[p].items())):
            ax.bar(i, n, bottom=bottom, width=0.8, color=BLUE if len(wpp[p]) > 1 else GRAY,
                   alpha=1.0 if j % 2 == 0 else 0.55, edgecolor="white", linewidth=0.6)
            bottom += n
    nm = cov["n_multi"]
    ax.axvline(nm - 0.5, color=MUTED, lw=0.6, ls=":")
    ax.text(nm / 2 - 0.5, ax.get_ylim()[1] * 0.97, f"{nm} with ≥2 rec.", ha="center", va="top", color=INK)
    ax.set_xticks([])
    ax.set_xlabel(f"participants (n={len(parts)})")
    ax.set_ylabel("valid 10 s windows")
    ax.set_title("(a) coverage", loc="left", fontsize=7.5)

    # (b) participant-level delta: cross-recording vs same-recording
    ax = axs[1]
    rng = np.random.default_rng(1)
    for x, key, col in ((0, "cross", BLUE), (1, "same", ORANGE)):
        d = np.array(r[key]["delta"])
        multi_sess = np.array([len(cov["sessions_per_multi"].get(p, [])) > 1 for p in r[key]["participants"]])
        jit = x + rng.uniform(-0.18, 0.18, len(d))
        ax.scatter(jit[~multi_sess], d[~multi_sess], s=9, color=col, alpha=0.45, lw=0)
        ax.scatter(jit[multi_sess], d[multi_sess], s=14, facecolor="white", edgecolor=col, lw=1.0,
                   label="has a 2nd session" if x == 0 else None)
        ci_bar(ax, x + 0.32, r[key]["delta_ci"], col)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks([0.15, 1.15], ["other\nrec.", "same\nrec."])
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel("Δ = d(diff.) − d(same)")
    ax.legend(frameon=False, fontsize=6, loc="upper left", handletextpad=0.2)
    ax.set_title("(b) same-person adv.", loc="left", fontsize=7.5)

    # (c) per-descriptor delta (cross-recording)
    ax = axs[2]
    pf = r["per_feature_cross"]
    names = list(pf)
    lab = [n.replace("wrist_", "wr. ").replace("_", " ").replace("excursion", "exc.")
           .replace("head ang speed", "head ang. sp.").replace("hand height", "hand ht.") for n in names]
    for i, n in enumerate(names):
        m, lo, hi = pf[n][:3]
        ax.plot([lo, hi], [i, i], color=BLUE, lw=1.4)
        ax.plot([m], [i], "o", color=BLUE, ms=4, mec="white", mew=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(names)), lab)
    ax.invert_yaxis()
    ax.set_xlabel("Δ (single descriptor)")
    ax.set_title("(c) per descriptor", loc="left", fontsize=7.5)

    # (d) reference selection
    ax = axs[3]
    rs = r["refsel"]
    c, s = np.array(rs["cont"]), np.array(rs["spread"])
    for a, b in zip(c, s):
        ax.plot([0, 1], [a, b], color=GRAY, lw=0.6, zorder=1)
    ax.scatter(np.zeros_like(c), c, s=9, color=ORANGE, zorder=2, lw=0)
    ax.scatter(np.ones_like(s), s, s=9, color=BLUE, zorder=2, lw=0)
    ci_bar(ax, -0.25, rs["cont_ci"], ORANGE)
    ci_bar(ax, 1.25, rs["spread_ci"], BLUE)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks([0, 1], ["1×10 s\ncont.", "2×5 s\nspread"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Δ")
    ax.set_title("(d) ref. selection", loc="left", fontsize=7.5)

    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=200)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
