"""Single-column Idea 2 figure.
(a) At a fixed 20 s total, change in Delta relative to one continuous 20 s clip, by number
    of clips and number of source recordings.
(b) Change in Delta relative to one random 5 s reference as the number of random 5 s
    references grows.

Usage: python make_figure_idea2.py A2_RESULTS_JSON OUT_PREFIX
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figure2 import pdiff

INK, GRID = "#222222", "#e6e6e6"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman"],
                     "mathtext.fontset": "stix", "font.size": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
                     "grid.linewidth": 0.5, "axes.axisbelow": True, "pdf.fonttype": 42})
BAR = {1: "#bdbdbd", 2: "#6f9bcb", 3: "#2c5a8f"}
STYLE = {1: ("#9a9a9a", "o"), 2: ("#4a7fb5", "s"), 3: ("#1f3f6e", "^")}


def series(ax, xs, cis, color, marker, label, dx=0.0):
    m = [c[0] for c in cis]
    ax.errorbar([x + dx for x in xs], m, yerr=[[a - c[1] for a, c in zip(m, cis)],
                                              [c[2] - a for a, c in zip(m, cis)]],
                color=color, marker=marker, ms=2.6, lw=0.8, elinewidth=0.7, capsize=1.6,
                capthick=0.7, label=label)


def main(a2, out):
    C = json.load(open(a2))["conditions"]
    zero = (0.0, 0.0, 0.0)
    fig, axs = plt.subplots(1, 2, figsize=(3.3, 1.65))

    ax = axs[0]
    groups = [("2 × 10 s", ["2x10s_1rec", "2x10s_2rec"]), ("4 × 5 s", ["4x5s_1rec", "4x5s_2rec", "4x5s_3rec"])]
    w = 0.24
    for g, (lab, names) in enumerate(groups):
        for j, n in enumerate(names):
            k = j + 1
            m, lo, hi = pdiff(C, n, "1x20s_1rec")
            x = g + (j - (len(names) - 1) / 2) * w
            ax.bar(x, m, w * 0.9, color=BAR[k], label=f"{k} rec." if g == 1 else None, zorder=2)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=0.6,
                        capsize=1.4, capthick=0.6, zorder=3)
    ax.axhline(0, color=INK, lw=0.5, zorder=2)
    ax.set_xticks([0, 1], [lab for lab, _ in groups])
    ax.set_xlim(-0.5, 1.55)
    ax.set_ylim(-0.03, 0.11)
    ax.set_xlabel("Clips")
    ax.set_ylabel(r"Change in $\Delta$")
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(-0.02, 0.98), handlelength=0.9,
              handletextpad=0.3, columnspacing=0.7, borderaxespad=0)
    ax.set_title("(a) Fixed 20 s total", fontsize=7, loc="left", pad=13)

    ax = axs[1]
    ks = [1, 2, 4, 8, 16]
    rel = lambda n: zero if n == "rand_k1" else pdiff(C, n, "rand_k1")
    series(ax, range(5), [rel(f"rand_k{k}") for k in ks], *STYLE[2], "Random")
    ax.set_xticks(range(5), [str(k) for k in ks])
    ax.set_xlabel("5 s references")
    ax.set_ylim(-0.02, 0.16)
    ax.set_title("(b) Number of references", fontsize=7, loc="left", pad=13)

    fig.tight_layout(pad=0.2, w_pad=0.9)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)


if __name__ == "__main__":
    main(*sys.argv[1:3])
