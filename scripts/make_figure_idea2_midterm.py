"""Figure for the Idea 2 midterm section.

(a) Pairwise accuracy A on held-out participants for each reference representation,
    one recording (1 x 20 s) vs three recordings (4 x 5 s), fixed 20 s budget.
(b) Change in A when 2 or 4 contaminated blocks are added to 4 clean blocks.

Usage: python make_figure_idea2_midterm.py RESULTS_JSON OUT_PREFIX
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

res = json.load(open(sys.argv[1]))
C, PR = res["conditions"], res["paired"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.35, 1]))

REPS = [("desc_mean", "Handcrafted"), ("learned_dyn_mean", "Learned, dynamics"),
        ("posture_static", "Static posture"), ("learned_mean", "Learned, full pose"),
        ("si_expr", "SI expression"), ("si_latent", "SI frame latent"), ("si_hyper", "SI hypernet")]
x = np.arange(len(REPS))
for off, cond, lab, col in [(-0.12, "1x20s_1rec", "1 recording (1 x 20 s)", "#8c8c8c"),
                            (0.12, "4x5s_3rec", "3 recordings (4 x 5 s)", "#1f5fa8")]:
    m = np.array([C[f"{r}|{cond}"]["A_ci"][:3] for r, _ in REPS])
    ax1.errorbar(x + off, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt="o", ms=3.5,
                 color=col, capsize=2, lw=1, label=lab)
ax1.axhline(0.5, color="k", lw=0.6, ls=":")
ax1.axvline(1.5, color="k", lw=0.5, alpha=0.3)
ax1.set_xticks(x)
ax1.set_xticklabels([l for _, l in REPS], rotation=30, ha="right")
ax1.set_ylabel("Pairwise accuracy A")
ax1.set_ylim(0.45, 1.02)
ax1.set_title("(a) Reference representation", loc="left", fontsize=8)
ax1.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.1), fontsize=7)

LQ = [("lq_jit_m2", "+2 jit."), ("lq_jit_m4", "+4 jit."), ("lq_wp_m2", "+2 other"), ("lq_wp_m4", "+4 other")]
LINES = [("desc_mean", "Handcrafted, mean", "#8c8c8c", "o", "-"),
         ("desc_median", "Handcrafted, median", "#8c8c8c", "s", "--"),
         ("learned_dyn_mean", "Learned dyn., mean", "#1f5fa8", "o", "-"),
         ("learned_dyn_gated", "Learned dyn., gated", "#c0392b", "s", "-")]
xx = np.arange(len(LQ))
for i, (r, lab, col, mk, ls) in enumerate(LINES):
    m = np.array([PR[f"{r}|{c} - lq_base4"]["A"][:3] for c, _ in LQ])
    ax2.errorbar(xx + (i - 1.5) * 0.08, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt=mk, linestyle="none",
                 ms=3.5, color=col, capsize=2, lw=1, label=lab)
ax2.axhline(0, color="k", lw=0.6, ls=":")
ax2.set_xticks(xx)
ax2.set_xticklabels([l for _, l in LQ])
ax2.set_ylabel("Change in A vs. 4 clean blocks")
ax2.set_title("(b) Contaminated references", loc="left", fontsize=8)
ax2.legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{sys.argv[2]}.{ext}", dpi=200, bbox_inches="tight")
print("saved", sys.argv[2])
