"""Two single-column figures for the Idea 2 midterm section (complement Table tab:idea2).

OUT_PREFIX_contam : paired change in pairwise accuracy A when 2 or 4 jittered blocks or
                    blocks of another participant are added to 4 clean blocks (dynamics
                    encoder with mean pooling / attention gate, frozen hypernetwork features).
OUT_PREFIX_perpart: per-participant A with one 20 s cross-recording reference, dynamics
                    encoder vs hypernetwork features; two-session participants marked.

Usage: python make_figure_idea2_midterm_v2.py RESULTS_JSON OUT_PREFIX
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

res = json.load(open(sys.argv[1]))
C, PR = res["conditions"], res["paired"]
out = sys.argv[2]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "serif", "mathtext.fontset": "cm"})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print("saved", name)


# ---- Figure 1: contaminated reference sets ----
LQ = [("lq_jit_m2", "+2\njittered"), ("lq_jit_m4", "+4\njittered"),
      ("lq_wp_m2", "+2\nintruder"), ("lq_wp_m4", "+4\nintruder")]
LINES = [("learned_dyn_mean", "Learned dynamics, mean pooling", "#1f5fa8", "o"),
         ("learned_dyn_gated", "Learned dynamics, attention gate", "#c0392b", "s"),
         ("si_hyper", "SI hypernetwork (appearance)", "#2e8b57", "D")]
fig, ax = plt.subplots(figsize=(3.4, 2.7))
xx = np.arange(len(LQ))
for i, (r, lab, col, mk) in enumerate(LINES):
    xs, m = [], []
    for j, (c, _) in enumerate(LQ):
        k = f"{r}|{c} - lq_base4"
        if k in PR:
            xs.append(j)
            m.append(PR[k]["A"][:3])
    xs, m = np.array(xs), np.array(m)
    ax.errorbar(xs + (i - 1) * 0.2, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]],
                fmt=mk, ms=4.5, color=col, capsize=2.5, lw=1, elinewidth=1, label=lab)
ax.axhline(0, color="k", lw=0.7, ls=":")
ax.axvline(1.5, color="k", lw=0.5, alpha=0.3)
ax.set_xticks(xx)
ax.set_xticklabels([l for _, l in LQ])
ax.set_xlim(-0.6, 3.6)
ax.set_ylim(-0.09, 0.015)
ax.set_ylabel("Change in $A$ vs. 4 clean blocks")
ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
fig.tight_layout()
save(fig, out + "1")

# ---- Figure 2: per participant ----
dyn = C["learned_dyn_mean|1x20s_1rec"]
hyp = C["si_hyper|1x20s_1rec"]
parts = dyn["participants"]
a_dyn = np.array(dyn["A"])
a_hyp = np.array([hyp["A"][hyp["participants"].index(p)] for p in parts])
order = np.argsort(a_dyn)
x = np.arange(len(parts))
fig, ax = plt.subplots(figsize=(3.4, 2.7))
two_session = {"P0047", "P0048"}
for xi, idx in enumerate(order):
    if parts[idx] in two_session:
        ax.axvspan(xi - 0.45, xi + 0.45, color="k", alpha=0.08, lw=0)
ax.plot(x, a_dyn[order], "o-", ms=4, lw=1, color="#1f5fa8", label="Learned dynamics, mean pooling")
ax.plot(x, a_hyp[order], "D-", ms=3.5, lw=1, color="#2e8b57", label="SI hypernetwork (appearance)")
ax.axhline(0.5, color="k", lw=0.7, ls=":")
ax.set_xticks(x)
ax.set_xticklabels([parts[i][1:] for i in order], rotation=90, fontsize=7)
ax.set_xlabel("Participant (sorted by learned dynamics $A$)")
ax.set_ylabel("Pairwise accuracy $A$")
ax.set_ylim(0.45, 1.03)
ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
fig.tight_layout()
save(fig, out + "2")
