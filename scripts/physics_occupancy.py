"""Physics occupancy figure: what the learned binning looks like.

The median seed's encoder (the one drawn in physics_fig4, selected by the same ceiling-free score) assigns every test
event to a category, and the bars are the resulting expected counts of the Asimov dataset at mu = 1. The side bars are the
templates at mu = 0 and 4, which is what the binned likelihood compares the data against. Categories are sorted by their
occupancy at mu = 1, since the index itself carries no meaning.

Writes figures/physics_occupancy.pdf and prints the occupancy of the largest categories.
"""

import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latentcat.common import figures_dir, mlp, paper_style, results_dir, scan_score
from latentcat.physics_data import FEATS, PhysicsData

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["event_s0", "event_s1", "event_s2"])
ap.add_argument("--alt", type=float, nargs=2, default=[0.0, 4.0], help="the two templates drawn over the data")
a = ap.parse_args()
RES, FIG = results_dir("physics"), figures_dir()

runs = {t: json.load(open(os.path.join(RES, f"eval101_{t}.json"))) for t in a.runs}
seeds = sorted((scan_score(np.array(r["grid"]), np.array(r["q_lat"]), PhysicsData.REPORT_POINTS), t) for t, r in runs.items())
tag = seeds[len(seeds) // 2][1]  # the same run physics_fig4 draws
print("seeds by ceiling-free score:", seeds, "| median:", tag)

args = runs[tag]["args"]
n_cat = args["latent_dim"]
d = PhysicsData(n_mu=args["n_mu"], device="cpu")
enc = mlp(len(FEATS), args["hidden"], n_cat, args["layers"])
enc.load_state_dict(torch.load(os.path.join(RES, f"state_{tag}.pt"), weights_only=False)["encoder"])

k = d.categories(enc, "test")
w = d.Wpart["test"]
lam = np.stack([np.bincount(k, weights=w[:, j], minlength=n_cat) for j in range(d.n_grid)], 1)
lam = lam[np.argsort(lam[:, d.i_ref])[::-1]]  # sort by the Asimov occupancy; the category index is arbitrary
alt = [int(np.argmin(np.abs(d.GRID - m))) for m in a.alt]
n_active = int((lam[:, d.i_ref] > 0.5).sum())

YLIM = (0.03, 1e3)
FIGNAME = "physics_occupancy.pdf"

paper_style()
fig, ax = plt.subplots(figsize=(6.4, 3.6))
n_used = int(np.max(np.nonzero(lam.max(1) > 0))) + 1  # empty categories carry no information; do not plot them
x = np.arange(n_used)
# grouped bars, one group per category: low template (blue) | Asimov data (charcoal) | high template (red)
nominal = d.GRID[d.i_ref]
SYM = r"\mu"
BW = 0.22  # the three bars of a category touch; the wide gap between groups separates the categories
series = [(lam[:n_used, alt[0]], d.GRID[alt[0]], -BW, "#1f4fb8"), (lam[:n_used, d.i_ref], nominal, 0.0, "#333333"),
          (lam[:n_used, alt[1]], d.GRID[alt[1]], BW, "#c0262d")]
for c in range(0, n_used, 2):  # faint band behind every other category
    ax.axvspan(c - 0.5, c + 0.5, color="#f3f3f3", lw=0, zorder=0)
fmt = (lambda v: f"{v:+g}" if v else "0") if nominal == 0 else (lambda v: f"{v:g}")
label = lambda v: rf"Asimov data, ${SYM} = {fmt(v)}$" if v == nominal else rf"template, ${SYM} = {fmt(v)}$"
for y, v, dx, colour in series:
    ax.bar(x + dx, y, width=BW, bottom=YLIM[0], color=colour, linewidth=0, zorder=3, label=label(v))
ax.set_yscale("log")
ax.set_ylim(YLIM)
ax.set_ylabel("expected events")
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=3, handlelength=1.0, columnspacing=1.2)
ax.set_xlim(-0.5, n_used - 0.5)
ax.set_xticks(x[::5])
ax.set_xlabel(f"latent category, sorted by occupancy ({n_used} of {n_cat} used)")
ax.grid(axis="y", color="0.9", lw=0.5, zorder=1)
ax.set_axisbelow(True)
ax.tick_params(which="both", direction="in", top=False, right=False, length=3)
ax.tick_params(axis="x", which="minor", bottom=False)
ax.tick_params(axis="y", which="minor", length=1.5, color="0.6")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("0.35")
plt.tight_layout()
plt.savefig(os.path.join(FIG, FIGNAME))
plt.close()

print(f"{n_active} of {n_cat} categories hold >= 0.5 events; total {lam[:, d.i_ref].sum():.1f}")
print("largest categories: " + "  ".join(f"{v:.1f}" for v in lam[:6, d.i_ref]))
print(f"smallest non-empty template entry: {lam[lam > 0].min():.2e}")
