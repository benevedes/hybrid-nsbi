"""Physics paper figure: test-set scans on the 101-point mu grid of the exact matrix-element likelihood, the m4l
histogram, the latent categories (one-event objective), and NSBI and MSS from one CARL network per process
(scripts/physics_train_carl.py, evaluated by scripts/physics_eval_carl.py).

Run selection is by name, never by score against the exact ceiling: --runs lists the seeds, of which the one with
the MEDIAN ceiling-free score is drawn. Writes figures/physics_fig4.pdf and prints the curves at a few mu values.
"""

import argparse
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latentcat.common import figures_dir, paper_style, results_dir, scan_score
from latentcat.physics_data import PhysicsData

REPORT_MU = PhysicsData.REPORT_POINTS
ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["event_s0", "event_s1", "event_s2"])
a = ap.parse_args()
RES, FIG = results_dir("physics"), figures_dir()
load_run = lambda tag: json.load(open(os.path.join(RES, f"eval101_{tag}.json")))

seeds = sorted((scan_score(np.array(r["grid"]), np.array(r["q_lat"]), REPORT_MU), t) for t, r in ((t, load_run(t)) for t in a.runs))
median_tag = seeds[len(seeds) // 2][1]
run = load_run(median_tag)
print("seeds by ceiling-free score:", seeds, "| median:", median_tag)
mu = np.array(run["grid"])
q_exact, q_hist, q_new = np.array(run["q_exact"]), np.array(run["q_hist"]), np.array(run["q_lat"])

carl = json.load(open(os.path.join(RES, "eval101_carl.json")))
assert np.allclose(carl["grid"], mu), "CARL scans on a different grid"
q_nsbi, q_mss = np.array(carl["q_nsbi"]), np.array(carl["q_mss"])

curves = [
    ("exact", q_exact, dict(color="0.55", lw=1.5, ls="--", label="Exact likelihood (matrix elements)")),
    ("nsbi", q_nsbi, dict(color="#c62828", lw=2.5, ls=":", zorder=5, label="NSBI")),  # on top: it overlaps MSS
    ("mss", q_mss, dict(color="#2e7d32", lw=2.5, label=f"MSS ({carl['args']['mss_bins']} bins)")),
    ("hist", q_hist, dict(color="black", lw=2.5, label=r"$m_{4l}$ histogram")),
    ("new", q_new, dict(color="#1f77b4", lw=2.5, label="Latent Categories")),
]
paper_style()
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for key, q, style in curves:
    ax.plot(mu, q, **style)
ax.axvline(1.0, color="0.35", ls=":", lw=1.3)
ax.axhline(1, color="0.55", ls="--", lw=1.0)
ax.axhline(4, color="0.55", ls=":", lw=1.0)
ax.set_xlim(0, 4)
ax.set_ylim(0, 28)
ax.set_xlabel(r"Signal strength $\mu$")
ax.set_ylabel(r"$-2\Delta\ln\mathcal{L}$")
ax.minorticks_on()
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.57, 1.0))  # tucked against the rising curves, clear of the mu = 1 line
plt.tight_layout()
plt.savefig(os.path.join(FIG, "physics_fig4.pdf"))
plt.close()

idx = [int(np.argmin(np.abs(mu - m))) for m in REPORT_MU]
print(f"{'curve':12s} | " + " | ".join(f"mu={mu[i]:.0f}" for i in idx) + "   (fraction of exact)")
for key, q, _ in curves:
    print(f"{key:12s} | " + " | ".join(f"{q[i]:5.2f} ({q[i]/q_exact[i]:.2f})" for i in idx))
