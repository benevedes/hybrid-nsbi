"""Gaussian-toy paper figure on a 41-point theta grid, from the test-set scans written by gaussian_eval.py and
gaussian_train_nsbi.py: the exact likelihood, NSBI, the histogram of the statistic that is optimal at the nominal
theta = 0 (the score, t = a x0 x1 + b x2 x3, 32 bins), and the latent categories (one-event objective).

Run selection is by name, never by score against the exact ceiling: --runs lists the seeds, of which the one with
the MEDIAN ceiling-free score is drawn. Writes figures/gaussian_fig3.pdf and prints the curves at a few theta values
(including the finely binned t histogram and the one-block statistic x0 x1, which are not plotted).
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
from latentcat.gaussian_cov_data import THETA_REF

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["event_s0", "event_s1", "event_s2"])
a = ap.parse_args()
RES, FIG = results_dir("gaussian"), figures_dir()
load_run = lambda tag: json.load(open(os.path.join(RES, f"eval41_{tag}.json")))

runs = {t: load_run(t) for t in a.runs}
REPORT = next(iter(runs.values()))["report_points"]
seeds = sorted((scan_score(np.array(r["grid"]), np.array(r["q_lat"]), REPORT), t) for t, r in runs.items())
median_tag = seeds[len(seeds) // 2][1]
med = runs[median_tag]
print("seeds by ceiling-free score:", seeds, "| median:", median_tag)
theta = np.array(med["grid"])
q_exact = np.array(med["q_exact"])

curves = [
    ("exact", q_exact, dict(color="0.55", lw=1.5, ls="--", label="Exact likelihood")),
    ("hist_t", np.array(med["q_hist"]), dict(color="black", lw=2.5, label=r"Histogram of $a\,x_0x_1 + b\,x_2x_3$")),
    ("latent", np.array(med["q_lat"]), dict(color="#1f77b4", lw=2.5, label="Latent Categories")),
]
nsbi_path = os.path.join(RES, "nsbi.json")
if os.path.exists(nsbi_path):
    nsbi = json.load(open(nsbi_path))
    assert np.allclose(nsbi["grid"], theta)
    curves.insert(1, ("nsbi", np.array(nsbi["q_nsbi_unnormalized"]), dict(color="#c62828", lw=2.5, ls=":", zorder=5, label="NSBI")))  # the ensemble-average ratio as it comes, like the physics NSBI

paper_style()
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for key, q, style in curves:
    ax.plot(theta, q, **style)
ax.axvline(THETA_REF, color="0.35", ls=":", lw=1.3)
ax.axhline(1, color="0.55", ls="--", lw=1.0)
ax.axhline(4, color="0.55", ls=":", lw=1.0)
ax.set_xlim(theta.min(), theta.max())
ax.set_ylim(bottom=0)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$-2\Delta\ln\mathcal{L}$")
ax.minorticks_on()
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.05, 1.0))
plt.tight_layout()
plt.savefig(os.path.join(FIG, "gaussian_fig3.pdf"))
plt.close()

idx = [int(np.argmin(np.abs(theta - v))) for v in REPORT]
table = curves + [("hist_t_200q", np.array(med["q_hist_t_fine"]), None), ("hist_x0x1", np.array(med["q_hist_x0x1"]), None)]
table += [(f"latent_{t}", np.array(r["q_lat"]), None) for t, r in runs.items()]
print(f"{'curve':18s} | " + " | ".join(f"theta={theta[i]:+.2f}" for i in idx) + "   (fraction of exact)")
for key, q, _ in table:
    print(f"{key:18s} | " + " | ".join(f"{q[i]:7.2f} ({q[i]/q_exact[i]:.2f})" for i in idx))
