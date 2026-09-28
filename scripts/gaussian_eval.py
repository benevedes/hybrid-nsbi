"""Covariance Gaussian toy: re-score every saved encoder in results/gaussian on the TEST set, on a fine 41-point theta
grid (evaluation only; training uses 21 points). All runs must share (a, b, theta_max). Writes eval41_<tag>.json."""

import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latentcat.common import mlp, results_dir
from latentcat.gaussian_cov_data import D, GaussianCovData

torch.set_num_threads(6)
RES = results_dir("gaussian")

files = sorted(glob.glob(os.path.join(RES, "run_*.json")))
conf = {tuple(json.load(open(f))["args"][k] for k in ("a", "b", "theta_max")) for f in files}
assert len(conf) == 1, f"runs with different (a, b, theta_max): {conf}"
A, B, TMAX = conf.pop()
d = GaussianCovData(a=A, b=B, theta_max=TMAX, n_grid=41, device="cpu")
for f in files:
    tag = os.path.basename(f)[4:-5]
    args = json.load(open(f))["args"]
    n_cat = args.get("latent_dim", 48)
    enc = mlp(D, args.get("hidden", 256), n_cat, args.get("layers", 3))
    enc.load_state_dict(torch.load(os.path.join(RES, f"state_{tag}.pt"), weights_only=False)["encoder"])
    q_lat, active = d.eval_latent(enc, n_cat)
    json.dump(
        dict(
            args=args,
            grid=d.GRID.tolist(),
            q_exact=d.q_exact.tolist(),
            q_hist=d.q_hist.tolist(),
            q_hist_t_fine=d.q_hist_t_fine.tolist(),
            q_hist_x0x1=d.q_hist_x0x1.tolist(),
            report_points=list(d.REPORT_POINTS),
            q_lat=q_lat.tolist(),
            active=active,
        ),
        open(os.path.join(RES, f"eval41_{tag}.json"), "w"),
    )
    print(f"{tag:20s} active={active:2d} | {d.report('latent', q_lat)}", flush=True)
