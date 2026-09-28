"""Re-score every saved physics encoder in results/physics on the TEST set, on a fine 101-point mu grid (evaluation only; training
uses 21 points). Writes results/physics/eval101_<tag>.json, which the figure script reads."""

import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latentcat.common import mlp, results_dir
from latentcat.physics_data import FEATS, PhysicsData

torch.set_num_threads(6)
RES = results_dir("physics")

d = PhysicsData(n_mu=101, device="cpu")
for f in sorted(glob.glob(os.path.join(RES, "run_*.json"))):
    tag = os.path.basename(f)[4:-5]
    args = json.load(open(f))["args"]
    n_cat = args.get("latent_dim", 48)
    enc = mlp(len(FEATS), args.get("hidden", 256), n_cat, args.get("layers", 3))
    enc.load_state_dict(torch.load(os.path.join(RES, f"state_{tag}.pt"), weights_only=False)["encoder"])
    q_lat, active = d.eval_latent(enc, n_cat)
    json.dump(
        dict(
            args=args,
            grid=d.GRID.tolist(),
            q_exact=d.q_exact.tolist(),
            q_hist=d.q_hist.tolist(),
            q_lat=q_lat.tolist(),
            active=active,
        ),
        open(os.path.join(RES, f"eval101_{tag}.json"), "w"),
    )
    print(f"{tag:20s} active={active:2d} | {d.report('latent', q_lat)}", flush=True)
