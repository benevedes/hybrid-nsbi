"""
Physics latent categories trained on the one-event objective (latentcat.training.train_one_event).

Checkpoint selection: the evaluation epoch whose validation latent scan has the largest mean -2 dlnL over mu in {0,2,3,4}
(PhysicsData.score). The exact parton-level likelihood is printed for reference but never used for selection.
Defaults are the settings behind the figure.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latentcat.common import default_device, mlp, results_dir
from latentcat.physics_data import FEATS, PhysicsData
from latentcat.training import train_one_event

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="event_s0")
ap.add_argument("--device", default=default_device())
ap.add_argument("--seed", type=int, default=500000)
ap.add_argument("--latent_dim", type=int, default=48)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--n_mu", type=int, default=21)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--n_events", type=int, default=55680, help="labelled events per step")
ap.add_argument("--tmpl_batch", type=int, default=60000)
ap.add_argument("--tau0", type=float, default=1.0)
ap.add_argument("--tau1", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--weight_decay", type=float, default=0.0)  # one event gives a small gradient: decay would dominate it
ap.add_argument("--eval_every", type=int, default=5)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
OUT = results_dir("physics")

d = PhysicsData(n_mu=a.n_mu, seed=a.seed, device=a.device)  # lumi and mu_max are fixed in PhysicsData, so
# scripts/physics_eval.py rebuilds exactly this dataset with only n_mu changed
print(d.report("exact ceiling", d.q_exact))
print(d.report("m4l hist 32b", d.q_hist), flush=True)


def evaluate(enc, part="val"):  # training-time selection uses val; the final report uses test
    q_lat, active = d.eval_latent(enc, a.latent_dim, part)
    return dict(q_lat=q_lat, active=active)


enc = mlp(len(FEATS), a.hidden, a.latent_dim, a.layers).to(d.device)
best_state, history = train_one_event(
    d, enc, a.latent_dim, evaluate, a.epochs, a.steps, a.n_events, a.tmpl_batch, a.tau0, a.tau1, a.lr, a.weight_decay, a.eval_every
)

# ---------------- final evaluation of the selected checkpoint ----------------
enc.load_state_dict(best_state)
ev = evaluate(enc, part="test")
torch.save(dict(encoder=best_state, args=vars(a), mean=d.mean, std=d.std, feats=FEATS), os.path.join(OUT, f"state_{a.tag}.pt"))
json.dump(
    dict(
        args=vars(a),
        grid=d.GRID.tolist(),
        q_exact=d.q_exact.tolist(),
        q_hist=d.q_hist.tolist(),
        q_lat=ev["q_lat"].tolist(),
        active=ev["active"],
        history=history,
    ),
    open(os.path.join(OUT, f"run_{a.tag}.json"), "w"),
)
print(f"RESULT {a.tag} | active={ev['active']}/{a.latent_dim} | " + d.report("latent", ev["q_lat"]), flush=True)
print("RESULT " + d.report("exact", d.q_exact) + " | " + d.report("m4l hist", d.q_hist), flush=True)
