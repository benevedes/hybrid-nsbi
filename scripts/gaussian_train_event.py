"""
Covariance Gaussian toy: latent categories trained on the one-event objective (latentcat.training.train_one_event).

Checkpoint selection: the evaluation epoch whose validation latent scan has the largest mean -2 dlnL over theta in
{-theta_max, -theta_max/2, theta_max/2, theta_max} (GaussianCovData.score). The exact likelihood is printed for
reference only. The defaults are the settings behind figures/gaussian_fig3.pdf.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latentcat.common import default_device, mlp, results_dir
from latentcat.gaussian_cov_data import D, GaussianCovData
from latentcat.training import train_one_event

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="event_s0")
ap.add_argument("--device", default=default_device())
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--latent_dim", type=int, default=48)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--a", type=float, default=0.8)
ap.add_argument("--b", type=float, default=0.4)
ap.add_argument("--theta_max", type=float, default=1.0)
ap.add_argument("--n_grid", type=int, default=21)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--steps", type=int, default=170)
ap.add_argument("--n_events", type=int, default=12800, help="labelled events per step")
ap.add_argument("--tmpl_batch", type=int, default=30000)
ap.add_argument("--tau0", type=float, default=1.0)
ap.add_argument("--tau1", type=float, default=1.0)  # no anneal: annealing to 0.1 lowered the validation score late in training
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--weight_decay", type=float, default=0.0)  # one event gives a small gradient: decay would dominate it
ap.add_argument("--eval_every", type=int, default=5)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
OUT = results_dir("gaussian")

d = GaussianCovData(a=a.a, b=a.b, theta_max=a.theta_max, n_grid=a.n_grid, seed=a.seed, device=a.device)
print(d.report("exact", d.q_exact))
print(d.report("hist t 32b", d.q_hist))
print(d.report("hist t 200 quantile bins", d.q_hist_t_fine))
print(d.report("hist x0x1 32b", d.q_hist_x0x1), flush=True)


def evaluate(enc, part="val"):  # training-time selection uses val; the final report uses test
    q_lat, active = d.eval_latent(enc, a.latent_dim, part)
    return dict(q_lat=q_lat, active=active)


enc = mlp(D, a.hidden, a.latent_dim, a.layers).to(d.device)
best_state, history = train_one_event(
    d, enc, a.latent_dim, evaluate, a.epochs, a.steps, a.n_events, a.tmpl_batch, a.tau0, a.tau1, a.lr, a.weight_decay, a.eval_every
)

# ---------------- final evaluation of the selected checkpoint ----------------
enc.load_state_dict(best_state)
ev = evaluate(enc, part="test")
torch.save(dict(encoder=best_state, args=vars(a), mean=d.mean, std=d.std), os.path.join(OUT, f"state_{a.tag}.pt"))
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
print("RESULT " + d.report("exact", d.q_exact) + " | " + d.report("hist t", d.q_hist), flush=True)
