"""
NSBI on the covariance Gaussian toy: an ensemble of classifiers conditioned on theta, whose logit estimates
log p_theta(x) / p_0(x).

Training data (standard NSBI, no per-event weights in the loss): for every alternative grid value theta_j, n_train
events drawn at theta_j (label 1) and n_train events drawn at the reference theta = 0 (label 0), all with theta_j as
the conditioning input (divided by theta_max). Events are drawn by resampling the shared training pool of
GaussianCovData with its weights, exactly like the latent categories' training events; the validation set is built the same way from
the VALIDATION pool and shared by all members.
Each member has its own initialization, sampling and shuffling seeds, trains with BCE and keeps the epoch with the
lowest validation BCE.

Evaluation on the TEST set, on a 41-point theta grid. The ensemble logit f(x, theta) is the mean over members, and
the ensemble-average ratio exp(f) is used as it comes, as the physics NSBI uses its ratios:
    -2 dlnL(theta) = -2 sum_i w_ref(x_i) f(x_i, theta),   shifted to a minimum of zero,
with w_ref the test-set weights at theta = 0 (summing to the 50 expected events; the rate does not depend on theta).

Writes results/gaussian/nsbi.json and state_nsbi.pt.
Defaults use the same 21-point training grid as scripts/gaussian_train_event.py.
--eval_only recomputes the evaluation from a saved state_nsbi.pt without retraining.

"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latentcat.common import default_device, results_dir
from latentcat.gaussian_cov_data import D, THETA_REF, GaussianCovData, exact_q

ap = argparse.ArgumentParser()
ap.add_argument("--device", default=default_device())
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--a", type=float, default=0.8)
ap.add_argument("--b", type=float, default=0.4)
ap.add_argument("--theta_max", type=float, default=1.0)
ap.add_argument("--n_grid", type=int, default=21)
ap.add_argument("--members", type=int, default=10)
ap.add_argument("--n_train", type=int, default=250000, help="events per class per alternative theta")
ap.add_argument("--n_val", type=int, default=75000, help="validation events per class per alternative theta")
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--epochs", type=int, default=80)
ap.add_argument("--batch", type=int, default=8000)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--weight_decay", type=float, default=0.0)
ap.add_argument("--n_eval_grid", type=int, default=41)
ap.add_argument("--n_norm", type=int, default=0, help="training-pool events for the per-theta normalization (0: the whole pool)")
ap.add_argument("--eval_only", action="store_true", help="skip training; evaluate the saved state_nsbi.pt")
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
OUT = results_dir("gaussian")
dev = torch.device(a.device)

saved = None
if a.eval_only:  # take the toy and the architecture from the saved ensemble, not from this run's defaults
    saved = torch.load(os.path.join(OUT, "state_nsbi.pt"), weights_only=False)
    for k in ("a", "b", "theta_max", "n_grid", "hidden", "layers"):
        setattr(a, k, saved["args"][k])

d = GaussianCovData(a=a.a, b=a.b, theta_max=a.theta_max, n_grid=a.n_grid, seed=a.seed, device=a.device)
THETA_SCALE = a.theta_max  # conditioning input is (theta - THETA_REF) / theta_max
alt = [j for j in range(d.n_grid) if j != d.i_ref]
# validation-pool resampling table, the counterpart of d.CDF for the training pool
W_VAL = d.Wpart["val"]
CDF_VAL = np.asfortranarray(np.cumsum(W_VAL / W_VAL.sum(0, keepdims=True), axis=0))


def draw(pool, cdf, j, n, rng):
    """n unweighted events at grid index j, resampled from `pool` (event indices) with its weights."""
    return pool[np.searchsorted(cdf[:, j], rng.random(n))]


def build_set(pool, cdf, n, rng):
    """Features (x, conditioning theta) and labels: for each alternative theta_j, n events at theta_j (1) and n at the
    reference theta (0)."""
    idx, cond, lab = [], [], []
    for j in alt:
        idx += [draw(pool, cdf, j, n, rng), draw(pool, cdf, d.i_ref, n, rng)]
        cond += [np.full(2 * n, d.GRID[j], np.float32)]
        lab += [np.ones(n, np.float32), np.zeros(n, np.float32)]
    idx = torch.tensor(np.concatenate(idx), device=dev)
    c = torch.tensor((np.concatenate(cond) - THETA_REF) / THETA_SCALE, device=dev).unsqueeze(1)
    return torch.cat([d.Xt[idx], c], 1), torch.tensor(np.concatenate(lab), device=dev).unsqueeze(1)


def classifier():
    layers = [nn.Linear(D + 1, a.hidden), nn.CELU()]
    for _ in range(a.layers - 1):
        layers += [nn.Linear(a.hidden, a.hidden), nn.CELU()]
    return nn.Sequential(*layers, nn.Linear(a.hidden, 1))


crit = nn.BCEWithLogitsLoss()


@torch.no_grad()
def val_loss(net):
    net.eval()
    B = 4 * a.batch
    tot = sum(crit(net(Xv[s : s + B]), yv[s : s + B]).item() * len(yv[s : s + B]) for s in range(0, len(yv), B))
    net.train()
    return tot / len(yv)


states, member_log = [], []
if a.eval_only:
    states, member_log = saved["states"], json.load(open(os.path.join(OUT, "nsbi.json")))["members"]
    a.members = len(states)
else:
    Xv, yv = build_set(d.idx["val"], CDF_VAL, a.n_val, np.random.default_rng([a.seed, 10**6]))
t0 = time.time()
for m in range(0 if a.eval_only else a.members):
    torch.manual_seed(a.seed * 1000 + m)
    rng = np.random.default_rng([a.seed, m])  # this member's training sample
    gen = torch.Generator().manual_seed(a.seed * 1000 + m)  # this member's shuffle order
    X, y = build_set(d.tr, d.CDF, a.n_train, rng)
    net = classifier().to(dev)
    opt = optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    best, best_state, best_epoch, losses = np.inf, None, -1, []
    for ep in range(a.epochs):
        order = torch.randperm(len(y), generator=gen).to(dev)
        for s in range(0, len(y), a.batch):
            b = order[s : s + a.batch]
            loss = crit(net(X[b]), y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        vl = val_loss(net)
        losses.append(vl)
        if vl < best:
            best, best_epoch, best_state = vl, ep, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    states.append(best_state)
    member_log.append(dict(member=m, best_val_loss=best, best_epoch=best_epoch, val_losses=losses))
    print(f"member {m:2d}: best val BCE {best:.5f} at epoch {best_epoch} [{(time.time()-t0)/60:.1f} min]", flush=True)
    del X, y

# ---------------- evaluation: ensemble-mean logit on the test set, Asimov scan on a fine theta grid ----------------
grid = np.linspace(-a.theta_max, a.theta_max, a.n_eval_grid)
w_ref = d.Wpart["test"][:, d.i_ref]  # numpy float64; the weighted sum is done on the CPU (MPS has no float64)
x_test = d.Xt[d.idx["test"]]
nets = []
for st in states:
    net = classifier().to(dev)
    net.load_state_dict(st)
    net.eval()
    nets.append(net)


@torch.no_grad()
def ensemble_logit(x, theta):
    """Ensemble-mean logit f(x, theta) for a feature tensor x, as float64 numpy (MPS has no float64)."""
    out = []
    for s in range(0, len(x), 200000):
        xs = x[s : s + 200000]
        xc = torch.cat([xs, torch.full((len(xs), 1), (theta - THETA_REF) / THETA_SCALE, device=dev)], 1)
        out.append(torch.stack([net(xc).squeeze(1) for net in nets]).mean(0).cpu().numpy().astype(np.float64))
    return np.concatenate(out)


# normalization sample: a uniform subsample of the training pool, weighted by its reference-point weights
norm_idx = np.random.default_rng([a.seed, 2 * 10**6]).choice(d.tr, size=a.n_norm if 0 < a.n_norm < len(d.tr) else len(d.tr), replace=False)
x_norm, w_norm = d.Xt[norm_idx], d.W[norm_idx, d.i_ref] / d.W[norm_idx, d.i_ref].sum()
llr, log_z = np.zeros(len(grid)), np.zeros(len(grid))  # sum_i w_ref(x_i) f(x_i, theta) on the test set, and log Z(theta)
for g, theta in enumerate(grid):
    f_norm = ensemble_logit(x_norm, theta)
    log_z[g] = np.log(np.dot(w_norm, np.exp(f_norm)))
    llr[g] = float(np.dot(w_ref, ensemble_logit(x_test, theta)))
q_raw = -2 * llr
q_raw -= q_raw.min()
q_nsbi = -2 * (llr - w_ref.sum() * log_z)
q_nsbi -= q_nsbi.min()
q_closed = exact_q(grid, a.a, a.b, d.n_exp)

if not a.eval_only:
    torch.save(dict(states=states, args=vars(a), mean=d.mean, std=d.std, theta_ref=THETA_REF, theta_scale=THETA_SCALE), os.path.join(OUT, "state_nsbi.pt"))
json.dump(
    dict(
        args=vars(a),
        grid=grid.tolist(),
        q_nsbi=q_nsbi.tolist(),
        q_nsbi_unnormalized=q_raw.tolist(),
        log_z=log_z.tolist(),
        q_exact_closed_form=q_closed.tolist(),
        members=member_log,
    ),
    open(os.path.join(OUT, "nsbi.json"), "w"),
)
idx = [int(np.argmin(np.abs(grid - c))) for c in d.REPORT_POINTS]
print("log Z(theta) at " + " ".join(f"theta={grid[i]:+.2f}: {log_z[i]:+.4f}" for i in idx), flush=True)
print("RESULT nsbi: " + "  ".join(f"theta={grid[i]:+.2f}: {q_nsbi[i]:6.2f} ({q_nsbi[i]/q_closed[i]:.2f} of closed form)" for i in idx), flush=True)

