"""Toy coverage on the Gaussian covariance toy at the nominal theta0 = 0, for the curves of the Gaussian paper figure.

Pseudo-experiments are drawn directly from the true density at theta0 (Sigma = I), with N ~ Poisson(50) events, so no
reweighting enters them; every method sees the same events. The expected yield is 50 at every theta, so only the
shape terms of the likelihood depend on theta:
  * exact: sum_i ln p_theta(x_i), closed form;
  * NSBI: sum_i f(x_i, theta), with f the ensemble-mean logit of results/gaussian/state_nsbi.pt, used as it comes
    (as in the paper figure);
  * latent categories: sum_i ln nu_k(x_i)(theta), with k the argmax category of the seed drawn in the paper figure
    (median ceiling-free score) and templates nu_k(theta) from the training pool reweighted to theta;
  * histogram of t = a x0 x1 + b x2 x3: the same, with the 32 bins of the paper figure (values beyond the edges, at
    the 0.005 and 99.995 training percentiles, go to the outer bins).
theta-hat maximizes lnL on a 201-point grid on [-1, 1], refined between grid points (latentcat.coverage.refine_max);
q0 = 2 (max lnL - lnL(theta0)).

Writes results/gaussian/coverage.npz (q0 and theta-hat of every pseudo-experiment) and figures/gaussian_coverage.pdf.
--plot_only redraws the figure from the saved pseudo-experiments.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latentcat.common import default_device, figures_dir, hard_categories, mlp, results_dir, scan_score
from latentcat.coverage import plot_coverage, refine_max, summarize
from latentcat.gaussian_cov_data import D, THETA_REF, GaussianCovData, log_density, rotated

ap = argparse.ArgumentParser()
ap.add_argument("--n_pe", type=int, default=2000)
ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--runs", nargs="+", default=["event_s0", "event_s1", "event_s2"])
ap.add_argument("--device", default=default_device())
ap.add_argument("--plot_only", action="store_true")
a = ap.parse_args()
RES, FIG = results_dir("gaussian"), figures_dir()
F_TOYS = os.path.join(RES, "coverage.npz")
KEYS = ("exact", "nsbi", "latent", "hist_t")

if not a.plot_only:
    dev = torch.device(a.device)
    nsbi_ck = torch.load(os.path.join(RES, "state_nsbi.pt"), map_location="cpu", weights_only=False)
    A, B, TMAX = (nsbi_ck["args"][k] for k in ("a", "b", "theta_max"))
    d = GaussianCovData(a=A, b=B, theta_max=TMAX, n_grid=21, device="cpu")
    grid = np.linspace(-TMAX, TMAX, 201)
    i0 = int(np.argmin(np.abs(grid - THETA_REF)))

    # pseudo-experiments at theta0 = 0, where Sigma = I
    rng = np.random.default_rng(a.seed)
    sizes = rng.poisson(d.n_exp, a.n_pe)
    X = rng.standard_normal((sizes.sum(), D))
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    Xs = torch.tensor(((X - d.mean) / d.std).astype(np.float32))
    per_pe = lambda v: np.add.reduceat(v, starts, axis=0)  # (events, grid) -> (pseudo-experiments, grid)

    lnL = {}
    Z2, x4sq = rotated(X) ** 2, X[:, 4] ** 2
    lnL["exact"] = per_pe(np.stack([log_density(Z2, x4sq, t, A, B) for t in grid], 1))

    # NSBI: the classifier of scripts/gaussian_train_nsbi.py, conditioned on (theta - theta_ref) / theta_max
    def classifier(hidden, layers):
        mods = [nn.Linear(D + 1, hidden), nn.CELU()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.CELU()]
        return nn.Sequential(*mods, nn.Linear(hidden, 1))

    nets = []
    for st in nsbi_ck["states"]:
        net = classifier(nsbi_ck["args"]["hidden"], nsbi_ck["args"]["layers"])
        net.load_state_dict(st)
        nets.append(net.to(dev).eval())
    x_dev = Xs.to(dev)
    f = np.empty((len(X), len(grid)))
    with torch.no_grad():
        for g, t in enumerate(grid):
            c = torch.full((len(x_dev), 1), (t - nsbi_ck["theta_ref"]) / nsbi_ck["theta_scale"], device=dev)
            f[:, g] = torch.stack([n(torch.cat([x_dev, c], 1)).squeeze(1) for n in nets]).mean(0).cpu().double().numpy()
    lnL["nsbi"] = per_pe(f)
    del f

    # binned methods: templates from the training pool, reweighted to each theta (the balance heuristic of
    # GaussianCovData, over its 11 generation points)
    X_tr = d.Xt[d.tr].numpy().astype(np.float64) * d.std + d.mean
    Z2_tr, x4sq_tr = rotated(X_tr) ** 2, X_tr[:, 4] ** 2
    log_mix = logsumexp(np.stack([log_density(Z2_tr, x4sq_tr, t, A, B) for t in np.linspace(-TMAX, TMAX, 11)]), axis=0)

    def binned_lnL(k_tr, k_pe, n_bins):
        out = np.empty((len(k_pe), len(grid)))
        for g, t in enumerate(grid):
            w = np.exp(log_density(Z2_tr, x4sq_tr, t, A, B) - log_mix)
            nu = d.n_exp * np.bincount(k_tr, weights=w, minlength=n_bins) / w.sum()
            out[:, g] = np.log(np.clip(nu, 1e-12, None))[k_pe]
        return per_pe(out)

    runs = {t: json.load(open(os.path.join(RES, f"eval41_{t}.json"))) for t in a.runs}
    scores = sorted((scan_score(np.array(r["grid"]), np.array(r["q_lat"]), d.REPORT_POINTS), t) for t, r in runs.items())
    tag = scores[len(scores) // 2][1]
    args = json.load(open(os.path.join(RES, f"run_{tag}.json")))["args"]
    enc = mlp(D, args["hidden"], args["latent_dim"], args["layers"])
    enc.load_state_dict(torch.load(os.path.join(RES, f"state_{tag}.pt"), map_location="cpu", weights_only=False)["encoder"])
    lnL["latent"] = binned_lnL(hard_categories(enc, d.Xt[d.tr]), hard_categories(enc, Xs), args["latent_dim"])

    t_tr = A * X_tr[:, 0] * X_tr[:, 1] + B * X_tr[:, 2] * X_tr[:, 3]
    t_pe = A * X[:, 0] * X[:, 1] + B * X[:, 2] * X[:, 3]
    lo, hi = np.percentile(t_tr, [0.005, 99.995])
    to_bin = lambda v: np.clip(np.digitize(v, np.linspace(lo, hi, 33)) - 1, 0, 31)
    lnL["hist_t"] = binned_lnL(to_bin(t_tr), to_bin(t_pe), 32)

    arrays, summary = {}, {}
    for k in KEYS:
        lnL_max, theta_hat = refine_max(lnL[k], grid)
        arrays[f"{k}_q0"], arrays[f"{k}_hat"] = 2 * (lnL_max - lnL[k][:, i0]), theta_hat
        summary[k] = summarize(arrays[f"{k}_q0"], theta_hat)
    np.savez_compressed(F_TOYS, **arrays)
    json.dump(dict(args=vars(a), latent_run=tag, mean_size=float(sizes.mean()), coverage=summary),
              open(os.path.join(RES, "coverage.json"), "w"), indent=1)
    print(f"theta0 = {THETA_REF:g}, {a.n_pe} pseudo-experiments (mean size {sizes.mean():.1f}); latent categories: {tag}")
    print(f"  {'method':8s} {'mean':>8s} {'std':>6s} {'1 sigma':>8s} {'2 sigma':>8s}   (nominal 0.6827 / 0.9545)")
    for k, s in summary.items():
        print(f"  {k:8s} {s['mean_hat']:8.4f} {s['std_hat']:6.3f} {s['cov_1sig']:8.3f} {s['cov_2sig']:8.3f}")

z = np.load(F_TOYS)
curves = [
    ("Exact likelihood", "exact", dict(color="0.45", lw=1.6, ls="--", zorder=6)),
    ("Latent Categories", "latent", dict(color="#1f77b4", lw=2.2, zorder=5)),
    ("NSBI", "nsbi", dict(color="#c62828", lw=1.8)),
    (r"Histogram of $a\,x_0x_1 + b\,x_2x_3$", "hist_t", dict(color="black", lw=1.8)),
]
plot_coverage({k: z[f"{k}_q0"] for k in KEYS}, curves, os.path.join(FIG, "gaussian_coverage.pdf"))
print("saved", os.path.join(FIG, "gaussian_coverage.pdf"))
