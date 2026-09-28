"""Toy coverage on the physics case at mu0 = 10 with the nominal expected yield, for the curves of the physics paper
figure plus a coarsely binned MSS.

Pseudo-experiments are drawn from the TEST set weighted to mu0, with a Poisson size of the full expected yield
nu(mu0) (the test set holds only part of the simulation); every method sees the same events. All methods use the
likelihood decomposition of scripts/physics_eval_carl.py (SBI basis, background as reference),
    nu(mu) p(x|mu) / p_bkg(x) = (1 - sqrt mu) nu_bkg + sqrt(mu) nu_sbi g_sbi(x) + (mu - sqrt mu) nu_sig g_sig(x),
    lnL(mu) = sum_i ln[...](x_i) - nu(mu),
and differ only in the per-event ratios g_J:
  * exact: the matrix-element ratios;
  * latent categories: h_J(k) / h_bkg(k) for the argmax category k of the seed drawn in the paper figure (median
    ceiling-free score), histograms from the training pool: the binned Poisson likelihood of the categories;
  * NSBI: r_J = exp(logit_J) of the CARL networks results/physics/carl_{sig,sbi}.pt, as they come;
  * MSS: the histogram ratio of each network's logit, in --mss_bins bins (the paper figure) and in --coarse_bins
    bins, at quantiles of the background-weighted training distribution;
  * m4l histogram: h_J(b) / h_bkg(b) in the 32 log-spaced m4l bins on 180-1000 GeV of the paper figure, plus an
    overflow bin for the ~0.1% of the weight above 1000 GeV.
mu-hat maximizes lnL on a 0.01 grid on mu0 +- 5, refined between grid points (latentcat.coverage.refine_max);
q0 = 2 (max lnL - lnL(mu0)).

Writes results/physics/coverage.npz (q0 and mu-hat of every pseudo-experiment) and figures/physics_coverage.pdf.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latentcat.common import carl_mlp, default_device, figures_dir, hard_categories, mlp, results_dir, scan_score
from latentcat.coverage import plot_coverage, refine_max, summarize
from latentcat.physics_data import FEATS, PhysicsData

ap = argparse.ArgumentParser()
ap.add_argument("--mu0", type=float, default=10.0)
ap.add_argument("--n_pe", type=int, default=2000)
ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--mss_bins", type=int, default=200)
ap.add_argument("--coarse_bins", type=int, default=3)
ap.add_argument("--runs", nargs="+", default=["event_s0", "event_s1", "event_s2"])
ap.add_argument("--chunk", type=int, default=20, help="pseudo-experiments per vectorized block")
ap.add_argument("--device", default=default_device())
ap.add_argument("--plot_only", action="store_true")
a = ap.parse_args()
RES, FIG = results_dir("physics"), figures_dir()
F_TOYS = os.path.join(RES, "coverage.npz")
KEYS = ("exact", "latent", "nsbi", "mss", "mss_coarse", "m4l")

if not a.plot_only:
    dev = torch.device(a.device)
    d = PhysicsData(n_mu=21, device="cpu")
    j0, j1, j4 = (int(np.argmin(np.abs(d.GRID - m))) for m in (0.0, 1.0, 4.0))
    w_all = {"bkg": d.W[:, j0], "sbi": d.W[:, j1], "sig": np.clip(0.5 * (d.W[:, j4] - 2 * d.W[:, j1] + d.W[:, j0]), 0, None)}
    nu = {k: v.sum() for k, v in w_all.items()}
    tr, te = d.tr, d.idx["test"]
    wb = w_all["bkg"][tr] / w_all["bkg"][tr].sum()

    def binned_ratio(k, n_bins):
        """(g_sig, g_sbi) for every event from its bin index k: h_J(k) / h_bkg(k), histograms from the training pool."""
        h_b = np.bincount(k[tr], weights=wb, minlength=n_bins)
        g = []
        for proc in ("sig", "sbi"):
            h_j = np.bincount(k[tr], weights=w_all[proc][tr], minlength=n_bins) / w_all[proc][tr].sum()
            g.append(np.where(h_b > 0, h_j / np.where(h_b > 0, h_b, 1.0), 0.0)[k])
        return tuple(g)

    def quantile_bins(summary, n_bins):
        """Bin index of every event, with edges at quantiles of the background-weighted training distribution."""
        s_tr = summary[tr]
        order = np.argsort(s_tr)
        edges = s_tr[order][np.searchsorted(np.cumsum(wb[order]), np.linspace(0, 1, n_bins + 1)[1:-1])]
        return np.searchsorted(edges, summary)

    with np.errstate(divide="ignore", invalid="ignore"):
        g = {"exact": tuple(np.where(w_all["bkg"] > 0, (w_all[p] / nu[p]) / (w_all["bkg"] / nu["bkg"]), 0.0) for p in ("sig", "sbi"))}

    runs = {t: json.load(open(os.path.join(RES, f"eval101_{t}.json"))) for t in a.runs}
    scores = sorted((scan_score(np.array(r["grid"]), np.array(r["q_lat"]), PhysicsData.REPORT_POINTS), t) for t, r in runs.items())
    tag = scores[len(scores) // 2][1]
    args = json.load(open(os.path.join(RES, f"run_{tag}.json")))["args"]
    enc = mlp(len(FEATS), args["hidden"], args["latent_dim"], args["layers"])
    enc.load_state_dict(torch.load(os.path.join(RES, f"state_{tag}.pt"), map_location="cpu", weights_only=False)["encoder"])
    g["latent"] = binned_ratio(hard_categories(enc, d.Xt), args["latent_dim"])

    need = np.concatenate([tr, te])  # the networks are only evaluated on training (histograms) and test (toys) events
    logit = {}
    for proc in ("sig", "sbi"):
        ck = torch.load(os.path.join(RES, f"carl_{proc}.pt"), map_location="cpu", weights_only=False)
        net = carl_mlp(len(FEATS), ck["args"]["n_nodes"], ck["args"]["n_layers"])
        net.load_state_dict(ck["state"])
        net.to(dev).eval()
        out = np.zeros(len(d.W))
        with torch.no_grad():
            for s in range(0, len(need), 65536):
                idx = need[s : s + 65536]
                out[idx] = net(d.Xt[torch.tensor(idx)].to(dev)).flatten().cpu().double().numpy()
        logit[proc] = out
    g["nsbi"] = (np.exp(logit["sig"]), np.exp(logit["sbi"]))
    for key, nb in (("mss", a.mss_bins), ("mss_coarse", a.coarse_bins)):
        g[key] = tuple(binned_ratio(quantile_bins(logit[p], nb), nb)[i] for i, p in enumerate(("sig", "sbi")))

    i_m = FEATS.index("4l_mass")
    m4l = d.Xt[:, i_m].numpy().astype(np.float64) * d.std[i_m] + d.mean[i_m]
    g["m4l"] = binned_ratio(np.clip(np.digitize(m4l, np.geomspace(180, 1000, 33)) - 1, 0, 32), 33)  # bin 32: > 1000 GeV

    # pseudo-experiments at mu0
    grid = np.round(np.arange(a.mu0 - 5, a.mu0 + 5 + 1e-9, 0.01), 10)
    grid = grid[grid >= 0]
    i0 = int(np.argmin(np.abs(grid - a.mu0)))
    sq = np.sqrt(grid)
    A, B, C = (1 - sq) * nu["bkg"], sq * nu["sbi"], (grid - sq) * nu["sig"]
    nu_mu = A + B + C
    s0 = np.sqrt(a.mu0)
    w0 = np.clip((1 - s0) * w_all["bkg"][te] + s0 * w_all["sbi"][te] + (a.mu0 - s0) * w_all["sig"][te], 0, None)
    rng = np.random.default_rng(a.seed)
    sizes = rng.poisson(float(nu_mu[i0]), a.n_pe)
    ev = te[np.searchsorted(np.cumsum(w0) / w0.sum(), rng.random(sizes.sum()))]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])

    arrays, summary = {}, {}
    for key in KEYS:
        g_sig, g_sbi = g[key]
        lnL = np.empty((a.n_pe, len(grid)))
        for c0 in range(0, a.n_pe, a.chunk):
            c1 = min(c0 + a.chunk, a.n_pe)
            e = ev[starts[c0] : starts[c1 - 1] + sizes[c1 - 1]]
            logd = np.log(np.clip(A + B * g_sbi[e, None] + C * g_sig[e, None], 1e-12, None))
            lnL[c0:c1] = np.add.reduceat(logd, starts[c0:c1] - starts[c0], axis=0) - nu_mu
        lnL_max, mu_hat = refine_max(lnL, grid)
        arrays[f"{key}_q0"], arrays[f"{key}_hat"] = 2 * (lnL_max - lnL[:, i0]), mu_hat
        summary[key] = summarize(arrays[f"{key}_q0"], mu_hat)
        at_edge = float(((mu_hat <= grid[0]) | (mu_hat >= grid[-1])).mean())
        if at_edge:
            print(f"WARNING {key}: {at_edge:.3f} of mu-hat at the edge of the scan", flush=True)
    np.savez_compressed(F_TOYS, **arrays)
    json.dump(dict(args=vars(a), latent_run=tag, mean_size=float(sizes.mean()), coverage=summary),
              open(os.path.join(RES, "coverage.json"), "w"), indent=1)
    print(f"mu0 = {a.mu0:g}, {a.n_pe} pseudo-experiments (mean size {sizes.mean():.0f}); latent categories: {tag}")
    print(f"  {'method':10s} {'mean':>7s} {'std':>6s} {'1 sigma':>8s} {'2 sigma':>8s}   (nominal 0.6827 / 0.9545)")
    for k, s in summary.items():
        print(f"  {k:10s} {s['mean_hat']:7.3f} {s['std_hat']:6.3f} {s['cov_1sig']:8.3f} {s['cov_2sig']:8.3f}")

cfg = json.load(open(os.path.join(RES, "coverage.json")))["args"]
z = np.load(F_TOYS)
green, light_green = "#2e7d32", "#81c784"
curves = [
    ("Exact likelihood", "exact", dict(color="0.45", lw=1.6, ls="--", zorder=6)),
    ("Latent Categories", "latent", dict(color="#1f77b4", lw=2.2, zorder=5)),
    ("NSBI", "nsbi", dict(color="#c62828", lw=1.8)),
    (f"MSS {cfg['mss_bins']} bins", "mss", dict(color=green, lw=1.8)),
    (f"MSS {cfg['coarse_bins']} bins", "mss_coarse", dict(color=light_green, lw=1.8)),
    (r"$m_{4l}$ histogram", "m4l", dict(color="black", lw=1.8)),
]
plot_coverage({k: z[f"{k}_q0"] for k in KEYS}, curves, os.path.join(FIG, "physics_coverage.pdf"), ylim=(-0.13, 0.045))
print("saved", os.path.join(FIG, "physics_coverage.pdf"))
