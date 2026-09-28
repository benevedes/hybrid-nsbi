"""NSBI and MSS scans for the physics figure, from the two CARL networks of scripts/physics_train_carl.py.

Both use the decomposition of the likelihood ratio into process ratios (SBI basis, background as reference):
    nu(mu) p(x|mu) / p_bkg(x) = (1 - sqrt mu) nu_bkg + sqrt(mu) nu_sbi g_sbi(x) + (mu - sqrt mu) nu_sig g_sig(x),
with the process weights w_bkg = W(mu=0), w_sbi = W(mu=1), w_sig = (W(4) - 2 W(1) + W(0)) / 2 and yields nu_J = sum of w_J.
  * NSBI: g_J(x) = r_J(x) = exp(logit_J(x)), the network's ratio as it comes (no calibration or normalization);
  * MSS: g_J(x) = h_J(b) / h_bkg(b), the histogram ratio of the network's own logit in --mss_bins bins at quantiles of the
    background-weighted training distribution (histograms from the training pool).
Scans are Asimov -2 dlnL on the TEST set with the mu = 1 template as data, on the same 101-point grid as
scripts/physics_eval.py; the exact likelihood written the same way is checked against PhysicsData's exact scan.
Writes results/physics/eval101_carl.json, which scripts/physics_figure.py reads.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latentcat.common import carl_mlp, default_device, results_dir
from latentcat.physics_data import FEATS, PhysicsData

ap = argparse.ArgumentParser()
ap.add_argument("--mss_bins", type=int, default=200)
ap.add_argument("--device", default=default_device())
a = ap.parse_args()
RES = results_dir("physics")
dev = torch.device(a.device)

d = PhysicsData(n_mu=101, device="cpu")
j0, j1, j4 = (int(np.argmin(np.abs(d.GRID - m))) for m in (0.0, 1.0, 4.0))
w_all = {"bkg": d.W[:, j0], "sbi": d.W[:, j1], "sig": np.clip(0.5 * (d.W[:, j4] - 2 * d.W[:, j1] + d.W[:, j0]), 0, None)}
nu = {k: v.sum() for k, v in w_all.items()}
tr, te = d.tr, d.idx["test"]
need = np.concatenate([tr, te])  # the network is only ever evaluated on training (histograms) and test (scans) events

logit, epochs = {}, {}
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
    logit[proc], epochs[proc] = out, ck["epoch"]
    wb = w_all["bkg"][tr] / w_all["bkg"][tr].sum()
    print(f"{proc}: epoch {ck['epoch']}, E_bkg[r] on the training pool {np.dot(wb, np.exp(out[tr])):.4f} (1 if normalized)", flush=True)


def hist_ratio(summary, proc, n_bins):
    s_tr, wb = summary[tr], w_all["bkg"][tr]
    order = np.argsort(s_tr)
    edges = s_tr[order][np.searchsorted(np.cumsum(wb[order]) / wb.sum(), np.linspace(0, 1, n_bins + 1)[1:-1])]
    b_tr = np.searchsorted(edges, s_tr)
    h_b = np.bincount(b_tr, weights=wb, minlength=n_bins) / wb.sum()
    h_j = np.bincount(b_tr, weights=w_all[proc][tr], minlength=n_bins) / w_all[proc][tr].sum()
    g_bin = np.where(h_b > 0, h_j / np.where(h_b > 0, h_b, 1.0), 0.0)
    return g_bin[np.searchsorted(edges, summary)]


sq = np.sqrt(d.GRID)
A, B, C = (1 - sq) * nu["bkg"], sq * nu["sbi"], (d.GRID - sq) * nu["sig"]
w_ref = d.Wpart["test"][:, d.i_ref]


def scan(g_sig, g_sbi):
    dens = A[None, :] + B[None, :] * g_sbi[te, None] + C[None, :] * g_sig[te, None]
    lnL = (w_ref[:, None] * np.log(np.clip(dens, 1e-12, None))).sum(0) - (A + B + C)
    return 2 * (lnL.max() - lnL)


with np.errstate(divide="ignore", invalid="ignore"):
    r_exact = {k: np.where(w_all["bkg"] > 0, (w_all[k] / nu[k]) / (w_all["bkg"] / nu["bkg"]), 0.0) for k in ("sig", "sbi")}
q_exact = scan(r_exact["sig"], r_exact["sbi"])
print(f"exact via the decomposition vs PhysicsData's exact scan: max abs dev {np.abs(q_exact - d.q_exact).max():.3f}", flush=True)
q_nsbi = scan(np.exp(logit["sig"]), np.exp(logit["sbi"]))
q_mss = scan(hist_ratio(logit["sig"], "sig", a.mss_bins), hist_ratio(logit["sbi"], "sbi", a.mss_bins))
json.dump(dict(args=vars(a), epochs=epochs, grid=d.GRID.tolist(), q_exact=d.q_exact.tolist(), q_nsbi=q_nsbi.tolist(),
               q_mss=q_mss.tolist()), open(os.path.join(RES, "eval101_carl.json"), "w"))
print(d.report("carl nsbi", q_nsbi))
print(d.report(f"carl mss {a.mss_bins}b", q_mss))
