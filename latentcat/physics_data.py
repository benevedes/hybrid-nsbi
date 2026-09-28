"""gg -> ZZ -> 4l data prep, used by every physics script.

Inputs: data/physics_cache_{bkg,sbi,sig}.npz, produced by scripts/physics_cache_csv.py from the generator CSVs
(see data/README.md). Each event carries its four SM squared matrix elements and its generator weight.

What this builds, in order:
  1. Load the three cached samples (bkg, sbi, sig) and their per-event matrix elements and MC weights.
  2. Exact matrix-element reweighting: an expected weight W[i, j] for every event i at every mu on the grid,
     combining the three samples with the balance heuristic so the union is one unbiased sample at any mu.
  3. A 30% holdout per sample, from a fixed split seed, halved into "val" (checkpoint selection) and "test" (all
     reported numbers). Training uses `tr` only.
  4. Feature standardization fitted on `tr`.
  5. Reference scans on the holdout (via WeightedEventData): the exact unbinned likelihood (a parton-level
     ceiling, reported but never used for selection), plus the m4l histogram baseline.

Conventions: grid points GRID[0..n_grid-1] are values of mu; `i_ref` is the index of mu = 1 (the Asimov point);
all -2 dlnL scans are Asimov expectations with the mu = 1 template as "data", shifted so their minimum is zero.
"""

import os

import numpy as np
import torch

from .common import WeightedEventData, data_dir

# The 17 input features of the encoder (four lepton four-vectors as pt/eta/phi/E, plus m4l).
FEATS = [
    "l1_pt",
    "l1_eta",
    "l1_phi",
    "l1_energy",
    "l2_pt",
    "l2_eta",
    "l2_phi",
    "l2_energy",
    "l3_pt",
    "l3_eta",
    "l3_phi",
    "l3_energy",
    "l4_pt",
    "l4_eta",
    "l4_phi",
    "l4_energy",
    "4l_mass",
]


class PhysicsData(WeightedEventData):
    REPORT_POINTS = (0.0, 2.0, 3.0, 4.0)  # mu values quoted in reports and used by the selection score
    SYMBOL = "mu"

    def __init__(self, lumi=100.0, mu_max=4.0, n_mu=21, holdout=0.3, split_seed=500000, seed=500000, device="cpu"):
        self.device = torch.device(device)

        # ---------- 1. load the three samples, concatenated into one event list ----------
        procs = ("bkg", "sbi", "sig")  # sample order; `src` below records which sample each event came from
        D = {p: np.load(os.path.join(data_dir(), f"physics_cache_{p}.npz")) for p in procs}
        cols = list(D["bkg"]["columns"])
        ci = {c: i for i, c in enumerate(cols)}
        # X: (N_events, 17) raw features, all samples concatenated in `procs` order
        X = np.concatenate([D[p]["X"][:, [ci[f] for f in FEATS]] for p in procs]).astype(np.float32)
        # msq[k]: (N_events,) squared matrix element of component k at each event's own kinematics.
        # Every event carries all four; msq["sbi"] == msq["bkg"] + msq["int"] + msq["sig"] to 1e-7.
        # The cache stores float32 (physics_cache_csv.py), so float64 here recovers no precision that the CSV had;
        # it only keeps the sums, products and logs built from these columns from rounding further.
        msq = {
            k: np.concatenate([D[p]["X"][:, ci[f"msq_{k}_sm"]] for p in procs]).astype(np.float64)
            for k in ("bkg", "int", "sig", "sbi")
        }
        # wt: (N_events,) generator weight of each event in its own sample, scaled to `lumi` (so sums are yields)
        wt = np.concatenate([D[p]["weights"] for p in procs]).astype(np.float64) * lumi
        # src: (N_events,) 0/1/2 = which sample (bkg/sbi/sig) the event was generated in
        src = np.concatenate([np.full(len(D[p]["X"]), i) for i, p in enumerate(procs)])
        n_s = np.array([len(D[p]["X"]) for p in procs], float)  # events per sample
        sig_s = np.array([D[p]["weights"].sum() * lumi for p in procs])  # total yield per sample

        # ---------- 2. exact reweighting: W[i, j] = expected weight of event i at mu = GRID[j] ----------
        # own: the matrix element of the process each event was actually generated from. The generator sampled
        # events with density ∝ own * (phase space / PDF factor), so dividing wt by own isolates that factor.
        own = np.where(src == 0, msq["bkg"], np.where(src == 1, msq["sbi"], msq["sig"]))
        # bad: 152 background rows have own == 0 (and wt == 0); they would give 0/0. Masked out, weight 0.
        bad = ~(own > 0) | ~np.isfinite(own) | ~np.isfinite(wt)
        own_safe = np.where(bad, 1.0, own)
        # g: per-event phase-space / PDF factor, independent of the hypothesis. Same g for any mu.
        g = np.where(bad, 0.0, wt / own_safe)
        # lam: balance-heuristic weight for combining the three overlapping samples into one estimator.
        # For an event from sample s, lam = (n_s/sig_s) own / sum_t (n_t/sig_t) msq_t. Summing lam over the
        # samples that could have produced a given phase-space point gives 1, so the union is unbiased,
        # and each event is weighted by how much its own sample dominates at that point (low variance).
        den = n_s[0] / sig_s[0] * msq["bkg"] + n_s[1] / sig_s[1] * msq["sbi"] + n_s[2] / sig_s[2] * msq["sig"]
        lam = np.where(den > 0, (n_s[src] / sig_s[src]) * own_safe / np.where(den > 0, den, 1.0), 0.0)
        # base: everything hypothesis-independent. The weight at mu is base * M(mu).
        base = g * lam
        grid = np.linspace(0.0, mu_max, n_mu)
        # M(mu): squared amplitude at signal strength mu (signal amplitude scales as sqrt(mu))
        M = lambda mu: msq["bkg"] + np.sqrt(mu) * msq["int"] + mu * msq["sig"]
        # W: (N_events, n_mu). Column j is an unbiased weighted sample of the physical distribution at GRID[j].
        self._set_weights(grid, 1.0, np.stack([base * M(mu) for mu in grid], axis=1))

        # ---------- 3. holdout split: 30% of each sample (half val, half test) ----------
        # No held-out EVENT enters a loss or a bin edge: training samples labelled events and templates from `tr`
        # only. The expected yields nu(mu) are summed over the whole simulation (see _set_weights), but they enter only
        # the scans, not the training loss.
        self._split(src, holdout, split_seed)

        # ---------- 4. features: standardize with training-set statistics ----------
        self.mean, self.std = X[self.tr].mean(0), X[self.tr].std(0) + 1e-6
        self.Xt = torch.tensor(((X - self.mean) / self.std).astype(np.float32), device=self.device)

        # ---------- 5. sampling tables, val / test templates, exact ceilings (see WeightedEventData._set_scans) ----------
        self._set_scans(seed)
        # m4l histogram baseline: 32 log-spaced bins on 180-1000 GeV, on the test set
        m4l = X[self.idx["test"], FEATS.index("4l_mass")]
        in_range = (m4l >= 180) & (m4l <= 1000)
        bins = np.clip(np.digitize(m4l, np.geomspace(180, 1000, 33)) - 1, 0, 31)
        self.q_hist = self.q_scan(bins[in_range], 32, in_range, part="test")
        print(
            f"[PhysicsData] events {len(X)}, dropped {int(bad.sum())}, "
            f"nu: mu=0 {self.nu[0]:.1f} mu=1 {self.nu[self.i_ref]:.1f} mu={mu_max:g} {self.nu[-1]:.1f}",
            flush=True,
        )

    def eval_latent(self, enc, n_cat, part="test"):
        """Scan of the argmax binning on `part`. Returns (q_latent, active_categories)."""
        k = self.categories(enc, part)
        return self.q_scan(k, n_cat, part=part), self.active_categories(k, n_cat, part)
