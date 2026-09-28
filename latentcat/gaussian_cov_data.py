"""Covariance Gaussian toy: x in R^5 with mean 0 and covariance Sigma(theta) = I except
Sigma_01 = Sigma_10 = a theta and Sigma_23 = Sigma_32 = b theta (a != b). theta in [-theta_max, theta_max], nominal 0
(where Sigma = I); each dataset has N ~ Poisson(50) events.

In the rotated coordinates u+- = (x0 +- x1)/sqrt2, v+- = (x2 +- x3)/sqrt2 the covariance is diagonal with four distinct
theta-dependent variances lambda = 1 +- a theta, 1 +- b theta (x4 carries no information). The log likelihood ratio
of theta against the nominal value is linear in (u+^2, u-^2, v+^2, v-^2) with weights that rotate with theta, so no
fixed 1D statistic is sufficient over the range. The histogram baseline bins the statistic that is optimal at the
nominal value, the score t(x) = d/dtheta log p_theta(x) at theta = 0 = [a (u+^2 - u-^2) + b (v+^2 - v-^2)] / 2
= a x0 x1 + b x2 x3.

Built on the same WeightedEventData machinery as the other toys:
  1. Simulate `n_per_point` events at each of 11 generation points theta_k spanning the range and pool them.
  2. Balance-heuristic reweighting to any theta grid, columns renormalized to n_exp.
  3. A 30% holdout per generation point from a fixed split seed, halved into "val" and "test".
  4. Features standardized with training-set statistics.
  5. Test-set reference scans: exact unbinned likelihood (checked against the closed form
     N sum_k [r_k - 1 - log r_k], r_k = lambda_k(0)/lambda_k(theta)), the 32-bin histogram of t (plotted), and, printed
     only, t with 200 quantile bins and the one-block statistic x0 x1 with 32 bins.
"""

import numpy as np
import torch
from scipy.special import logsumexp

from .common import WeightedEventData

D = 5
THETA_REF = 0.0


def variances(theta, a, b):
    """lambda(theta) for (u+, u-, v+, v-)."""
    return np.array([1 + a * theta, 1 - a * theta, 1 + b * theta, 1 - b * theta])


def rotated(X):
    """(u+, u-, v+, v-) for each row of X."""
    s = np.sqrt(0.5)
    return np.stack([s * (X[:, 0] + X[:, 1]), s * (X[:, 0] - X[:, 1]), s * (X[:, 2] + X[:, 3]), s * (X[:, 2] - X[:, 3])], 1)


def log_density(Z2, x4sq, theta, a, b):
    """log p_theta(x) up to a theta-independent constant, from the squared rotated coordinates Z2 = (u+^2, u-^2, v+^2, v-^2)."""
    lam = variances(theta, a, b)
    return -0.5 * (np.log(lam).sum() + (Z2 / lam).sum(1) + x4sq)


def exact_q(grid, a, b, n_exp):
    """Closed-form Asimov -2 dlnL: 2 n_exp KL(p_0 || p_theta) = n_exp sum_k [r_k - 1 - log r_k], r_k = lambda_k(0)/lambda_k(theta)."""
    out = []
    for t in np.asarray(grid, dtype=float):
        r = variances(THETA_REF, a, b) / variances(t, a, b)
        out.append(n_exp * np.sum(r - 1 - np.log(r)))
    return np.array(out)


class GaussianCovData(WeightedEventData):
    SYMBOL = "theta"
    GRID_FMT = "{:+.2f}"

    def __init__(
        self,
        a=0.8,
        b=0.4,
        theta_max=1.0,
        n_grid=11,
        n_exp=50.0,
        n_per_point=300000,
        holdout=0.3,
        sim_seed=1,
        split_seed=11,
        seed=0,
        device="cpu",
    ):
        assert a != b and max(abs(a), abs(b)) * theta_max < 1, "need a != b and a positive-definite covariance"
        self.device = torch.device(device)
        self.n_exp = n_exp
        self.REPORT_POINTS = (-theta_max, -theta_max / 2, theta_max / 2, theta_max)

        # ---------- 1. simulate at the generation points and pool ----------
        gen = np.linspace(-theta_max, theta_max, 11)
        sim_rng = np.random.default_rng(sim_seed)
        s = np.sqrt(0.5)
        parts = []
        for t in gen:
            z = sim_rng.standard_normal((n_per_point, 4)) * np.sqrt(variances(t, a, b))  # (u+, u-, v+, v-)
            x4 = sim_rng.standard_normal(n_per_point)
            parts.append(np.stack([s * (z[:, 0] + z[:, 1]), s * (z[:, 0] - z[:, 1]), s * (z[:, 2] + z[:, 3]), s * (z[:, 2] - z[:, 3]), x4], 1))
        X = np.concatenate(parts)
        gen_id = np.repeat(np.arange(len(gen)), n_per_point)

        # ---------- 2. balance-heuristic reweighting to every grid point ----------
        Z2, x4sq = rotated(X) ** 2, X[:, 4] ** 2
        log_mix = logsumexp(np.stack([log_density(Z2, x4sq, t, a, b) for t in gen]), axis=0) + np.log(n_per_point)
        grid = np.linspace(-theta_max, theta_max, n_grid)
        W = n_exp * np.exp(np.stack([log_density(Z2, x4sq, t, a, b) - log_mix for t in grid], axis=1))
        W *= n_exp / W.sum(0)  # the true yield is exactly n_exp at every theta
        self._set_weights(grid, THETA_REF, W)

        # ---------- 3. holdout split into val and test ----------
        self._split(gen_id, holdout, split_seed)

        # ---------- 4. features ----------
        t = a * X[:, 0] * X[:, 1] + b * X[:, 2] * X[:, 3]  # the score at theta = 0, the nominal optimal statistic
        x0x1 = X[:, 0] * X[:, 1]
        X = X.astype(np.float32)
        self.mean, self.std = X[self.tr].mean(0), X[self.tr].std(0)
        self.Xt = torch.tensor((X - self.mean) / self.std, device=self.device)

        # ---------- 5. sampling tables, val / test templates, exact ceilings, histograms ----------
        self._set_scans(seed)
        q_closed_form = exact_q(self.GRID, a, b, n_exp)
        self.q_hist = self.hist_scan(t, n_bins=32)  # plotted baseline
        self.q_hist_t_fine = self.hist_scan(t, n_bins=200, quantile=True)
        self.q_hist_x0x1 = self.hist_scan(x0x1, n_bins=32)
        far = q_closed_form > 1.0
        dev = np.max(np.abs(self.q_exact[far] / q_closed_form[far] - 1)) if far.any() else 0.0
        print(
            f"[GaussianCovData a={a} b={b}] events {len(X)}, val {len(self.idx['val'])}, test {len(self.idx['test'])}; "
            f"exact test scan vs closed form: max rel. dev. {dev:.3f}",
            flush=True,
        )

    def hist_scan(self, values, n_bins, quantile=False, part="test"):
        """Test-set scan of a 1D histogram of `values` (one per event). Edges from the training pool: uniform between
        the 0.005 and 99.995 percentiles, or at equal-weight quantiles of the reference distribution."""
        v_tr = values[self.tr]
        v = values[self.idx[part]]
        if quantile:
            w = self.Wtr[:, self.i_ref]
            order = np.argsort(v_tr)
            cdf = np.cumsum(w[order]) / w.sum()
            inner = v_tr[order][np.searchsorted(cdf, np.linspace(0, 1, n_bins + 1)[1:-1])]
            return self.q_scan(np.searchsorted(inner, v), n_bins, part=part)
        lo, hi = np.percentile(v_tr, [0.005, 99.995])
        sel = (v >= lo) & (v <= hi)
        bins = np.clip(np.digitize(v, np.linspace(lo, hi, n_bins + 1)) - 1, 0, n_bins - 1)
        return self.q_scan(bins[sel], n_bins, sel, part=part)

    def eval_latent(self, enc, n_cat, part="test"):
        """Scan of the argmax binning on `part`. Returns (q_latent, active_categories)."""
        k = self.categories(enc, part)
        return self.q_scan(k, n_cat, part=part), self.active_categories(k, n_cat, part)
