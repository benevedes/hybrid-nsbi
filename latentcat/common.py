"""Pieces shared by the Gaussian toy and the physics case.

Conventions used everywhere:
  * a "scan" is the Asimov expected -2 dlnL over a grid of the parameter of interest, shifted so its minimum is 0;
  * "templates" are expected counts per category (or bin) at each grid point, shape (n_bins, n_grid), and the
    column at the reference index is the Asimov "data";
  * checkpoint and model selection never use an exact / oracle likelihood, only quantities available from simulation.
"""

import os

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_device():
    """cuda if available, else Apple mps, else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def data_dir():
    """Directory with the raw inputs and caches (see data/README.md). Override with LATENTCAT_DATA."""
    return os.environ.get("LATENTCAT_DATA", os.path.join(ROOT, "data"))


def results_dir(domain):
    """results/<domain>/: run records and encoder states behind the figures."""
    d = os.path.join(ROOT, "results", domain)
    os.makedirs(d, exist_ok=True)
    return d


def figures_dir():
    d = os.path.join(ROOT, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def mlp(n_in, hidden, n_out, n_hidden_layers):
    """Event encoder: n_hidden_layers of width `hidden` with CELU, then Linear(hidden, n_out) + CELU.
    n_out is the number of latent categories; an event's category is the argmax of the output."""
    layers = [nn.Linear(n_in, hidden), nn.CELU()]
    for _ in range(n_hidden_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.CELU()]
    return nn.Sequential(*layers, nn.Linear(hidden, n_out), nn.CELU())


def paper_style():
    """Serif text and maths for the paper figures: Computer Modern from matplotlib's bundled fonts (no LaTeX needed)."""
    import matplotlib

    matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                                "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
                                # embed TrueType, not Type 3: some PDF viewers drop Type 3 glyphs such as the cmsy10 minus
                                "pdf.fonttype": 42, "ps.fonttype": 42})


def carl_mlp(n_in, n_nodes=1024, n_layers=16):
    """The CARL ratio network of scripts/physics_train_carl.py: Linear(n_in, n_nodes) + SiLU,
    then n_layers x [Linear(n_nodes, n_nodes) + SiLU], then Linear(n_nodes, 1). The output is the logit s of the
    classifier, and the ratio estimate is exp(s)."""
    layers = [nn.Sequential(nn.Linear(n_in, n_nodes), nn.SiLU())]
    layers += [nn.Sequential(nn.Linear(n_nodes, n_nodes), nn.SiLU()) for _ in range(n_layers)]
    return nn.Sequential(*layers, nn.Linear(n_nodes, 1))


def q_from_counts(lam, i_ref, floor=1e-9):
    """Asimov -2 dlnL scan of a binned Poisson likelihood.

    lam: (n_bins, n_grid) expected counts per bin at each grid point; column i_ref is the Asimov data.
    Per bin the Poisson log-likelihood is d ln(lam) - lam (the ln d! term does not depend on the parameter).
    `floor` only keeps ln(lam) finite for a bin that no event can reach at some grid point; it is deliberately far
    below any resolvable expected count, so it does not otherwise shape the scan.
    """
    lam = np.clip(np.asarray(lam, dtype=float), floor, None)
    d = lam[:, [i_ref]]
    lnL = (d * np.log(lam) - lam).sum(0)
    return 2 * (lnL.max() - lnL)


def grid_index(grid, points):
    return [int(np.argmin(np.abs(np.asarray(grid) - p))) for p in points]


def report(name, grid, q, points, q_ref=None, symbol="mu", grid_fmt="{:.2g}"):
    """One line: the scan at a few grid points, with the fraction of a reference scan in parentheses if given."""
    parts = []
    for i in grid_index(grid, points):
        frac = f" ({q[i] / q_ref[i]:.2f})" if q_ref is not None and q_ref[i] > 0 else ""
        parts.append(f"{symbol}={grid_fmt.format(grid[i])}: {q[i]:6.2f}{frac}")
    return f"{name}: " + "  ".join(parts)


def scan_score(grid, q, points):
    """Ceiling-free selection score: the scan's own mean -2 dlnL over fixed grid points."""
    return float(np.mean([q[i] for i in grid_index(grid, points)]))


@torch.no_grad()
def hard_categories(enc, X, chunk=200000):
    """argmax category of every row of X (a tensor on the encoder's device)."""
    was = enc.training
    enc.eval()
    out = [enc(X[s : s + chunk]).argmax(-1).cpu().numpy() for s in range(0, len(X), chunk)]
    enc.train(was)
    return np.concatenate(out)


class WeightedEventData:
    """Machinery shared by the Gaussian toy and the physics case: a pool of simulated events, each with an expected
    weight at every point of a parameter grid, split into a training pool and a holdout, and the holdout split in
    two halves: "val" (checkpoint selection only) and "test" (every reported number and figure).

    A subclass __init__ builds W and the raw features, then calls, in this order:
        self._set_weights(grid, ref_value, W)     W[i, j] = expected weight (in events) of event i at grid[j]
        self._split(strata, holdout, split_seed)  holdout fraction per stratum (e.g. per generated sample)
        (standardize features on self.tr and set self.Xt, a tensor on self.device)
        self._set_scans(seed)                     sampling tables, val / test templates, exact ceilings, rng

    The split seed is fixed independently of the training seed, so every run and every evaluation of a given
    dataset uses the same val and test sets.
    Subclasses define REPORT_POINTS (grid values quoted in reports and used by the selection score), SYMBOL, and
    GRID_FMT (how a grid value is printed in a report line).
    """

    REPORT_POINTS = ()
    SYMBOL = "theta"
    GRID_FMT = "{:.2g}"

    def _set_weights(self, grid, ref_value, W):
        self.GRID = np.asarray(grid, dtype=float)
        self.n_grid = len(self.GRID)
        self.i_ref = int(np.argmin(np.abs(self.GRID - ref_value)))
        assert abs(self.GRID[self.i_ref] - ref_value) < 1e-9, "the reference value must lie on the grid"
        self.W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
        assert (self.W >= 0).all()
        # nu: (n_grid,) expected total yield at each grid point, the rate term of every scan. Summed over the whole
        # simulation, val and test included: a yield is a property of the samples, not of a split. It does not enter
        # the training loss.
        self.nu = self.W.sum(0)

    def _split(self, strata, holdout, split_seed):
        """Per stratum: a `holdout` fraction is held out, half of it for validation and half for test."""
        split_rng = np.random.default_rng(split_seed)
        part = np.zeros(len(strata), np.int8)  # 0 = train, 1 = val, 2 = test
        for s in np.unique(strata):
            idx = np.where(strata == s)[0]
            held = split_rng.choice(idx, size=int(holdout * len(idx)), replace=False)
            part[held[: len(held) // 2]] = 1
            part[held[len(held) // 2 :]] = 2
        self.tr = np.where(part == 0)[0]
        self.idx = {"val": np.where(part == 1)[0], "test": np.where(part == 2)[0]}

    def _set_scans(self, seed):
        self.rng = np.random.default_rng(seed)  # all training-time sampling
        # training side: weights of training events, and CDF[:, j] over training events at grid[j] for sample_events
        self.Wtr = self.W[self.tr]
        # Fortran order so that each column CDF[:, j] is contiguous (searchsorted would otherwise copy it every call)
        self.CDF = np.asfortranarray(np.cumsum(self.Wtr / self.Wtr.sum(0, keepdims=True), axis=0))
        # val / test side: weights rescaled so each column sums to the full yield nu (units: events per dataset)
        self.Wpart, self.q_exact_part = {}, {}
        for part, idx in self.idx.items():
            w = self.W[idx] * (self.nu / self.W[idx].sum(0))[None, :]
            self.Wpart[part] = w
            # exact unbinned Asimov -2 dlnL: ln L(theta) = sum_i w_i(ref) ln w_i(theta) - nu(theta), up to a constant.
            # Only available on a benchmark with a known likelihood; reported, never used for selection.
            w1 = w[:, self.i_ref]
            keep = w1 > 0
            lnL = (w1[keep, None] * np.log(np.clip(w[keep], 1e-300, None))).sum(0) - w.sum(0)
            self.q_exact_part[part] = 2 * (lnL.max() - lnL)
        self.q_exact = self.q_exact_part["test"]

    # ---------- evaluation on the val or test set ----------
    def q_scan(self, bins, n_bins, mask=None, part="test"):
        """Asimov -2 dlnL scan of a hard binning: `bins` is the bin index of each event of `part`
        (or of the events of `part` selected by `mask`)."""
        w = self.Wpart[part] if mask is None else self.Wpart[part][mask]
        lam = np.stack([np.bincount(bins, weights=w[:, j], minlength=n_bins) for j in range(self.n_grid)], 1)
        return q_from_counts(lam, self.i_ref)

    def categories(self, enc, part="test"):
        """argmax category of every event of `part`."""
        return hard_categories(enc, self.Xt[self.idx[part]])

    def active_categories(self, k, n_cat, part="test"):
        """Number of categories holding >= 0.5 expected events at the reference point."""
        return int((np.bincount(k, weights=self.Wpart[part][:, self.i_ref], minlength=n_cat) > 0.5).sum())

    def report(self, name, q, part="test"):
        """One line at REPORT_POINTS, with the fraction of the exact ceiling of the same part (reporting only)."""
        return report(name, self.GRID, q, self.REPORT_POINTS, q_ref=self.q_exact_part[part], symbol=self.SYMBOL, grid_fmt=self.GRID_FMT)

    def score(self, q):
        """Selection score: the scan's own mean -2 dlnL at REPORT_POINTS (never normalized by q_exact)."""
        return scan_score(self.GRID, q, self.REPORT_POINTS)

    # ---------- training events ----------
    def sample_events(self, j, n):
        """Indices of n training events drawn at grid[j]: from the training pool, with probability proportional to
        their weight at grid[j]."""
        return self.tr[np.searchsorted(self.CDF[:, j], self.rng.random(n))]
