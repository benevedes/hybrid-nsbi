"""Toy coverage of the -2 dlnL interval: profile-likelihood maxima on a grid, and the coverage-curve figure.

For a nominal coverage alpha the interval is {theta : -2 dlnL(theta) < F^-1_chi2_1(alpha)}, so a pseudo-experiment's
interval covers the true value theta0 when q0 = 2 (max lnL - lnL(theta0)) is below that threshold. The observed
coverage at alpha is the fraction of pseudo-experiments that do.
"""

import numpy as np
from scipy.stats import chi2


def refine_max(lnL, grid):
    """Maximum of each row of lnL (pseudo-experiments x grid points) and its position, from the parabola through the
    grid maximum and its two neighbours.

    On the bare grid, a fraction ~ step / (sqrt(2 pi) sigma(theta-hat)) of pseudo-experiments has its maximum exactly
    on theta0 and so q0 = 0 exactly, which shows up as excess coverage at small alpha. Near its maximum lnL is
    quadratic to a good approximation, so the parabola removes this; rows whose three points are not concave keep
    the grid maximum, and the refined maximum is never below the grid maximum.
    """
    j = np.clip(lnL.argmax(1), 1, len(grid) - 2)
    r = np.arange(len(lnL))
    lm, l0, lp = lnL[r, j - 1], lnL[r, j], lnL[r, j + 1]
    curv = lm - 2 * l0 + lp
    dx = np.clip(np.where(curv < 0, 0.5 * (lm - lp) / np.where(curv < 0, curv, -1.0), 0.0), -1, 1)
    lnL_max = np.maximum(l0 - 0.25 * (lm - lp) * dx, lnL.max(1))
    return lnL_max, grid[j] + dx * (grid[1] - grid[0])


def summarize(q0, theta_hat):
    """Coverage at 1 and 2 sigma (q0 < 1 and q0 < 4: nominal 68.27% and 95.45%) and the spread of theta-hat."""
    return dict(cov_1sig=float((q0 < 1).mean()), cov_2sig=float((q0 < 4).mean()), mean_hat=float(theta_hat.mean()),
                std_hat=float(theta_hat.std()))


def plot_coverage(q0, curves, out, ylim=(-0.055, 0.045)):
    """Observed vs expected coverage (top, square: both axes are coverages on [0, 1]) and observed - expected (bottom),
    with the +-1 sigma binomial error of the pseudo-experiment statistics around each curve in both panels.

    q0: {key: q0 of every pseudo-experiment}; curves: [(label, key, line style)], in legend order.
    """
    import matplotlib.pyplot as plt

    from .common import paper_style

    alpha = np.linspace(0.0, 1.0, 1001)[1:-1]
    thr = chi2.ppf(alpha, df=1)
    paper_style()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.0, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[1, 0.32], hspace=0.06))
    ax.set_box_aspect(1)
    bx.set_box_aspect(0.32)
    nominal = dict(color="0.15", lw=1.1, ls=(0, (1, 1.6)), zorder=20)  # drawn on top: the calibrated curves cover it
    ax.plot([0, 1], [0, 1], label="Nominal", **nominal)
    bx.axhline(0, **nominal)
    for label, key, style in curves:
        obs = (q0[key][None, :] < thr[:, None]).mean(1)
        err = np.sqrt(obs * (1 - obs) / len(q0[key]))
        ax.plot(alpha, obs, label=label, **style)
        ax.fill_between(alpha, obs - err, obs + err, color=style["color"], alpha=0.18, lw=0, zorder=1)
        bx.plot(alpha, obs - alpha, **style)
        bx.fill_between(alpha, obs - alpha - err, obs - alpha + err, color=style["color"], alpha=0.18, lw=0, zorder=1)
    bx.fill_between([], [], [], color="0.5", alpha=0.3, lw=0, label=r"shading: $\pm 1\sigma$ toy statistics")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    bx.set_ylim(*ylim)
    ax.set_ylabel("Observed coverage")
    bx.set_xlabel("Expected coverage")
    bx.set_ylabel("Observed $-$ expected")
    for a in (ax, bx):
        a.minorticks_on()
        a.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    bx.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
