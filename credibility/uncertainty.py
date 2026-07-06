"""Monte Carlo uncertainty quantification.

Two families of estimand are reported in the paper:

* binary regime probabilities -- frequencies over independent replications.
  Reported with **Wilson score** 95% intervals (good small-sample coverage and
  never leaves [0,1]); a nonparametric bootstrap interval is also available.

* means (group regret, correction lag, discrepancy) -- reported with the
  Monte Carlo standard error and a nonparametric **percentile bootstrap** 95%
  interval over replications.

For the ablation contrasts we use **common random numbers** (the same seed, and
hence the same reward/diffusion shocks, across model variants), so the
replication-paired difference identifies the effect of disabling a mechanism
component with variance reduction.  Paired differences are summarised by the
paired bootstrap (resampling replication indices jointly).
"""
from __future__ import annotations

import numpy as np

Z95 = 1.959963984540054


def wilson_interval(k, n, z=Z95):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def proportion_summary(indicator, z=Z95):
    """(p, lo, hi, k, n) for a boolean/0-1 replication array."""
    x = np.asarray(indicator, dtype=float)
    n = x.size
    k = float(x.sum())
    lo, hi = wilson_interval(k, n, z)
    return dict(p=(k / n if n else float("nan")), lo=lo, hi=hi, k=k, n=n)


def bootstrap_mean(x, n_boot=4000, alpha=0.05, seed=0):
    """Percentile-bootstrap mean with Monte Carlo SE. NaNs dropped."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"),
                    se=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = x[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    se = x.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return dict(mean=float(x.mean()), lo=float(lo), hi=float(hi),
                se=float(se), n=int(n))


def paired_bootstrap_diff(a, b, n_boot=4000, alpha=0.05, seed=0):
    """Paired difference mean(a-b) with bootstrap CI over common replications.

    ``a`` (full model) and ``b`` (ablation) are aligned per replication under
    common random numbers.  Pairs with a non-finite entry are dropped.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    n = d.size
    if n == 0:
        return dict(diff=float("nan"), lo=float("nan"), hi=float("nan"),
                    se=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return dict(diff=float(d.mean()), lo=float(lo), hi=float(hi),
                se=float(se), n=int(n))
