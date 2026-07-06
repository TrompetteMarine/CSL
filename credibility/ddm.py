"""Drift--diffusion choice engine.

Documented implementation choice (Stage 1 of the validation contract).
We use *Option 1*: an **exact choice probability** combined with an
**approximate reaction time**.

For a Wiener process ``X`` with ``X(0)=0``, drift ``v``, infinitesimal
standard deviation ``sigma`` and symmetric absorbing barriers at ``+a`` and
``-a``:

* Probability of absorption at ``+a`` (arm 1) before ``-a`` (arm 2)

    P(+a) = 1 / (1 + exp(-2 v a / sigma^2))  =  logistic(2 v a / sigma^2)   [EXACT]

  This follows from the gambler's-ruin solution of the Wiener first-passage
  problem and is exact, not an approximation.

* Mean first-passage (decision) time

    E[tau] = (a / v) * tanh(a v / sigma^2),   v != 0                        [EXACT]
           = a^2 / sigma^2,                    v  = 0.

The decision *time* feeding the confidence map is drawn from an
inverse-Gaussian (Wald) law whose mean equals ``E[tau]`` and whose squared
coefficient of variation is a fixed dispersion ``rt_dispersion``.  This keeps
reaction times stochastic and strictly positive while matching the analytic
central tendency; it avoids the cost of stepping every diffusion path.

An Euler--Maruyama first-passage sampler is provided for *verification only*
(``euler_maruyama_fpt``); the test suite checks that the analytic choice
probability and mean decision time agree with direct simulation.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

_TINY = 1e-12


def choice_prob_up(v, a, sigma):
    """Exact probability of hitting the +a boundary (choosing arm 1)."""
    v = np.asarray(v, dtype=float)
    return expit(2.0 * v * a / (sigma * sigma))


def mean_fpt(v, a, sigma):
    """Exact mean first-passage time for symmetric +/-a barriers, start at 0.

    Uses the numerically stable form ``(a^2/sigma^2) * tanh(u)/u`` with
    ``u = a v / sigma^2`` and the removable singularity ``tanh(u)/u -> 1`` as
    ``u -> 0`` handled by a short Taylor expansion.
    """
    v = np.asarray(v, dtype=float)
    s2 = sigma * sigma
    u = a * v / s2
    small = np.abs(u) < 1e-6
    ratio = np.where(small, 1.0 - u * u / 3.0, np.tanh(np.where(small, 1.0, u)) / np.where(small, 1.0, u))
    return (a * a / s2) * ratio


def sample_decision_time(v, a, sigma, rng, rt_dispersion=0.3):
    """Draw a decision time per trial from a Wald law matched to ``mean_fpt``.

    ``rt_dispersion`` is the squared coefficient of variation Var/mean^2, held
    constant across drifts (shape parameter lambda = mean / rt_dispersion).
    """
    mean = np.maximum(mean_fpt(v, a, sigma), _TINY)
    scale = mean / max(rt_dispersion, _TINY)
    return rng.wald(mean, scale)


def simulate_choice(v, a, sigma, rng):
    """Sample actions from the exact choice probability.

    Returns
    -------
    action : int array in {1, 2}     (1 = +a boundary, 2 = -a boundary)
    p_chosen : float array           balance-of-evidence for the chosen arm,
                                      i.e. P(chosen boundary) in [0.5, 1).
    p_up : float array               P(arm 1).
    """
    p_up = choice_prob_up(v, a, sigma)
    up = rng.random(np.shape(p_up)) < p_up
    action = np.where(up, 1, 2)
    p_chosen = np.where(up, p_up, 1.0 - p_up)
    return action, p_chosen, p_up


def euler_maruyama_fpt(v, a, sigma, rng, dt=1e-3, max_steps=200000):
    """Reference Euler--Maruyama first-passage sampler (verification only).

    Vectorised over an array of drifts ``v``.  Returns ``(action, tau)`` with
    ``action`` in {1, 2} and ``tau`` the absorption time.  Used by the test
    suite to confirm that :func:`choice_prob_up` and :func:`mean_fpt` match a
    direct simulation of the diffusion.
    """
    v = np.atleast_1d(np.asarray(v, dtype=float))
    x = np.zeros_like(v)
    tau = np.full_like(v, np.nan)
    action = np.zeros_like(v, dtype=int)
    active = np.ones_like(v, dtype=bool)
    sqrt_dt = np.sqrt(dt)
    for step in range(1, max_steps + 1):
        n = active.sum()
        if n == 0:
            break
        x[active] += v[active] * dt + sigma * sqrt_dt * rng.standard_normal(n)
        hit_up = active & (x >= a)
        hit_dn = active & (x <= -a)
        t_now = step * dt
        if hit_up.any():
            tau[hit_up] = t_now
            action[hit_up] = 1
            active &= ~hit_up
        if hit_dn.any():
            tau[hit_dn] = t_now
            action[hit_dn] = 2
            active &= ~hit_dn
    # Any path not absorbed by max_steps: assign by current sign.
    if active.any():
        action[active] = np.where(x[active] >= 0, 1, 2)
        tau[active] = max_steps * dt
    return action, tau
