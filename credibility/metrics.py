"""Observables and operational regime classification.

Regime thresholds follow the validation contract (terminal mass >= 0.9):

* efficient consensus -- every community puts >= thr mass on the optimal arm;
* wrong consensus     -- every community puts >= thr mass on the *suboptimal* arm;
* polarisation        -- communities lock onto different arms;
* unresolved          -- none of the above.
"""
from __future__ import annotations

import numpy as np

REGIMES = ["efficient", "wrong", "polarised", "unresolved"]
EFFICIENT, WRONG, POLARISED, UNRESOLVED = 0, 1, 2, 3


def classify_regime(term_m, a_star, thr=0.9):
    """Classify each replication. ``term_m`` is (R, M, 2). Returns int codes (R,)."""
    term_m = np.asarray(term_m, dtype=float)
    R, M, _ = term_m.shape
    a_other = 1 - a_star
    m_star = term_m[:, :, a_star]
    m_other = term_m[:, :, a_other]
    eff = np.all(m_star >= thr, axis=1)
    wrong = np.all(m_other >= thr, axis=1)
    any0 = np.any(term_m[:, :, 0] >= thr, axis=1)
    any1 = np.any(term_m[:, :, 1] >= thr, axis=1)
    pol = any0 & any1
    code = np.full(R, UNRESOLVED, dtype=int)
    code[pol] = POLARISED
    code[wrong] = WRONG
    code[eff] = EFFICIENT
    return code


def regime_frequencies(term_m, a_star, thr=0.9):
    """Return a dict mapping regime name -> probability over replications."""
    code = classify_regime(term_m, a_star, thr)
    R = len(code)
    return {name: float(np.mean(code == i)) for i, name in enumerate(REGIMES)}


def polarisation_path(traj_m):
    """Polarisation index over time, Pi_t = mean_{c<d} |m_c,t(0) - m_d,t(0)|.

    ``traj_m`` is (R, T+1, M, 2) or (T+1, M, 2). Returns the index averaged
    over community pairs, preserving the rep and time axes.
    """
    traj_m = np.asarray(traj_m, dtype=float)
    single = traj_m.ndim == 3
    if single:
        traj_m = traj_m[None]
    m0 = traj_m[:, :, :, 0]                          # (R, T+1, M)
    R, Tt, M = m0.shape
    diffs = []
    for c in range(M):
        for d in range(c + 1, M):
            diffs.append(np.abs(m0[:, :, c] - m0[:, :, d]))
    pi = np.mean(diffs, axis=0) if diffs else np.zeros((R, Tt))
    return pi[0] if single else pi


def time_to_consensus(traj_m, thr=0.9):
    """First time index at which some arm holds >= thr mass in every community.

    Returns an int per rep (or scalar), or -1 if consensus is not reached.
    """
    traj_m = np.asarray(traj_m, dtype=float)
    single = traj_m.ndim == 3
    if single:
        traj_m = traj_m[None]
    R, Tt, M, _ = traj_m.shape
    reached_arm = np.all(traj_m >= thr, axis=2)      # (R, T+1, 2)
    reached = np.any(reached_arm, axis=2)            # (R, T+1)
    out = np.full(R, -1, dtype=int)
    for r in range(R):
        idx = np.argmax(reached[r])
        if reached[r, idx]:
            out[r] = idx
    return int(out[0]) if single else out


def correction_lag(traj_m, a_star, major=0.5):
    """Time to overturn an early wrong aggregate lead.

    Defined per rep as the first time the optimal arm holds majority mass in
    every community, given that at some earlier trial the suboptimal arm led
    the population.  NaN if there was no early wrong lead or it was never
    corrected.
    """
    traj_m = np.asarray(traj_m, dtype=float)
    single = traj_m.ndim == 3
    if single:
        traj_m = traj_m[None]
    R, Tt, M, _ = traj_m.shape
    a_other = 1 - a_star
    pop_other = traj_m[:, :, :, a_other].mean(axis=2)     # (R, T+1) population mass on wrong arm
    all_star_major = np.all(traj_m[:, :, :, a_star] > major, axis=2)   # (R, T+1)
    out = np.full(R, np.nan)
    for r in range(R):
        led = np.where(pop_other[r] > 0.5)[0]
        if len(led) == 0:
            continue
        t_lead = led[0]
        after = np.where(all_star_major[r, t_lead:])[0]
        if len(after) > 0:
            out[r] = after[0]
    return float(out[0]) if single else out


def micro_meso_discrepancy(micro_traj_m, meso_traj_m, a_star):
    """Mean absolute gap between micro ensemble-mean and meso optimal-arm mass."""
    micro_mean = np.asarray(micro_traj_m)[:, :, :, a_star].mean(axis=0)   # (T+1, M)
    meso = np.asarray(meso_traj_m)[:, :, a_star]                          # (T+1, M)
    return float(np.mean(np.abs(micro_mean - meso)))
