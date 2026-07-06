"""Canonical experiment definitions -- the single source of truth.

Every figure, table and the manuscript text draw their parameters from here so
that prose and code cannot drift apart.  Scenario builders take a compute
``scale`` ("preview" | "standard" | "publication") that only changes
population size, replication count and grid resolution -- never the mechanism
parameters -- so results are comparable across scales.

Operating regimes (optimal arm = arm 0, the higher-mean arm):

* ``efficient``  -- moderate transmission corrects; both communities -> arm 0.
* ``wrong``      -- an early confident wrong lead, amplified by strong
                    anticipatory transmission, locks both communities on arm 1.
* ``polarised``  -- low permeability + opposed initial leads -> divergence.
"""
from __future__ import annotations

import numpy as np
from .config import Params, SCALE


def _n(scale):
    return SCALE[scale]["N_per"]


# --------------------------------------------------------------------------
# Stage 1 -- baseline trajectory scenarios (Figure 2)
# --------------------------------------------------------------------------
def efficient(scale="standard"):
    n = _n(scale)
    return Params(sizes=(n, n), T=300, mu=(0.60, 0.40),
                  lam=0.40, eta=0.30, b_self=0.70,
                  q_init=((0.50, 0.50), (0.50, 0.50)))


def wrong(scale="standard"):
    n = _n(scale)
    # symmetric early confident lead on the suboptimal arm (arm 1)
    return Params(sizes=(n, n), T=300, mu=(0.55, 0.45),
                  lam=0.90, eta=0.20, b_self=0.70,
                  q_init=((0.45, 0.62), (0.45, 0.62)))


def polarised(scale="standard"):
    n = _n(scale)
    # opposed leads, near-isolated communities (Gamma = 0.02)
    return Params(sizes=(n, n), T=300, mu=(0.55, 0.45),
                  lam=0.90, eta=0.25, b_self=0.98,
                  q_init=((0.62, 0.42), (0.40, 0.64)))


BASELINE = {"efficient": efficient, "wrong": wrong, "polarised": polarised}


# --------------------------------------------------------------------------
# Stage 3 -- connectivity phase diagram (Figure 3)
# --------------------------------------------------------------------------
def phase_base(scale="standard"):
    """Asymmetric base: community 2 carries an early wrong lead."""
    n = SCALE[scale].get("N_phase", 100)
    return Params(sizes=(n, n), T=240, mu=(0.55, 0.45), eta=0.25,
                  q_init=((0.52, 0.50), (0.40, 0.64)))


def phase_grid(scale="standard"):
    g = SCALE[scale]["grid"]
    lams = np.linspace(0.0, 1.6, g)
    gammas = np.linspace(0.01, 0.50, g)            # permeability Gamma = 1 - b_self
    reps = SCALE[scale]["reps_phase"]
    return lams, gammas, reps


# --------------------------------------------------------------------------
# Stage 2 -- ablations (Figure 4)
# --------------------------------------------------------------------------
def ablation_point(scale="standard"):
    """A contested operating point where the social mechanism is decisive."""
    n = _n(scale)
    return Params(sizes=(n, n), T=260, mu=(0.55, 0.45),
                  lam=0.60, eta=0.30, b_self=0.85,
                  q_init=((0.52, 0.50), (0.42, 0.62)))


def ablations(base: Params):
    """Return {name: Params} for the full model and five decisive ablations."""
    return {
        "full": base,
        "no-anticipatory": base.replace(lam=0.0),
        "no-retrospective": base.replace(eta=0.0),
        "no-confidence-wt": base.replace(credibility_weight=False,
                                         social_lr_mode="constant"),
        "constant-lr": base.replace(private_lr_mode="constant",
                                    social_lr_mode="constant"),
        "alt-confidence": base.replace(confidence_mode="balance"),
    }


# --------------------------------------------------------------------------
# Stage 2/5 -- confidence mechanism (Figure 5)
# --------------------------------------------------------------------------
def confidence_scenario(scale="standard"):
    # neutral-init, contested gap: early choices are genuinely ambiguous, so
    # confidence spans a wide range and attaches to both correct and wrong
    # early choices before the optimal arm is identified.
    n = min(_n(scale), 150)
    return Params(sizes=(n, n), T=260, mu=(0.55, 0.45),
                  lam=0.50, eta=0.25, b_self=0.85,
                  q_init=((0.50, 0.50), (0.50, 0.50)))


# --------------------------------------------------------------------------
# Stage 4 -- quotient validation (Figure 6)
# --------------------------------------------------------------------------
def quotient_trajectories(scale="standard"):
    """Two-community scenarios for micro vs meso trajectory overlay."""
    return {"efficient": efficient(scale), "wrong": wrong(scale)}


def quotient_many(scale="standard"):
    """A four-community modular example for the quotient (Stage 4, M>2)."""
    n = SCALE[scale].get("N_phase", 100)
    # ring-like modular coupling: cohesion 0.7, leak to two neighbours.
    B = np.array([
        [0.70, 0.15, 0.00, 0.15],
        [0.15, 0.70, 0.15, 0.00],
        [0.00, 0.15, 0.70, 0.15],
        [0.15, 0.00, 0.15, 0.70],
    ])
    q_init = ((0.55, 0.47), (0.47, 0.55), (0.55, 0.47), (0.47, 0.55))
    return Params(sizes=(n, n, n, n), T=240, mu=(0.55, 0.45),
                  lam=0.6, eta=0.25, B=tuple(map(tuple, B)), q_init=q_init)


def quotient_grid(scale="standard"):
    g = min(SCALE[scale]["grid"], 11)
    lams = np.linspace(0.0, 1.6, g)
    gammas = np.linspace(0.01, 0.50, g)
    reps = max(60, SCALE[scale]["reps_quotient"] // 3)
    n = min(SCALE[scale].get("N_phase", 100), 100)
    base = phase_base(scale).replace(sizes=(n, n), T=200)
    return base, lams, gammas, reps
