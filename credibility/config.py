"""Parameter container and named presets.

A single :class:`Params` dataclass holds every behavioural, social and network
parameter.  Ablations are expressed as light modifications of a base
``Params`` via :meth:`Params.replace`, so that the *same* simulator runs the
full model and every ablation (no parallel code paths).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace, asdict
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class Params:
    # --- environment -------------------------------------------------------
    sizes: Tuple[int, ...] = (200, 200)      # community sizes (N_1, ..., N_M)
    T: int = 300                              # number of trials
    mu: Tuple[float, float] = (0.6, 0.5)      # Bernoulli arm means (arm1, arm2)

    # --- coupling matrix B (row-stochastic, M x M) -------------------------
    b_self: float = 0.85                      # diagonal cohesion (symmetric B)
    B: Tuple[Tuple[float, ...], ...] = None   # explicit B overrides b_self

    # --- social channel strengths -----------------------------------------
    lam: float = 0.6                          # anticipatory weight lambda
    eta: float = 0.3                          # retrospective weight eta

    # --- DDM ---------------------------------------------------------------
    beta: float = 6.0                         # drift scaling v = beta * Delta
    sigma: float = 1.0                        # diffusion s.d.
    a_thr: float = 1.0                        # symmetric boundary a
    rt_dispersion: float = 0.3                # squared CV of reaction times

    # --- confidence --------------------------------------------------------
    confidence_mode: str = "decision"         # "decision" | "balance"
    rt_mode: str = "ig"                       # "ig" (stochastic) | "mean" (deterministic MFPT)
    kappa1: float = 3.0                       # weight on evidence strength
    kappa2: float = 1.0                       # weight on decision time
    tau0: float = 0.5                         # reaction-time scale

    # --- learning rates ----------------------------------------------------
    private_lr_mode: str = "confidence"       # "confidence" | "constant"
    alpha_min: float = 0.05
    alpha_max: float = 0.40
    alpha_const: float = 0.20                 # used when private_lr_mode=="constant"

    social_lr_mode: str = "confidence"        # "confidence" | "constant"
    gamma: float = 0.5                        # social LR scale
    omega: float = 1.0                        # social LR confidence exponent
    eps_soc: float = 1e-3                     # epsilon in Cbar denominator

    # --- transmission switch ----------------------------------------------
    credibility_weight: bool = True           # weight transmitted actions by C

    # --- initial values ----------------------------------------------------
    # q_init[c] = (Q_c0(arm1), Q_c0(arm2)).  If None, all arms start at 0.5.
    q_init: Tuple[Tuple[float, float], ...] = None

    # ----------------------------------------------------------------------
    @property
    def M(self) -> int:
        return len(self.sizes)

    @property
    def N(self) -> int:
        return int(sum(self.sizes))

    def coupling(self) -> np.ndarray:
        """Return the M x M row-stochastic coupling matrix B."""
        if self.B is not None:
            Bm = np.asarray(self.B, dtype=float)
        else:
            M = self.M
            Bm = np.full((M, M), (1.0 - self.b_self) / (M - 1)) if M > 1 else np.array([[1.0]])
            np.fill_diagonal(Bm, self.b_self)
        return Bm

    def q_init_array(self) -> np.ndarray:
        """Return an (M, 2) array of community initial values."""
        if self.q_init is None:
            return np.full((self.M, 2), 0.5)
        return np.asarray(self.q_init, dtype=float)

    def replace(self, **kw) -> "Params":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Named presets.  Compute scale only differs in N, reps and grid resolution;
# the mechanism parameters are shared so figures are comparable across scales.
# --------------------------------------------------------------------------
SCALE = {
    "preview":     dict(N_per=80,  N_phase=50,  reps_traj=120,  reps_phase=60,  grid=9,  reps_quotient=100),
    "standard":    dict(N_per=200, N_phase=80,  reps_traj=300,  reps_phase=120, grid=13, reps_quotient=200),
    "publication": dict(N_per=500, N_phase=250, reps_traj=1000, reps_phase=600, grid=21, reps_quotient=800),
}


def base_params(scale: str = "standard") -> Params:
    """Baseline two-community parameters at the requested compute scale."""
    n = SCALE[scale]["N_per"]
    return Params(sizes=(n, n))
