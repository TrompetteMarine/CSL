"""Neuroeconomic figures for Paper I (decision->confidence readout; confidence-
gated plasticity).  All curves are the *exact* model maps, not fits:

  drift            v      = beta * Delta                         (config.beta)
  choice prob      P(+a)  = logistic(2 v a / sigma^2)            (ddm, EXACT)
  mean decision t  E[tau] = (a/v) tanh(a v / sigma^2)            (ddm, EXACT)
  confidence       C      = sigmoid(k1 |v|/a - k2 log(1+tau/tau0))  (confidence.py)
  private LR       alpha  = alpha_min + span*C   if delta<0      (model._private_lr)
                          = alpha_max - span*C   if delta>=0
  social LR        gamma * Cbar^omega                            (model._social_lr)

Writes fig8_decision_confidence.{pdf,png} and fig9_confidence_plasticity.{pdf,png}.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common
import plotstyle
from credibility.config import Params

plotstyle.use()
_PAL = plotstyle.PALETTE
BLUE, ORANGE, GREEN, RED = _PAL["blue"], _PAL["orange"], _PAL["green"], _PAL["red"]
P = Params()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def confidence(v_abs, tau):
    z = P.kappa1 * (v_abs / P.a_thr) - P.kappa2 * np.log1p(tau / P.tau0)
    return sigmoid(z)


def mean_tau(v):
    u = P.a_thr * v / P.sigma**2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(u) < 1e-8, 1.0, np.tanh(u) / np.where(u == 0, 1, u))
    return (P.a_thr**2 / P.sigma**2) * ratio


# ==========================================================================
def fig8_decision_confidence():
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(9.8, 3.0))

    # ---- (a) accumulation-to-bound sample paths -------------------------
    rng = np.random.default_rng(7)
    a = P.a_thr
    dt, T = 0.004, 1.6
    steps = int(T / dt)
    t = np.arange(steps + 1) * dt
    scenarios = [(P.beta * 0.20, BLUE, "strong evidence"),
                 (P.beta * 0.05, ORANGE, "weak evidence")]
    a1.axhline(a, color="k", lw=1.0)
    a1.axhline(-a, color="k", lw=1.0)
    a1.text(0.02, a + 0.04, r"upper bound $+a$ (choose $A{=}1$)", fontsize=6.8)
    a1.text(0.02, -a - 0.16, r"lower bound $-a$", fontsize=6.8)
    handles = []
    for v, col, lab in scenarios:
        # faint sample paths
        for k in range(3):
            x = np.zeros(steps + 1)
            hit = steps
            for i in range(steps):
                x[i + 1] = x[i] + v * dt + P.sigma * np.sqrt(dt) * rng.standard_normal()
                if abs(x[i + 1]) >= a:
                    x[i + 1] = np.sign(x[i + 1]) * a
                    hit = i + 1
                    break
            a1.plot(t[:hit + 1], x[:hit + 1], color=col, lw=0.8, alpha=0.45)
        # bold mean-drift guide (expected trajectory), clipped at the bound
        mean = np.clip(v * t, -a, a)
        h, = a1.plot(t, mean, color=col, lw=2.4, label=lab)
        handles.append(h)
    a1.set_xlim(0, 1.2)
    a1.set_ylim(-a - 0.28, a + 0.28)
    a1.set_xlabel("decision time")
    a1.set_ylabel("accumulated evidence")
    a1.set_title(r"(a) evidence accumulation")
    a1.legend(handles=handles, loc="lower right", fontsize=6.4,
              frameon=True, framealpha=0.92, edgecolor="0.8")
    a1.grid(False)

    # ---- (b) confidence readout surface C(|v|, tau) ---------------------
    v_abs = np.linspace(0.0, 3.0, 200)
    taus = np.linspace(0.02, 1.2, 200)
    VV, TT = np.meshgrid(v_abs, taus)
    CC = confidence(VV, TT)
    im = a2.contourf(VV, TT, CC, levels=np.linspace(0, 1, 21), cmap="viridis")
    # overlay the model's own tau(|v|) ridge (what the process actually visits)
    vpos = np.linspace(0.05, 3.0, 100)
    a2.plot(vpos, mean_tau(vpos), color="w", lw=2.0, ls="--",
            label=r"model ridge $E[\tau\,|\,v]$")
    a2.set_xlabel(r"evidence strength $|v|=\beta|\Delta|$")
    a2.set_ylabel(r"decision time $\tau$")
    a2.set_title(r"(b) confidence readout $C$")
    a2.legend(loc="upper right", fontsize=6.8, labelcolor="w")
    a2.grid(False)
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.03)
    cb.set_label(r"$C=\sigma(\kappa_1|v|/a-\kappa_2\log(1+\tau/\tau_0))$", fontsize=6.6)

    # ---- (c) choice accuracy + realised confidence vs evidence ----------
    Delta = np.linspace(-0.5, 0.5, 300)
    v = P.beta * Delta
    pacc = sigmoid(2 * v * P.a_thr / P.sigma**2)
    C_real = confidence(np.abs(v), mean_tau(np.where(np.abs(v) < 1e-6, 1e-6, np.abs(v))))
    a3.plot(Delta, pacc, color=GREEN, label=r"choice prob. $P(+a)$")
    a3.plot(Delta, C_real, color=BLUE, ls="--",
            label=r"confidence $C(v,E[\tau])$")
    a3.axvline(0, color="k", lw=0.6, alpha=0.4)
    a3.set_xlabel(r"value contrast $\Delta=Q(1)-Q(0)$")
    a3.set_ylabel("probability")
    a3.set_ylim(0, 1.02)
    a3.set_title("(c) accuracy vs confidence")
    a3.legend(loc="lower right", fontsize=7)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{common.FIGDIR}/fig8_decision_confidence.{ext}", dpi=300)
    plt.close(fig)
    print("wrote fig8_decision_confidence")


# ==========================================================================
def fig9_confidence_plasticity():
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(9.8, 3.0))
    C = np.linspace(0, 1, 200)
    span = P.alpha_max - P.alpha_min

    # ---- (a) confidence-gated valence-asymmetric plasticity -------------
    a_pos = P.alpha_max - span * C          # good news (delta>=0)
    a_neg = P.alpha_min + span * C          # bad news  (delta<0)
    a1.plot(C, a_pos, color=GREEN, label=r"positive PE ($\delta\geq0$)")
    a1.plot(C, a_neg, color=RED, label=r"negative PE ($\delta<0$)")
    a1.fill_between(C, a_pos, a_neg, color="k", alpha=0.05)
    a1.set_xlabel(r"decision confidence $C$")
    a1.set_ylabel(r"private learning rate $\alpha$")
    a1.set_title("(a) confidence-gated plasticity")
    a1.legend(loc="upper center", fontsize=6.6, frameon=True, framealpha=0.93,
              edgecolor="0.8")

    # ---- (b) valence-asymmetry index ------------------------------------
    asym = a_neg - a_pos                      # >0 => bad news weighted more
    a2.axhline(0, color="k", lw=0.6, alpha=0.5)
    a2.plot(C, asym, color=BLUE)
    a2.fill_between(C, 0, asym, where=asym >= 0, color=BLUE, alpha=0.12)
    a2.fill_between(C, 0, asym, where=asym < 0, color=ORANGE, alpha=0.12)
    a2.set_xlabel(r"decision confidence $C$")
    a2.set_ylabel(r"asymmetry $\alpha^--\alpha^+$")
    a2.set_title("(b) optimism $\\leftrightarrow$ revision")

    # ---- (c) social credibility weight + social LR ----------------------
    # credibility weight of a neighbour with confidence C among peers ~U(0,1)
    peer_mean = 0.5
    w_rel = C / (C + peer_mean + P.eps_soc)      # relative pull vs an average peer
    a3.plot(C, w_rel, color=BLUE, label=r"credibility weight $C/(C+\bar C)$")
    for om, ls in [(1.0, "-"), (2.0, "--")]:
        a3.plot(C, P.gamma * C**om, color=ORANGE, ls=ls,
                label=fr"social LR $\gamma C^{{{om:.0f}}}$")
    a3.set_xlabel(r"sender confidence $C$")
    a3.set_ylabel("social transmission")
    a3.set_title("(c) credibility as network weight")
    a3.legend(loc="lower right", fontsize=6.6, frameon=True, framealpha=0.93,
              edgecolor="0.8")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{common.FIGDIR}/fig9_confidence_plasticity.{ext}", dpi=300)
    plt.close(fig)
    print("wrote fig9_confidence_plasticity")


def fig_plasticity_quant():
    """Quantitative companion to the pedagogical Fig.~3: the exact learning-rate
    law and the credibility/social-transmission curves."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    C = np.linspace(0, 1, 200)
    span = P.alpha_max - P.alpha_min

    # (a) valence-asymmetric private learning rate
    a_pos = P.alpha_max - span * C
    a_neg = P.alpha_min + span * C
    a1.plot(C, a_pos, color=GREEN, label=r"good news ($\delta\geq0$)")
    a1.plot(C, a_neg, color=RED, label=r"bad news ($\delta<0$)")
    a1.fill_between(C, a_pos, a_neg, color="k", alpha=0.05)
    a1.axvline(0.5, color="0.5", lw=0.8, ls=":")
    a1.set_xlabel(r"decision confidence $C$")
    a1.set_ylabel(r"private learning rate $\alpha$")
    a1.set_title("(a) confidence-gated learning rate")
    a1.legend(loc="center right", fontsize=8, frameon=True, framealpha=0.93, edgecolor="0.8")

    # (b) credibility weight + social learning rate
    peer_mean = 0.5
    w_rel = C / (C + peer_mean + P.eps_soc)
    a2.plot(C, w_rel, color=BLUE, label=r"credibility weight $C/(C+\bar C)$")
    for om, ls in [(1.0, "-"), (2.0, "--")]:
        a2.plot(C, P.gamma * C**om, color=ORANGE, ls=ls, label=fr"social LR $\gamma C^{{{om:.0f}}}$")
    a2.set_xlabel(r"sender confidence $C$")
    a2.set_ylabel("social transmission")
    a2.set_title("(b) credibility as network weight")
    a2.legend(loc="upper left", fontsize=7.6, frameon=True, framealpha=0.93, edgecolor="0.8")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{common.FIGDIR}/fig_plasticity_quant.{ext}", dpi=300)
    plt.close(fig)
    print("wrote fig_plasticity_quant")


if __name__ == "__main__":
    fig8_decision_confidence()
    fig9_confidence_plasticity()
    fig_plasticity_quant()
