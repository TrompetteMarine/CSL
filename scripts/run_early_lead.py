"""Endogenous early-lead robustness (Appendix: tests Proposition 3 without an imposed lead).

Neutral initialisation -> a short burn-in lets a stochastic early asymmetry
emerge -> we measure the confidence-weighted mass gap at the burn-in -> and test
whether (i) wrong consensus appears endogenously once lambda exceeds the local
amplification threshold lambda*, and (ii) the terminal consensus locks onto the
early-led arm above lambda*. Writes a figure, a CSV and results npz.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common
from credibility import metrics
from credibility.model import simulate_blocks
from credibility.config import Params, SCALE
from credibility import ddm, confidence as confmod

BURN = 25


def lambda_star(p):
    """sigma^2 / (beta a C0 b_bar), with C0 the indifference-point confidence."""
    tau0fpt = ddm.mean_fpt(np.array(0.0), p.a_thr, p.sigma)
    C0 = float(confmod.confidence_decision(np.array(0.0), tau0fpt, p.a_thr,
                                            p.kappa1, p.kappa2, p.tau0))
    b_bar = 1.0
    return p.sigma ** 2 / (p.beta * p.a_thr * C0 * b_bar), C0


def main():
    scale = common.get_scale()
    n = SCALE[scale]["N_per"]
    reps = min(SCALE[scale]["reps_traj"], 200)
    # neutral init, small reward gap, symmetric communities: no imposed lead.
    base = Params(sizes=(n, n), T=260, mu=(0.53, 0.47), eta=0.12, b_self=0.70,
                  q_init=((0.5, 0.5), (0.5, 0.5)))
    lstar, C0 = lambda_star(base)
    sizes = np.asarray(base.sizes); N = sizes.sum()
    lams = np.round(np.linspace(0.0, 1.3, 9), 3)

    rows = []
    for li, lam in enumerate(lams):
        out = simulate_blocks(base.replace(lam=float(lam)), reps, seed=900 + li, record_traj=True)
        a = out["a_star"]; a_other = 1 - a
        tm = out["traj_m"]                                   # (R,T+1,M,2)
        popm = (sizes[None, None, :, None] * tm).sum(axis=2) / N   # (R,T+1,2)
        early_arm = np.where(popm[:, BURN, 0] >= popm[:, BURN, 1], 0, 1)
        term_arm = np.where(popm[:, -1, 0] >= popm[:, -1, 1], 0, 1)
        lock = (term_arm == early_arm)
        gap_burn = np.abs(popm[:, BURN, 0] - popm[:, BURN, 1])
        codes = metrics.classify_regime(out["term_m"], a)
        wrong = codes == metrics.WRONG; eff = codes == metrics.EFFICIENT
        toward_wrong = early_arm == a_other
        pw_tw = float(wrong[toward_wrong].mean()) if toward_wrong.any() else np.nan
        pw_tc = float(wrong[~toward_wrong].mean()) if (~toward_wrong).any() else np.nan
        rows.append(dict(lam=float(lam), p_wrong=float(wrong.mean()),
                         p_eff=float(eff.mean()), p_lock=float(lock.mean()),
                         p_early_wrong=float(toward_wrong.mean()),
                         mean_gap_burn=float(gap_burn.mean()),
                         pwrong_early_wrong=pw_tw, pwrong_early_correct=pw_tc))
    print(f"lambda* = {lstar:.2f} (C0={C0:.2f})")
    for r in rows:
        print(f"  lam={r['lam']:.2f} Pwrong={r['p_wrong']:.2f} Peff={r['p_eff']:.2f} "
              f"lock={r['p_lock']:.2f}")

    # save CSV + npz
    with open(f"{common.FIGDIR}/../tables/endogenous_early_lead.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    np.savez_compressed(f"{common.RESULTS}/early_lead_{scale}.npz",
                        lams=lams, lstar=lstar,
                        p_wrong=np.array([r["p_wrong"] for r in rows]),
                        p_eff=np.array([r["p_eff"] for r in rows]),
                        p_lock=np.array([r["p_lock"] for r in rows]),
                        pwrong_early_wrong=np.array([r["pwrong_early_wrong"] for r in rows]),
                        pwrong_early_correct=np.array([r["pwrong_early_correct"] for r in rows]))

    # figure
    import plotstyle
    plotstyle.use()
    def wilson(p, nrep, z=1.96):
        p = np.asarray(p, float); denom = 1 + z * z / nrep
        centre = (p + z * z / (2 * nrep)) / denom
        half = z / denom * np.sqrt(p * (1 - p) / nrep + z * z / (4 * nrep * nrep))
        return np.clip(p - (centre - half), 0, None), np.clip((centre + half) - p, 0, None)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    L = np.array([r["lam"] for r in rows])
    pw = np.array([r["p_wrong"] for r in rows]); pe = np.array([r["p_eff"] for r in rows])
    pew = np.array([r["p_early_wrong"] for r in rows])
    for y, col, lab in [(pw, "#b2182b", "P(wrong)"), (pe, "#2a7a36", "P(efficient)")]:
        elo, ehi = wilson(y, reps)
        a1.errorbar(L, y, yerr=[elo, ehi], fmt="-o", color=col, ms=3,
                    capsize=2, elinewidth=0.9, label=lab)
    a1.axvline(lstar, color="k", ls="--", lw=1.0); a1.text(lstar + 0.02, 0.05, r"$\lambda^\star$", fontsize=8)
    a1.set_xlabel(r"anticipatory weight $\lambda$"); a1.set_ylabel("terminal probability")
    a1.set_ylim(-0.02, 1.02); a1.set_title("(a) Endogenous wrong consensus"); a1.legend(frameon=False, fontsize=7.5)
    elo, ehi = wilson(pew, reps)
    a2.errorbar(L, pew, yerr=[elo, ehi], fmt="-s", color="#d2691e", ms=3, capsize=2,
                elinewidth=0.9, label=r"P(early lead $\to$ suboptimal)")
    a2.plot(L, [r["p_lock"] for r in rows], ":", color="#6a51a3", lw=1.4,
            label="lock-in fidelity")
    a2.axvline(lstar, color="k", ls="--", lw=1.0); a2.text(lstar + 0.02, 0.05, r"$\lambda^\star$", fontsize=8)
    a2.set_xlabel(r"anticipatory weight $\lambda$"); a2.set_ylim(-0.02, 1.02)
    a2.set_title("(b) Amplification manufactures and locks early leads")
    a2.legend(frameon=False, fontsize=6.5)
    fig.suptitle(rf"Endogenous early lead under neutral initialisation "
                 rf"(Wilson 95% CIs, $R={reps}$)", y=1.03, fontsize=9.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{common.FIGDIR}/endogenous_early_lead.{ext}", dpi=300, bbox_inches="tight")
    print("saved endogenous_early_lead figure/csv/npz")


if __name__ == "__main__":
    main()
