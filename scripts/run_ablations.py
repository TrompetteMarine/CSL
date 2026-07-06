"""Stage 2 -- mechanism isolation as a counterfactual experiment (Figure 4).

Identification by common random numbers (CRN). Every variant is simulated with
the *same* seed, so each replication fixes the same exogenous environment -- a
common uniform reward draw and decision-noise draw per agent and trial -- and
each variant realises its chosen arm's reward by thresholding that shared uniform
at the arm's mean. The only structural difference across variants is the switch
that disables one mechanism component (a do-operation on the model). The replication-paired difference
between the full model and an ablation therefore identifies the causal effect of
that component on the outcome, with the shock sequence held fixed. We report
Wilson intervals for regime probabilities, bootstrap intervals for regret and
lag, and paired-bootstrap intervals for the full-vs-ablation contrasts.

Saves results/ablations_<scale>.npz (figure/table arrays) and
results/ablations_<scale>.json (full uncertainty + contrasts).
"""
import numpy as np
import common
from credibility import scenarios, metrics, uncertainty as unc
from credibility.model import simulate_blocks
from credibility.config import SCALE

ORDER = ["full", "no-anticipatory", "no-retrospective",
         "no-confidence-wt", "constant-lr", "alt-confidence"]
CRN_SEED = 20260617          # one seed -> common random numbers across variants


def main():
    scale = common.get_scale()
    reps = min(SCALE[scale]["reps_traj"], 400)
    base = scenarios.ablation_point(scale)
    variants = scenarios.ablations(base)

    per = {}                                  # per-rep outcomes by variant
    for name in ORDER:
        out = simulate_blocks(variants[name], n_reps=reps, seed=CRN_SEED, record_traj=True)
        a = out["a_star"]
        codes = metrics.classify_regime(out["term_m"], a)
        per[name] = dict(
            wrong=(codes == metrics.WRONG).astype(float),
            eff=(codes == metrics.EFFICIENT).astype(float),
            pol=(codes == metrics.POLARISED).astype(float),
            regret=out["regret"],
            lag=metrics.correction_lag(out["traj_m"], a),
            mstar=out["traj_m"][:, :, :, a].mean(axis=0),
        )

    rows, contrasts = {}, {}
    for name in ORDER:
        d = per[name]
        pw = unc.proportion_summary(d["wrong"]); pe = unc.proportion_summary(d["eff"])
        pp = unc.proportion_summary(d["pol"])
        reg = unc.bootstrap_mean(d["regret"], seed=1)
        lag = unc.bootstrap_mean(d["lag"], seed=2)
        rows[name] = dict(R=reps, p_wrong=pw, p_efficient=pe, p_polarised=pp,
                          regret=reg, lag=lag)
        if name != "full":
            dw = unc.paired_bootstrap_diff(per["full"]["wrong"], d["wrong"], seed=3)
            dr = unc.paired_bootstrap_diff(per["full"]["regret"], d["regret"], seed=4)
            flip = float(np.mean(per["full"]["wrong"] != d["wrong"]))
            contrasts[name] = dict(d_p_wrong=dw, d_regret=dr, regime_flip_rate=flip)
        print(f"[{name:16s}] P(wrong)={pw['p']:.2f} [{pw['lo']:.2f},{pw['hi']:.2f}] "
              f"regret={reg['mean']:.0f}±{reg['se']:.0f}")

    # arrays for the figure (point estimates + CIs in ORDER)
    def col(getter):
        return np.array([getter(rows[n]) for n in ORDER], dtype=float)
    store = dict(
        order=np.array(ORDER), R=reps, crn_seed=CRN_SEED,
        p_wrong=col(lambda r: r["p_wrong"]["p"]),
        p_wrong_lo=col(lambda r: r["p_wrong"]["lo"]),
        p_wrong_hi=col(lambda r: r["p_wrong"]["hi"]),
        p_efficient=col(lambda r: r["p_efficient"]["p"]),
        p_efficient_lo=col(lambda r: r["p_efficient"]["lo"]),
        p_efficient_hi=col(lambda r: r["p_efficient"]["hi"]),
        p_polarised=col(lambda r: r["p_polarised"]["p"]),
        regret=col(lambda r: r["regret"]["mean"]),
        regret_lo=col(lambda r: r["regret"]["lo"]),
        regret_hi=col(lambda r: r["regret"]["hi"]),
        correction_lag=col(lambda r: r["lag"]["mean"]),
        scale=scale,
    )
    for name in ORDER:
        store[f"{name}_mstar_mean"] = per[name]["mstar"]
    np.savez_compressed(f"{common.RESULTS}/ablations_{scale}.npz", **store)
    common.dump_json(f"{common.RESULTS}/ablations_{scale}.json",
                     {"operating_point": {k: v for k, v in base.as_dict().items() if v is not None},
                      "crn_seed": CRN_SEED, "R": reps,
                      "variants": rows, "contrasts_vs_full": contrasts})
    print(f"saved ablations_{scale}.npz (CRN seed={CRN_SEED})")


if __name__ == "__main__":
    main()
