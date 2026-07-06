"""Stage 4 -- quotient (micro vs meso) validation (Figure 6 data).

Three products:
  * two-community trajectory overlays (micro ensemble mean vs meso);
  * a four-community modular example (Stage 4, M > 2);
  * a micro--meso discrepancy heatmap over (lambda, permeability).

Trajectory/many-community products are cached; the discrepancy grid is
computed row-by-row with checkpointing and a per-invocation wall budget so the
script resumes cleanly.  Saves quotient_traj_<scale>.npz and
quotient_grid_<scale>.npz.
"""
import os
import time
import numpy as np
import common
from credibility import scenarios, metrics
from credibility.model import simulate_blocks, simulate_meso
from credibility.config import SCALE

BUDGET = float(os.environ.get("ECSL_BUDGET", "1e9"))   # per-invocation wall budget


def fast_products(scale):
    path = f"{common.RESULTS}/quotient_traj_{scale}.npz"
    if os.path.exists(path):
        return
    reps = min(SCALE[scale]["reps_traj"], 250)
    store = {}
    for name, p in scenarios.quotient_trajectories(scale).items():
        micro = simulate_blocks(p, n_reps=reps, seed=400, record_traj=True)
        a = micro["a_star"]
        store[f"{name}_micro_mean"] = micro["traj_m"][:, :, :, a].mean(axis=0)
        store[f"{name}_micro_lo"] = np.percentile(micro["traj_m"][:, :, :, a], 10, axis=0)
        store[f"{name}_micro_hi"] = np.percentile(micro["traj_m"][:, :, :, a], 90, axis=0)
        store[f"{name}_meso"] = simulate_meso(p)["traj_m"][:, :, a]
    # four-community modular example
    p4 = scenarios.quotient_many(scale)
    micro = simulate_blocks(p4, n_reps=reps, seed=401, record_traj=True)
    a = micro["a_star"]
    store["many_micro_mean"] = micro["traj_m"][:, :, :, a].mean(axis=0)
    store["many_meso"] = simulate_meso(p4)["traj_m"][:, :, a]
    store["many_micro_term"] = micro["term_m"].mean(axis=0)
    store["many_meso_term"] = simulate_meso(p4)["term_m"]
    store["scale"] = scale
    np.savez_compressed(path, **store)
    print(f"saved quotient_traj_{scale}.npz")


def grid(scale):
    base, lams, gammas, reps = scenarios.quotient_grid(scale)
    path = f"{common.RESULTS}/quotient_grid_{scale}.npz"
    ng, nl = len(gammas), len(lams)
    if os.path.exists(path):
        d = np.load(path)
        disc = d["disc"].copy(); agree = d["agree"].copy(); done = d["done"].copy()
    else:
        disc = np.full((ng, nl), np.nan); agree = np.full((ng, nl), np.nan)
        done = np.zeros(ng, dtype=bool)
    t0 = time.time()
    for gi in range(ng):
        if done[gi]:
            continue
        for li in range(nl):
            p = base.replace(lam=float(lams[li]), b_self=float(1 - gammas[gi]))
            micro = simulate_blocks(p, n_reps=reps, seed=500 + gi * 100 + li, record_traj=False)
            a = micro["a_star"]
            meso = simulate_meso(p)
            micro_m = micro["term_m"][:, :, a].mean(axis=0)      # (M,)
            meso_m = meso["term_m"][:, a]                        # (M,)
            disc[gi, li] = float(np.mean(np.abs(micro_m - meso_m)))
            micro_reg = np.bincount(metrics.classify_regime(micro["term_m"], a),
                                    minlength=4).argmax()
            meso_reg = metrics.classify_regime(meso["term_m"][None], a)[0]
            agree[gi, li] = float(micro_reg == meso_reg)
        done[gi] = True
        np.savez_compressed(path, disc=disc, agree=agree, done=done,
                            lams=lams, gammas=gammas)
        print(f"grid row {gi+1}/{ng} done  (mean disc so far={np.nanmean(disc):.3f})")
        if time.time() - t0 > BUDGET and not done.all():
            print(f"[budget] pausing at row {gi+1}/{ng}; re-run to continue.")
            return False
    print(f"quotient grid COMPLETE: mean|micro-meso|={np.nanmean(disc):.3f}, "
          f"regime agreement={np.nanmean(agree):.2f}")
    return True


def main():
    scale = common.get_scale()
    fast_products(scale)
    done = grid(scale)
    print("ALL DONE" if done else "GRID INCOMPLETE")


if __name__ == "__main__":
    main()
