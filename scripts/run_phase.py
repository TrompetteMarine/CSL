"""Stage 3 -- connectivity phase diagram (Figure 3 data).

Sweeps the anticipatory weight lambda (x) against cross-community permeability
Gamma = 1 - B_cc (y) in the asymmetric scenario (community 2 carries an early
wrong lead).  Each cell stores the probability of each terminal regime.

The grid is computed row-by-row with checkpointing and a per-invocation wall
budget, so several short invocations complete a large sweep.  Re-run until it
prints ALL DONE.  Saves results/phase_<scale>.npz.
"""
import os
import time
import numpy as np
import common
from credibility import scenarios, metrics
from credibility.model import simulate_blocks

BUDGET = float(os.environ.get("ECSL_BUDGET", "1e9"))   # per-invocation wall budget


def main():
    scale = common.get_scale()
    base = scenarios.phase_base(scale)
    lams, gammas, reps = scenarios.phase_grid(scale)
    ng, nl = len(gammas), len(lams)
    path = f"{common.RESULTS}/phase_{scale}.npz"
    if os.path.exists(path):
        d = np.load(path)
        P = d["P"].copy(); done = d["done"].copy()
        cell_done = d["cell_done"].copy() if "cell_done" in d else ~np.isnan(P[:, :, 0])
    else:
        P = np.full((ng, nl, 4), np.nan)              # eff, wrong, pol, unresolved
        done = np.zeros(ng, dtype=bool)
        cell_done = np.zeros((ng, nl), dtype=bool)

    def _save():
        np.savez_compressed(path, P=P, done=done, cell_done=cell_done,
                            lams=lams, gammas=gammas, reps=reps, scale=scale)

    t0 = time.time()
    for gi in range(ng):
        if done[gi]:
            continue
        for li in range(nl):
            if cell_done[gi, li]:
                continue
            p = base.replace(lam=float(lams[li]), b_self=float(1 - gammas[gi]))
            out = simulate_blocks(p, n_reps=reps, seed=700 + gi * 100 + li)
            fr = metrics.regime_frequencies(out["term_m"], out["a_star"])
            P[gi, li] = [fr["efficient"], fr["wrong"], fr["polarised"], fr["unresolved"]]
            cell_done[gi, li] = True
            # cell-level checkpoint: a large grid survives many short invocations
            if time.time() - t0 > BUDGET:
                _save()
                print(f"[budget] paused mid-row {gi+1}/{ng} at cell {li+1}/{nl}; "
                      f"{int(cell_done.sum())}/{ng*nl} cells done. Re-run to continue.")
                return
        done[gi] = True
        _save()
        print(f"phase row {gi+1}/{ng} (Gamma={gammas[gi]:.3f}) done; "
              f"{int(done.sum())}/{ng} rows complete")
    print(f"ALL DONE: phase_{scale}.npz  ({ng}x{nl} grid, {reps} reps/cell)")


if __name__ == "__main__":
    main()
