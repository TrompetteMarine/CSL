"""Stage 1 -- baseline two-community trajectories (Figure 2 data).

Runs the efficient, wrong and polarised scenarios, records ensemble
trajectories of optimal-arm mass, a representative single run for each, regime
frequencies and the polarisation path.  Saves to results/baseline_<scale>.npz.
"""
import numpy as np
import common
from credibility import scenarios, metrics
from credibility.model import simulate_blocks
from credibility.config import SCALE

SEEDS = {"efficient": 101, "wrong": 102, "polarised": 103}
TARGET = {"efficient": metrics.EFFICIENT, "wrong": metrics.WRONG,
          "polarised": metrics.POLARISED}


def main():
    scale = common.get_scale()
    reps = SCALE[scale]["reps_traj"]
    store, summary = {}, {}
    for name, builder in scenarios.BASELINE.items():
        p = builder(scale)
        out = simulate_blocks(p, n_reps=reps, seed=SEEDS[name], record_traj=True)
        a = out["a_star"]
        mstar = out["traj_m"][:, :, :, a]                       # (R, T+1, M)
        codes = metrics.classify_regime(out["term_m"], a)
        # representative run: first replication landing in the target regime.
        hit = np.where(codes == TARGET[name])[0]
        rep = int(hit[0]) if len(hit) else int(np.argmax(codes == np.bincount(codes).argmax()))
        store[f"{name}_mstar_mean"] = mstar.mean(axis=0)
        store[f"{name}_mstar_lo"] = np.percentile(mstar, 10, axis=0)
        store[f"{name}_mstar_hi"] = np.percentile(mstar, 90, axis=0)
        store[f"{name}_mstar_rep"] = mstar[rep]
        store[f"{name}_pol"] = metrics.polarisation_path(out["traj_m"]).mean(axis=0)
        freqs = metrics.regime_frequencies(out["term_m"], a)
        ttc = metrics.time_to_consensus(out["traj_m"])
        ttc_reached = ttc[ttc >= 0]
        summary[name] = dict(
            a_star=int(a), R=int(reps), regime_freqs=freqs,
            mean_regret=float(out["regret"].mean()),
            regret_se=float(out["regret"].std(ddof=1) / np.sqrt(reps)),
            time_to_consensus=(float(ttc_reached.mean()) if ttc_reached.size else None),
            params={k: v for k, v in p.as_dict().items() if v is not None},
        )
        print(f"[{name:10s}] {freqs}  regret={summary[name]['mean_regret']:.0f}")
    store["scale"] = scale
    np.savez_compressed(f"{common.RESULTS}/baseline_{scale}.npz", **store)
    common.dump_json(f"{common.RESULTS}/baseline_{scale}.json", summary)
    print(f"saved baseline_{scale}.npz")


if __name__ == "__main__":
    main()
