"""Stage 2/5 -- confidence mechanism (Figure 5 data).

Records endogenous confidence conditioned on (correct vs wrong choice) and
(early vs late phase) in the wrong-consensus scenario, where early confident
errors drive the amplification.  Saves results/confidence_<scale>.npz.
"""
import numpy as np
import common
from credibility import scenarios
from credibility.model import simulate_blocks

CAP = 60000          # max samples stored per (correctness x phase) cell


def _subsample(x, rng, cap=CAP):
    if len(x) > cap:
        idx = rng.choice(len(x), cap, replace=False)
        return x[idx]
    return x


def main():
    scale = common.get_scale()
    p = scenarios.confidence_scenario(scale).replace(sizes=(120, 120))
    out = simulate_blocks(p, n_reps=60, seed=303, record_conf=True)
    C = out["conf_C"]; correct = out["conf_correct"].astype(bool); phase = out["conf_phase"]
    rng = np.random.default_rng(0)
    groups = {
        "correct_early": C[correct & (phase == 0)],
        "wrong_early":   C[~correct & (phase == 0)],
        "correct_late":  C[correct & (phase == 1)],
        "wrong_late":    C[~correct & (phase == 1)],
    }
    store = {k: _subsample(v, rng) for k, v in groups.items()}
    summary = {k: dict(n=int(v.size), mean=float(v.mean()) if v.size else float("nan"),
                       median=float(np.median(v)) if v.size else float("nan"))
               for k, v in groups.items()}
    store["scale"] = scale
    np.savez_compressed(f"{common.RESULTS}/confidence_{scale}.npz", **store)
    common.dump_json(f"{common.RESULTS}/confidence_{scale}.json", summary)
    for k, s in summary.items():
        print(f"[{k:14s}] n={s['n']:7d} mean C={s['mean']:.3f}")
    print(f"saved confidence_{scale}.npz")


if __name__ == "__main__":
    main()
