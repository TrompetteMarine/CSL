"""Quantify micro--meso accuracy by grid distance from dominant-regime boundaries."""
import csv
from collections import deque
import numpy as np
import common


def boundary_distances(regime):
    regime = np.asarray(regime)
    dist = np.full(regime.shape, -1, int)
    q = deque()
    for i, j in np.ndindex(regime.shape):
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < regime.shape[0] and 0 <= jj < regime.shape[1] and regime[ii, jj] != regime[i, j]:
                dist[i, j] = 0; q.append((i, j)); break
    while q:
        i, j = q.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < dist.shape[0] and 0 <= jj < dist.shape[1] and dist[ii, jj] < 0:
                dist[ii, jj] = dist[i, j] + 1; q.append((ii, jj))
    return dist


def main():
    scale = common.get_scale()
    q = np.load(f"{common.RESULTS}/quotient_grid_{scale}.npz")
    regime = q["micro_dom"]
    dist = boundary_distances(regime)
    groups = [("boundary / distance <= 1", dist <= 1),
              ("interior distance = 2", dist == 2),
              ("deep interior distance >= 3", dist >= 3),
              ("aggregate", np.ones_like(dist, bool))]
    rows = []
    for name, mask in groups:
        valid = mask & np.isfinite(q["disc"]) & np.isfinite(q["agree"])
        rows.append(dict(group=name, cells=int(valid.sum()),
                         mean_discrepancy=float(np.mean(q["disc"][valid])) if valid.any() else np.nan,
                         median_discrepancy=float(np.median(q["disc"][valid])) if valid.any() else np.nan,
                         regime_agreement=float(np.mean(q["agree"][valid])) if valid.any() else np.nan))
    path = f"{common.FIGDIR}/../tables/micro_meso_boundary_accuracy.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
