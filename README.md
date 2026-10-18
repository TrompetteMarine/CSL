# CSL — Cognitive Social Learning

Code for decision-generated credibility in networked social learning.

The latest audited replication snapshot is [nhb_replication/2026-10-16](nhb_replication/2026-10-16/). It contains the revised sparse simulator, frozen designs, statistical and numerical audits, figure builders, 12 of 14 source-data workbooks and pinned Lean sources. The new snapshot excludes the article and supplementary text.

```bash
git clone https://github.com/TrompetteMarine/CSL.git
cd CSL/nhb_replication/2026-10-16
python -m pip install -r requirements.txt
make check
```

See the [snapshot README](nhb_replication/2026-10-16/README.md), [reproduction guide](nhb_replication/2026-10-16/docs/REPRODUCIBILITY.md) and [formal verification scope](nhb_replication/2026-10-16/docs/FORMAL_SCOPE.md). Seventeen larger derived-data paths are retained in the complete author package while transfer is pending; see the snapshot data-availability manifest. Raw simulation archives are supplied separately; the planned data-release tag is `v2026.10.16`. Use locally supplied archives until release assets are published.

The existing implementation and history remain available at their original paths. Its instructions are retained in [LEGACY_README.md](LEGACY_README.md). Those paths describe the historical drift–diffusion formulation; the revised snapshot uses propensity-derived decisiveness and should be reproduced with its own commands and specification. Results and formal claims should not be transferred between the two specifications without checking their assumptions.
