# Decision-generated credibility

Research code and replication data for **decision-generated credibility in networked social learning**. A sender's decision state generates the strength of its outgoing action signal; community geometry routes that signal into other agents' later decisions. Separate reward and choice channels allow learning to be distinguished from its public expression.

This snapshot contains all executable simulation, audit, statistical, figure-builder, workbook-builder and Lean source files; all figure assets; and 12 of 14 source-data workbooks. Seventeen larger derived-data paths, including two workbooks and typed workbook inputs, remain in the complete author-supplied package while GitHub transfer is pending. Exact filenames, sizes and hashes are recorded in [data availability](replication/DATA_AVAILABILITY.json). The article and supplementary documents, their text sources and historical manuscript-containing archives are excluded.

## Start here

From the root of a cloned CSL repository, first run `cd nhb_replication/2026-10-16`. This snapshot is separate from the historical implementation at the repository root.

Python 3.12 and a C++17 compiler are required. Figure rebuilding also uses Matplotlib; standalone diagrams require pdfLaTeX and TikZ. Lean verification has its own pinned toolchain and dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make check
```

`make check` verifies the exact published Git subset and source integrity and the saved formal-verification records, then runs 18 independent dense-equation comparisons against the compiled simulator. It neither reruns the full population study nor performs a fresh Lean kernel build.

## Restore the replication data

The complete prepared source ZIP contains the seventeen deferred files listed in the availability manifest. Restore those exact paths before rebuilding every figure, workbook and statistical export; `python tools/verify_package.py --require-source-manifest --full-source` verifies that complete source snapshot. The raw population archives below are separate.

The raw population arrays and large design/graph metadata are packaged separately for release as `v2026.10.16` in [TrompetteMarine/CSL](https://github.com/TrompetteMarine/CSL/releases). Until that release is available, use the audited archives already supplied by the authors with `--archives-dir`. This keeps large binary data out of Git history. The checked manifests under `replication/` specify archive sizes, SHA-256 hashes and every extracted member.

```bash
python tools/prepare_data.py --download  # after release v2026.10.16 is published
make full-audit
```

For downloaded archives already on disk:

```bash
python tools/prepare_data.py --archives-dir /path/to/release-assets
```

The preparer verifies checksums and extracts only the exact allowlisted replication files. It does not restore the older archives that also contained manuscripts. See [the reproduction guide](docs/REPRODUCIBILITY.md) for the calculation sequence and optional tasks.

| Release asset | Contents |
|---|---|
| `agent_study_arrays.zip` | 616 saved array files for the agent study and continuations |
| `many_communities_arrays.zip` | 48 saved array files for the 10,000-community extension |
| `replication_metadata.zip` | Frozen designs, seeds, exposure matrices and recorded summaries |
| `source_data_workbooks.zip` | 14 Excel workbooks with their CSV/NPZ provenance and builders |

Release target: `v2026.10.16`. The manifests are frozen replication records; the tag is a software/data version, not a journal publication status.

## Repository map

| Path | Purpose |
|---|---|
| `baseline/nhb_smallworld_20261012/src/` | Original sparse C++ simulator and Python experiment drivers |
| `audit/` | Independent reconstruction, graph, dense-engine and precision checks |
| `statistics/` | Population/batch estimates, paired covariance, Monte Carlo intervals and exact conditional noise |
| `figures/` | Six main and ten supplementary figure assets, source exports, plotting code and four standalone TikZ diagrams |
| `source_data/` | Fourteen workbooks, typed inputs, builders and reconciliation records |
| `formal/project/` | Lean sources, toolchain and exact dependency pins |
| `formal/` | Protected-source hashes, declaration/type/axiom audits and fresh-build driver |
| `replication/` | Archive and source integrity manifests |
| `tools/` | Portable restoration and package checks |

The historical directory name is retained to keep the original engine and analysis paths unchanged. Raw `data/results/` arrays, restored metadata and dependency/build caches are deliberately ignored by Git.

## Evidence and interpretation

The main partition study uses 10,000 agents; its two-community case has 5,000 agents per community. A separate extension uses 10,000 communities and 100,000–500,000 agents. The fresh designs contain 652 configurations and 2,512 population trajectories; shared innovations and equivalent controls mean these totals are not independent observations across conditions.

Intervals are approximate Monte Carlo intervals over seed-batch means. Agents and trials are dependent. Primary cells have four batches; the many-community extension has two, so interval calibration cannot be established from those small counts. Exploratory follow-ups, pointwise trajectory bands, conditional action noise, rounding error and unquantified model uncertainty remain distinct.

The score is specified propensity-derived decisiveness, not independently validated human confidence or learned adviser reliability. Strong constant influence reproduces broadcast impairment; endogenous score variation is not necessary for that particular outcome. There are no human observations in this package.

Saved Lean records cover 93 theorem/lemma declarations and 24 definitions. Complete stochastic population-limit and stationary-law arguments are not all formalised. A checked auxiliary statement does not certify a parent probabilistic claim or the cognitive interpretation. [Formal scope](docs/FORMAL_SCOPE.md) describes the distinction.

## Related work and citation

The originating formulation is Gabriel Bontemps and Abhishek Banerjee, [*Learning Whom to Trust: Decision-Generated Credibility in Social Learning*, arXiv:2608.24851v1](https://arxiv.org/abs/2608.24851v1). This replication version retains sender-generated social weighting and the two-channel architecture while replacing the diffusion/time-based readout with propensity-derived decisiveness. Its numerical results refer to the revised specification; the full stochastic models are not equivalent merely because a conditional choice marginal matches.

Use `CITATION.cff` to cite this software/data version. No journal acceptance, normative optimality result or empirical confidence calibration is implied.

## Licence

The host CSL repository retains its existing licence. Third-party dependencies retain their respective licences. No separate data licence or new licence grant is introduced by this snapshot.
