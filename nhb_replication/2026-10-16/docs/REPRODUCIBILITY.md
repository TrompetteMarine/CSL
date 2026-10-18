# Reproduction guide

## Levels of reproduction

1. **Code checks without the large arrays:** install `requirements.txt`, then `make check`. The actual simulator is compiled and compared against an independent dense implementation under supplied action/reward innovations, including unequal community sizes and separate channel matrices. Saved Lean records are checked against their source hashes.
2. **Audit the archived study:** restore the three raw-array/metadata release archives, then `make full-audit`. These calculations use saved populations; they do not rerun the original experiment grid.
3. **Rebuild visual and spreadsheet outputs:** use the commands below. No population simulation is required once the saved arrays and metadata are restored.
4. **Fresh simulation or Lean kernel execution:** these are separate tasks with their own costs and dependencies.

## Frozen inputs

```bash
python tools/prepare_data.py --download
```

Run these commands from `nhb_replication/2026-10-16/` within CSL. The raw-data release may be pending; use the already supplied archives offline until the tagged assets are published. The default release target is `TrompetteMarine/CSL`, tag `v2026.10.16`. Supply `--archives-dir` for offline use or explicit repository/tag options for a mirror. Archive and payload hashes are verified before content is installed. Existing conflicting files are rejected rather than silently overwritten.

Use `python tools/prepare_data.py --help` for metadata-only restoration and the optional workbook archive. The source workbooks already present in the repository are identical to the frozen verified exports.

## Numerical and statistical sequence

From the repository root:

```bash
python audit/recompute.py
python audit/check_graphs.py
python audit/check_engine.py
python audit/check_precision.py
python statistics/analyse_uncertainty.py
python statistics/conditional_noise.py
python statistics/write_statistics.py
python statistics/verify_statistics.py
```

The staging version of `write_statistics.py` creates exact CSV selections for statistical tables. Manuscript and supplementary prose generation has been removed.

`check_precision.py` replays one historical configuration to resolve a stored float32 tie. This is a precision check, not a new scientific design cell. Historical raw files remain unchanged. Statistical scripts preserve pairing, nested seed batches and temporal covariance. Exact conditional action-noise calculations use the saved complete states and do not simulate another routing policy.

## Figures

```bash
python figures/rebuild_figures.py
cd figures
pdflatex -interaction=nonstopmode -halt-on-error main01_concept.tex
pdflatex -interaction=nonstopmode -halt-on-error main06_population_limits.tex
pdflatex -interaction=nonstopmode -halt-on-error supp09_signal_anatomy.tex
pdflatex -interaction=nonstopmode -halt-on-error supp10_noise_propagation.tex
python -c 'import rebuild_figures as r; r.legacy_aliases()'
python export_source_data.py
python final_figure_audit.py
```

Figure numbers and file stems are mapped in `figures/Source_Data_Manifest.json`. Numerical source exports retain full available precision. PDF byte hashes can change after rebuilding because of timestamps or TeX versions; equality of displayed estimates and mathematical content is the relevant reproduction target.

## Workbooks

The frozen 14 workbooks have been independently reconciled to their source values. A typed-input builder preserves units, row filters, batch/replication identifiers and the special scopes of deterministic grids and conditional-noise calculations.

For a portable value-equivalent rebuild:

```bash
python -m pip install -r requirements-workbooks.txt
python source_data/prepare_workbook_inputs.py
python source_data/build_workbooks.py
```

Portable rebuilds are written under `source_data/rebuilt/` by default and do not overwrite the frozen workbooks. Use `--help` for individual-book selection. The original JavaScript authoring source is retained for environments with `@oai/artifact-tool`; the portable Python path does not require that runtime. Byte-identical styling across spreadsheet engines is not claimed.

## Fresh simulations

`baseline/nhb_smallworld_20261012/src/study.py` and the companion follow-up, choice-only, withdrawal and many-community drivers contain the original design generation and execution code. Read their command-line options before running them. The archived fresh-start study involved 36.672 billion agent-trial decisions, with further entry replays/continuations. A complete rerun is not part of `make check` or `make full-audit`.

## Integrity and generated files

`replication/source_manifest.json` records the checked source/derived-output snapshot. Release manifests separately record data payloads. Regenerated audit summaries, PDFs and workbooks may have new byte hashes; keep original records when comparing a fresh calculation. `make check` distinguishes saved-record integrity from actual Lean kernel execution, which is documented separately.

## Published source subset and complete prepared snapshot

All executable source files and final figure assets are published in Git. The exact published files are checked by `replication/git_source_manifest.json`; the full prepared snapshot is checked by `replication/source_manifest.json`. These manifests exclude each other to avoid a hash cycle. `replication/DATA_AVAILABILITY.json` lists 17 derived files whose GitHub upload is pending, including the two larger workbooks and typed workbook inputs. Their exact bytes are retained in the complete author-supplied source ZIP. Copy them into their listed paths, then run `python tools/verify_package.py --require-source-manifest --full-source`.

`make check` works on the published Git subset without those files. A full statistics/figure/workbook rebuild additionally needs the deferred source files and the separate raw-array/metadata archives. No completed GitHub data release is implied.
