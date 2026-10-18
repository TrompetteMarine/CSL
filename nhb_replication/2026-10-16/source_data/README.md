# Source data for the revised NHB Article

These Excel workbooks contain the numerical values used in each data-bearing figure. A separate workbook accompanies each main data figure (Figures 3–6) and each supplementary data figure (Supplementary Figures 1–8 and 10). Figures 1–2 and Supplementary Figure 9 are conceptual TikZ schematics; their editable sources are supplied in `../figures/` and they do not contain empirical measurements. A fourteenth workbook contains the statistical tables and their paired source records.

Simulation-derived workbooks contain saved population outcomes or their analytic conditional-noise summaries. Supplementary Figure 8 contains only deterministic analytic function grids. These records are not human participant data. Agents and trials within a population are dependent and are never counted as independent replications. Seed batches combine the graph and initialisation seed. Population replications within a batch use separate action and reward innovations, subject to the paired-stream design described in the Supplementary Methods.

Each workbook identifies its plotted panels, metric definitions, cohort filters and numerical source files. Figure series and Monte Carlo intervals are stored at full available precision. The source CSVs and the scripts that create the workbooks are included alongside the workbooks.

The statistical tables retain probabilities and probability differences as fractions in their source files; the Article displays them multiplied by 100 as percentages or percentage points. In Supplementary Figure 10d, the plotted coordinates equal 1,000 times `ring_field_rms_sd` or `broadcast_field_sd`, and the axis states ×10⁻³. Its source data retain the natural score units.

Approximate 95% Monte Carlo intervals use seed-batch means and Student's t critical values with B−1 degrees of freedom. Their calibration is unvalidated for the small number of batches used here. They describe simulation sampling uncertainty conditional on the model and design; they do not quantify uncertainty about human behaviour or the adequacy of the model. Pointwise trajectory intervals are not simultaneous confidence bands. Deterministic analytic grids have no sampling interval. Histogram bins describe within-population community outcomes, not independent community-level estimates.

The final index records workbook paths, figure mappings and source hashes. No new simulations are created by the workbook builders.

The frozen workbooks were originally authored by `build_workbooks.mjs` using `@oai/artifact-tool` from the Codex primary runtime. That historical authoring path uses `--max-old-space-size=6144 --expose-gc` for the complete population-trajectory sheets. It is retained for provenance. `Workbook_Verification.json` records an independent ZIP/XML reconciliation of every exported data value, as well as native filtered tables, frozen headers and Excel error-cell checks.

`build_workbooks.py` provides a portable reproduction path using only Python and openpyxl. From the repository root:

```bash
python -m pip install -r requirements-workbooks.txt
python source_data/build_workbooks.py --verify
```

It reads the same frozen typed `workbook_inputs.json`, streams data with openpyxl's write-only mode, and writes all 14 workbooks under `source_data/rebuilt/`. For a smaller check, use `--book S8 --book F6 --verify`. Use `--output-dir PATH` to choose another destination. Existing workbooks are never overwritten; the frozen files under `outputs/nhb_20261016/` remain intact. The portable exporter retains every typed source value, including literal strings that could otherwise be interpreted as Excel formulas, case-distinct population/community-size column names, native filtered tables and frozen header rows. Formatting and ZIP bytes can differ from the original files. `--verify` independently compares every exported data cell against its typed source using ZIP/XML readers and writes `Portable_Workbook_Build_Report.json` in the chosen output directory. The CSV source records also remain directly usable without an Excel authoring dependency.
