# Statistical source tables

All original arrays and configurations are read-only. Run `analyse_uncertainty.py`, `conditional_noise.py`, then `write_statistics.py` from any directory to reconstruct these outputs. NumPy, pandas and SciPy are required.

`endpoint_intervals.csv` has one row per design cell and metric, with batch-based mean/MCSE/95% t interval; conditional-on-realised-graph innovation MCSE is a separate column and estimand. Raw confidence endpoints are not clipped. `batch_outcomes.csv` and `population_outcomes.csv` retain the numerical sampling units. `paired_contrasts.csv` and the corresponding population/batch files preserve matched stream indices and covariance. `trajectory_intervals.csv` contains pointwise bands after ten-trial trailing smoothing within each population, plus full population and batch source curves in separate files.

`conditional_noise_*` files describe exact one-step action noise at fixed states. Their ensemble MCSE is uncertainty of the average conditional summary across states, not conditional noise itself. The ring-routing comparison fixes actual broadcast p,d,v and changes only the routing operator.

`propagation_checks.json` verifies covariance-aware cumulative-regret propagation and direct nonlinear withdrawal sensitivities. The rounding bound is deterministic readout precision, not a statistical interval or a whole-simulator forward-error proof. No fitted-parameter posterior, hypothesis test or p value is supplied.

Primary B=4 and many-community B=2 intervals rely on an independent-seed/approximately-Gaussian-batch working model whose calibration cannot be validated with so few batches. The distribution-free Hoeffding alternatives are correspondingly broad. Expanded B=16 geometry follow-ups were selected after primary inspection and remain exploratory.
