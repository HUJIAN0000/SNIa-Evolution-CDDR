# Results and execution map

## Numerical stages

| Stage | Script | Products in a fresh run |
|---|---|---|
| Historical raw-data matching | `reproduce_historical.py` | `analysis/historical_mcmc/`; 48 pairs, old covariance order |
| Certified assignment, forms and GP | `complete_review_checks.py` | Eight matching runs, 48-anchor GP variants, grouped five-fold validation, bounded fits |
| Same-anchor GP comparison | `gp_same_milp_anchors.py` | GP at the same 50 radio anchors as MILP |
| Empirical residual model | `scatter_sensitivity.py` | Five scatter fits, per-source and leave-one-out diagnostics |
| Four baseline triangle figures | `plot_baseline.py` | `figures/sn_evolution_error10_{evol,joint,log_evol,ddr}.pdf` and PNGs |
| Three new diagnostic figures | `make_review_figures.py` | `review_prior`, `review_gp`, `review_comparison`, PDF and PNG |
| Numerical verification | `verify_results.py` | `audit/result_verification.json`; input order, original likelihood identity, posteriors and certificates |

`matching.py`, `bounded_inference.py` and `gp.py` are libraries called by these stages. `code/vendor/CosmoMatcherv1.4.3.py` is the exact supplied backend. Its geometric allocation is not a cosmology-independent distance measurement and does not guarantee best parameter precision.

## Keep the reference layers distinct

`original/` preserves the historical effective analysis and its saved chains. Its four-model script writes `sn_evolution_v4_*` despite having `v5m` in its source filename. It does not include the additional 10% term.

`baseline/` contains the controlled 48-pair, fixed-10% MCMC result used for the main triangle plots. The long recalculation writes `analysis/baseline_mcmc/`, with optional joint refinement; it never updates `baseline/` automatically.

`reference_results/analysis/` contains 42 supplementary fits using bounded numerical integration and independent rejection draws, not the same Monte Carlo arrays as the frozen baseline. Small baseline-versus-supplemental quantile differences are sampling/integration comparisons, not changed inputs.

## Array columns

- Fixed-covariance `*_posterior.npy`: 100000 rows, columns `[M_B0, epsilon, eta0, l_m_pc]`. Inactive coefficients are zero. Read each JSON's `evolution` and `cddr` fields.
- Empirical-scatter `*_posterior.npy`: 100000 rows, columns `[M_B0, epsilon, scatter_mag, l_m_pc]`. The third column is **not eta0**. Read `quantiles_025_16_50_84_975` in the JSON.
- `baseline/posterior_*.npy` and `original/sn_evolution_v4_chain_*.npy`: model-specific dimensions and labels; use their associated summaries or original class labels.
- Matching CSVs retain radio `obj_index` and original `sn_row`. The historical map uses `sn_matched_idx` instead.

## Important reference values

| Construction, exact CDDR | Linear epsilon median [mag] | Interpretation |
|---|---:|---|
| Historical 48 with fixed 10%, frozen baseline | 0.063 | Conditional result; chi-square/dof about 3.194 |
| Same 48 with signed transport | 0.051 | Finite-redshift sensitivity |
| MILP 50 with signed transport | -0.014 | Certified assignment under declared constraints |
| Strict redshift window, 27 pairs | 0.413 | 68% interval does not contain zero; 95% does |
| GP at historical 48 anchors | 0.047 | Shares SN observations with paired fits |
| GP at MILP 50 anchors | 0.034 | Same-anchor method comparison |
| Historical 48 with empirical scatter | 0.009 | Error-model sensitivity under its scatter prior |

The strict sample has radio zmax=1.417, while the main sample reaches 2.198. Changing its slope is not by itself a diagnosis of sample variance.

The release verifier also reports descriptive fixed-covariance joint-minus-linear delta AIC=-0.8106 and delta BIC=1.0606 using the stated identifiable-parameter counts. These are not calibrated physical model-selection evidence under an inadequate residual model.
