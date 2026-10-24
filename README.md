# SNIa-Evolution-CDDR: reproducibility package

Code, input snapshots and reference results for the ApJ manuscript AAS76998, *Testing Type Ia Supernova Luminosity Evolution with Compact Radio Sources*. Package version: **2026.10.24**.

The calculations compare Pantheon+ standardized apparent magnitudes with 120 compact radio sources. They retain the historical 48-pair baseline, add a fixed 10% fractional angular-size uncertainty, and test certified assignment, finite-redshift transport, Gaussian-process reconstruction, functional forms, prior bounds and empirical residual scatter.

This repository contains the numerical work. It does not claim journal acceptance, a completed DOI deposit or an independently measured joint evolution/CDDR signal. The fixed-error residual fits remain poor, and the joint posterior depends on the prior boundary. See [scientific limits](docs/SCIENTIFIC_SCOPE.md).

Upload instructions: [GitHub upload guide (Chinese)](docs/UPLOAD_zh.md).

## Install

Use a Python 3.11 environment on CPU. No NVIDIA GPU, LaTeX installation or external PDF renderer is needed. The optional original GUI needs Python's Tk support; the main numerical pipeline does not open a GUI.

```bash
python -m pip install -r requirements-tested.txt
```

`requirements-tested.txt` records the environment used for this release. `requirements.txt` lists the runtime dependencies with minimum versions; other versions may change floating-point optimization or random draws.

## Verify the download

```bash
python verify_package.py
python reproduce.py --check
```

The first command uses the standard library to check file hashes, compressed-input integrity and Python syntax. The second also checks covariance indexing, original likelihood identity, all 42 posterior summaries and eight matching certificates. It reads frozen results and does not refit models.

The large Pantheon+ covariance is supplied as `data/Pantheon+SH0ES_STAT+SYS.cov.gz`. The code restores it automatically, then verifies the uncompressed SHA-256. Gzip changes the storage size, not the matrix values, text encoding or row order. The restored `.cov` file is ignored by Git.

## Recompute results

```bash
python reproduce.py
```

This creates a fresh directory under `_runs/`, reruns historical matching, recomputes the 42 supplemental posterior products, and generates the four baseline triangle plots and three diagnostic figures. It compares the new results with `reference_results/`. The packaged reference files are left unchanged. Logs identify each stage and its result.

Choose a new output directory explicitly if needed:

```bash
python reproduce.py --output /path/to/new/run
```

The directory must be empty or absent. On Windows, activate the scientific Python environment first; `run_reproduction.cmd` calls the same entry point.

The default calculation uses the supplied baseline MCMC samples to regenerate Figures 1, 2, 6 and 7. Repeating those long MCMC calculations is a separate operation:

```bash
python reproduce.py --mode baseline-mcmc
```

This runs two independent fixed-10% ensembles, numerical integration checks and the joint-chain refinement. It can create large chain-history files inside its run directory. It does not silently replace the frozen baseline used in the default figures.

Other modes:

```bash
python reproduce.py --mode figures
python reproduce.py --mode historical
python reproduce.py --mode historical-mcmc
```

`figures` regenerates plots from frozen posteriors. `historical` repeats the original 48-pair matching; `historical-mcmc` also runs the unmodified four-model first-draft sampler using its original error prescription. Historical output names contain `milp`, but that old cluster branch is greedy.

## Contents

| Path | Purpose |
|---|---|
| `code/` | Current inference, matching, GP, scatter and plotting code; portable input preparation and verification |
| `code/vendor/` | Exact author-supplied CosmoMatcher v1.4.3 snapshot and its upstream licensing statement |
| `data/` | The three raw inputs, including lossless compressed covariance; hashes and attribution |
| `baseline/` | Fixed-10% 48-pair MCMC posteriors, summaries, pair map and covariance |
| `original/` | Original effective matcher, likelihood and plotting helper, historical pair map and four saved chains |
| `reference_results/analysis/` | All 42 frozen supplemental posteriors, GP predictions, matching tables, solver reports and diagnostics |
| `reference_results/figures/` | Three reference diagnostic figures |
| `reference_results/baseline_figures/` | Four reference baseline triangle figures |
| `reference_results/validation/` | Numerical checks from the source analysis, retained as historical evidence |
| `docs/` | Data interpretation, result-to-code map, scientific limits and upload instructions |
| `PACKAGE_SHA256.json` | Hashes and sizes of the tracked release files |

No reviewer correspondence, private backup trees, manuscript-author administration, reference-paper PDFs or nested submission archives are part of this repository. The source manuscript generators are unnecessary for running the numerical analysis and are kept in the authors' manuscript workspace.

## Reference interpretation

The controlled 48-pair, fixed-10% linear fit gives epsilon1 about 0.063 mag and chi-square/dof about 3.194. Certified MILP gives 50 pairs. A strict absolute-redshift window of 0.005 gives 27 pairs and a shorter radio-redshift range. The GP is trained on 494 SN records, with complete observational and predictive covariance.

Five separately labeled analyses add an empirical scatter parameter. That extension broadens intervals; it does not identify the physical source of residuals. GP and paired results share SN observations and are not independent measurements to multiply together.

The default full replay reports both elementwise posterior agreement and quantile agreement. Exact sample equality depends on the numerical environment; the verifier also checks matching indices and covariance matrices. Reproducibility establishes the implementation and outputs, not physical validity of the adopted error model.

## Data, credit and rights

Cite Brout et al. (2022, ApJ 938, 110), Scolnic et al. (2022, ApJ 938, 113) for Pantheon+, and Cao et al. (2017, A&A 606, A15; [DOI](https://doi.org/10.1051/0004-6361/201730551)) for the radio sources. Data origins and units are in [DATA.md](docs/DATA.md).

The root `LICENSE` preserves the MIT license already present in the SNIa-Evolution-CDDR repository, including its original copyright notice. Bundled catalogs and the separately developed vendor tool retain their own rights; see [THIRD_PARTY.md](THIRD_PARTY.md). The paper author list is not used to infer software contributors, and no software DOI has been invented.
