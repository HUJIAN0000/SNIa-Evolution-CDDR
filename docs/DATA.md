# Frozen input data

All row indices are zero-based Python indices into the supplied files. Keep catalog order and covariance order together.

| File | Role | Interpretation |
|---|---|---|
| `Pantheon+SH0ES.dat` | 1701 SN light-curve records | Use `zHD`, corrected apparent magnitude `m_b_corr`, and physical identifier `CID`. Records are not all distinct physical SNe |
| `Pantheon+SH0ES_STAT+SYS.cov.gz` | Complete 1701 by 1701 covariance | Lossless storage of the original text matrix. Already includes statistical and systematic covariance; do not add the statistical diagonal again |
| `ucsdata.CSV` | 120 compact radio sources | `z`: radio redshift; `theta`: angular size in mas; `dtheta`: tabulated angular-size uncertainty in mas |

The data files came from the author-supplied raw snapshots used in the analysis. Exact compressed and uncompressed hashes are in `data/INPUTS.json`. The covariance is expanded automatically when needed; an existing uncompressed file with a different hash causes an error rather than being overwritten.

Pantheon+ upstream: [official data release](https://github.com/PantheonPlusSH0ES/DataRelease). Cite Brout et al. 2022, ApJ 938, 110, and Scolnic et al. 2022, ApJ 938, 113.

Radio source reference: Cao et al. 2017, A&A 606, A15, [doi:10.1051/0004-6361/201730551](https://doi.org/10.1051/0004-6361/201730551). The adopted extra 10% fractional angular uncertainty is added in quadrature to the tabulated `dtheta`; it is not a replacement for that column.

The distance conversion is `D_A[Mpc] = 206.265*l_m[pc]/theta[mas]`. The shared ruler calibration is a sampled parameter with `l_m ~ N(11.03,0.25^2)` pc. It is distinct from each source's independent angular-error term.

Tiny roundoff asymmetry in the full covariance is handled by the supplied backend's recorded normalization tolerance. It does not change the frozen input file or add an uncertainty component.

JLA experiments and unrelated strong-lens data are not inputs to this manuscript and are not required by any release entry point. This package includes the complete inputs actually used by the released pipeline.
