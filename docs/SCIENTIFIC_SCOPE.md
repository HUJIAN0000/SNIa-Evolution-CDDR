# Scientific scope and limits

The fixed additional angular error is `sigma_theta_eff^2 = dtheta^2 + (0.10*theta)^2`. The baseline retains the original 48 pairs, mean models and physical prior bounds. A fixed-10% linear fit still has chi-square/dof about 3.2. Its formal intervals do not establish a complete physical uncertainty budget.

The observed curve depends on the shared intercept `M_B0 + 5*log10(l_m/11.03)` and on compensating luminosity-evolution and CDDR terms. Their joint marginals depend on the eta prior boundary. Nine models form a 3 by 3 restriction matrix; only four have both physical channels free.

The GP retains complete conditional predictive covariance and mean-coefficient uncertainty, but REML estimates the hyperparameters rather than marginalizing their entire posterior. Kernel and mean variants are sensitivity checks. GP and paired fits reuse the same SN catalog.

An empirical model adds `s^2 I` to the fixed covariance and retains the covariance determinant, with a half-normal scatter prior. It quantifies sensitivity to residual dispersion; it does not establish whether the dispersion comes from radio sources, SN modeling, selection, approximation or correlated systematics. A coherent redshift bias is not cured by independent scatter.

Under the empirical baseline scatter model the epsilon1 central 95% interval is approximately [-0.464,0.468] mag; the strict 27-pair scatter result has approximately [-0.697,1.063] mag. Do not turn these into a universal exclusion of absolute epsilon1 above 0.45 mag, or interpret consistency with zero as proof of no evolution.

Radio anchors begin at z=0.462. A completed lower-redshift step changes the extrapolated high-redshift intercept and is outside the direct radio-ruler test. External calibration assumptions must be stated when used; no artificial narrow absolute-magnitude prior is introduced here.

Algorithmic optimality, numerical replay, uncertainty-model validity and journal acceptance are separate claims. This package checks the first two within its declared scope.
