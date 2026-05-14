# -*- coding: utf-8 -*-
"""
Created on Thu May 14 22:48:34 2026

@author: Administrator

Created on Thu May 14 19:08:31 2026

@author: Administrator

Updated Master Script for Pantheon+ & CRS Cross-Probe Cosmography
Optimized for CPU (Float64 Precision) with Explicit MLE Log-Determinant
"""
import numpy as np
import pandas as pd
import emcee
import os
import sys
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt

try:
    import cosmo_tools
    HAS_COSMO_TOOLS = True
except ImportError:
    HAS_COSMO_TOOLS = False
    print("[WARN] cosmo_tools not found. Triangle plot will be skipped.")

# ============================================================
# 1. 全局基础配置
# ============================================================
CONFIG = {
    'data_csv':   'matched_cluster_milp_dd_pantheon.csv',
    'cov_sn':     'cov_cluster_milp_dd_pantheon_lens.txt',
    'lm_prior_mean':  11.03,
    'lm_prior_sigma': 0.25,
    'nwalkers':       32,
    'nsteps':         8000,        
    'auto_extend':    True,        
    'max_extensions': 2,           
    'seed':           42,
    'output_pdf_base':     'sn_evolution_v4',
    'diagnostic_png_base': 'sn_evolution_v4_diagnostics',
    'save_chain_base':     'sn_evolution_v4_chain', 
}

# ============================================================
# 2. 似然函数类 (包含严谨的 ln|C| 计算)
# ============================================================
class SNEvolutionLikelihood:
    def __init__(self, z, theta_obs, sigma_theta, mb, cov_sn,
                 fit_mode='evol', lm_prior_mean=11.03, lm_prior_sigma=0.25):
        
        # 默认使用高精度 float64
        self.z           = np.asarray(z, dtype=np.float64)
        self.theta_obs   = np.asarray(theta_obs, dtype=np.float64)
        self.sigma_theta = np.asarray(sigma_theta, dtype=np.float64)
        self.mb          = np.asarray(mb, dtype=np.float64)
        self.n           = len(self.z)

        self.fit_mode = fit_mode
        self.lm_mu    = lm_prior_mean
        self.lm_sigma = lm_prior_sigma

        # 构建总协方差矩阵 C_tot = C_SN + C_CRS
        sigma_mu_crs = (5.0 / np.log(10.0)) * (self.sigma_theta / self.theta_obs)
        cov_crs = np.diag(sigma_mu_crs**2)
        self.cov_tot = cov_sn + cov_crs

        # Cholesky 分解用于快速求逆
        try:
            self.cho = cho_factor(self.cov_tot, lower=True)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Covariance matrix is not positive definite: {e}")

        # ---------------------------------------------------------
        # 【核心修正】显式化 MLE 的 Log-Determinant 项 (对应论文 Eq.12)
        # ---------------------------------------------------------
        sign, logdet = np.linalg.slogdet(self.cov_tot)
        if sign <= 0:
            raise RuntimeError("Covariance matrix has non-positive determinant.")
        # logdet_term = ln|C| + N*ln(2π)
        self.logdet_term = logdet + self.n * np.log(2.0 * np.pi)

        # 模式判定与参数设置
        if fit_mode == 'evol':
            self.ndim   = 3
            self.labels = [r"M_B^0", r"\epsilon", r"l_m\ ({\rm pc})"]
            self.p0     = np.array([-19.10, 0.0, lm_prior_mean])
        elif fit_mode == 'log_evol': # 对应论文中 Section 4.3 的对数演化测试
            self.ndim   = 3
            self.labels = [r"M_B^0", r"\epsilon_2", r"l_m\ ({\rm pc})"]
            self.p0     = np.array([-19.10, 0.0, lm_prior_mean])
        elif fit_mode == 'ddr':
            self.ndim   = 3
            self.labels = [r"M_B^0", r"\eta_0", r"l_m\ ({\rm pc})"]
            self.p0     = np.array([-19.10, 0.0, lm_prior_mean])
        else:  # joint
            self.ndim   = 4
            self.labels = [r"M_B^0", r"\epsilon", r"\eta_0", r"l_m\ ({\rm pc})"]
            self.p0     = np.array([-19.10, 0.0, 0.0, lm_prior_mean])

    def _unpack(self, theta):
        if self.fit_mode == 'evol':
            MB0, epsilon, lm = theta
            epsilon2 = 0.0; eta0 = 0.0
        elif self.fit_mode == 'log_evol':
            MB0, epsilon2, lm = theta
            epsilon = 0.0; eta0 = 0.0
        elif self.fit_mode == 'ddr':
            MB0, eta0, lm = theta
            epsilon = 0.0; epsilon2 = 0.0
        else:
            MB0, epsilon, eta0, lm = theta
            epsilon2 = 0.0
        return MB0, epsilon, epsilon2, eta0, lm

    def model_mu_predicted(self, theta):
        _, _, _, eta0, lm = self._unpack(theta)
        D_A = (lm * 206.2650) / self.theta_obs        
        D_L_naive = D_A * (1.0 + self.z)**2           
        eta_z = 1.0 + eta0 * self.z                   
        if np.any(eta_z <= 0):
            return None
        D_L = D_L_naive * eta_z
        return 5.0 * np.log10(D_L) + 25.0

    def log_prior(self, theta):
        MB0, epsilon, epsilon2, eta0, lm = self._unpack(theta)
        # 宽平先验
        if not (-21.0 < MB0 < -18.0):     return -np.inf
        if not (-5.0  < epsilon < 5.0):   return -np.inf
        if not (-5.0  < epsilon2 < 5.0):  return -np.inf
        if not (-2.0  < eta0    < 2.0):   return -np.inf
        if not ( 5.0  < lm      < 20.0):  return -np.inf
        
        # CRS l_m 的高斯先验
        lp = -0.5 * ((lm - self.lm_mu) / self.lm_sigma)**2
        return lp

    def log_likelihood(self, theta):
        MB0, epsilon, epsilon2, eta0, lm = self._unpack(theta)
        mu_pred = self.model_mu_predicted(theta)
        if mu_pred is None:
            return -np.inf
        
        # 统一计算 M_B(z) 的演化项 (自动适配 evol 或 log_evol)
        MB_z = MB0 + epsilon * self.z + epsilon2 * np.log(1.0 + self.z)
        
        # 残差矩阵 Δμ
        r = self.mb - (mu_pred + MB_z)
        
        # 快速计算 chi^2 = r^T * C^{-1} * r
        alpha = cho_solve(self.cho, r)
        chi2 = float(np.dot(r, alpha))
        
       
        return -0.5 * chi2 - 0.5 * self.logdet_term

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

# ============================================================
# 3. 采样与后处理函数
# ============================================================
def run_mcmc(like, nwalkers, nsteps, seed=42, auto_extend=True, max_extensions=2):
    np.random.seed(seed)
    ndim = like.ndim
    scale = np.array([1e-3] * (ndim - 1) + [1e-2], dtype=np.float64)
    pos = like.p0 + scale * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, like.log_probability)
    sampler.run_mcmc(pos, nsteps, progress=True)

    extensions = 0
    while True:
        try:
            tau = sampler.get_autocorr_time(quiet=True)
            total_steps = sampler.iteration
            print(f"\n[Convergence] Autocorrelation time tau = {tau}")
            print(f"[Convergence] Chain length / max(tau) = {total_steps / np.max(tau):.1f}  (recommend >= 50)")
            if np.max(tau) * 50 <= total_steps:
                print("[Convergence] OK: chain is long enough.")
                break
            if not auto_extend or extensions >= max_extensions:
                print("[Convergence] WARNING: chain may be too short. Consider increasing nsteps.")
                break
            extensions += 1
            print(f"[Convergence] Extending chain by {nsteps} more steps (extension {extensions}/{max_extensions}) ...")
            sampler.run_mcmc(None, nsteps, progress=True)
        except emcee.autocorr.AutocorrError as e:
            print(f"[Convergence] tau estimate unreliable: {e}")
            if not auto_extend or extensions >= max_extensions:
                tau = np.full(ndim, sampler.iteration / 50.0)
                break
            extensions += 1
            print(f"[Convergence] Extending chain ({extensions}/{max_extensions}) ...")
            sampler.run_mcmc(None, nsteps, progress=True)

    tau_max = float(np.max(tau))
    burn   = int(2 * tau_max)
    thin   = max(1, int(tau_max / 2))
    print(f"[Chain] Using burn-in = {burn}, thin = {thin}")

    flat_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    flat_logprob = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
    return sampler, flat_samples, flat_logprob, (burn, thin, tau)

def make_diagnostics(like, flat_samples, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(like.z, bins=20, color='#00529B', alpha=0.7)
    axes[0].set_xlabel('z'); axes[0].set_ylabel('count')
    axes[0].set_title(f'Matched sample redshift (N={like.n})')

    lm0 = like.lm_mu
    D_A = (lm0 * 206.265) / like.theta_obs#206265 to 206.265
    D_L_naive = D_A * (1.0 + like.z)**2
    mu_crs_naive = 5.0 * np.log10(D_L_naive) + 25.0
    resid = like.mb - mu_crs_naive
    sigma_mu = (5.0 / np.log(10)) * (like.sigma_theta / like.theta_obs)
    axes[1].errorbar(like.z, resid, yerr=sigma_mu, fmt='o', color='#00529B', alpha=0.7, ms=4)
    axes[1].set_xlabel('z')
    axes[1].set_ylabel(r'$m_b - \mu_{\rm CRS}^{\rm naive}\ (l_m=%.2f)$' % lm0)
    axes[1].set_title('Observed residual vs z')
    axes[1].axhline(np.median(resid), color='k', ls='--', alpha=0.5, label=f'median = {np.median(resid):.2f}')
    axes[1].legend()

    theta_med = np.median(flat_samples, axis=0)
    mu_pred = like.model_mu_predicted(theta_med)
    MB0, epsilon, epsilon2, eta0, lm = like._unpack(theta_med)
    MB_z = MB0 + epsilon * like.z + epsilon2 * np.log(1.0 + like.z)
    final_resid = like.mb - (mu_pred + MB_z)
    axes[2].errorbar(like.z, final_resid, yerr=sigma_mu, fmt='o', color='#B22222', alpha=0.7, ms=4)
    axes[2].axhline(0, color='k', ls='--', alpha=0.5)
    axes[2].set_xlabel('z'); axes[2].set_ylabel('post-fit residual (mag)')
    axes[2].set_title(f'Residual at posterior median\nrms = {np.std(final_resid):.3f} mag')

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[Diagnostics] Saved: {out_png}")

# ============================================================
# 4. 主控函数
# ============================================================
def main(cfg):
    print("Loading CosmoMatcher matched data and covariance...")
    df = pd.read_csv(cfg['data_csv'])
    z           = df['z'].values.astype(np.float64)
    theta_obs   = df['theta'].values.astype(np.float64)
    sigma_theta = df['dtheta'].values.astype(np.float64)
    mb          = df['sn_matched_m_b_corr'].values.astype(np.float64)
    cov_sn      = np.loadtxt(cfg['cov_sn'], dtype=np.float64) 

    print(f"  N = {len(z)} matched pairs")
    print(f"  z range: [{z.min():.3f}, {z.max():.3f}]")

    # 循环加入了对数测试模式 'log_evol'，与论文 Test 3 严格对应
    fit_modes = ['evol', 'log_evol', 'ddr', 'joint']

    for mode in fit_modes:
        print("\n" + "="*80)
        print(f"🚀 STARTING FIT MODE: {mode.upper()}")
        print("="*80)

        pdf_file   = f"{cfg['output_pdf_base']}_{mode}.pdf"
        png_file   = f"{cfg['diagnostic_png_base']}_{mode}.png"
        chain_file = f"{cfg['save_chain_base']}_{mode}.npy" if cfg.get('save_chain_base') else None

        like = SNEvolutionLikelihood(
            z, theta_obs, sigma_theta, mb, cov_sn,
            fit_mode      = mode,
            lm_prior_mean = cfg['lm_prior_mean'],
            lm_prior_sigma= cfg['lm_prior_sigma'],
        )

        sampler, flat_samples, flat_logprob, (burn, thin, tau) = run_mcmc(
            like,
            nwalkers       = cfg['nwalkers'],
            nsteps         = cfg['nsteps'],
            seed           = cfg['seed'],
            auto_extend    = cfg['auto_extend'],
            max_extensions = cfg['max_extensions'],
        )

        print(f"\n[Chain] Flat samples shape = {flat_samples.shape}")

        if chain_file:
            np.save(chain_file, flat_samples)
            print(f"[Chain] Saved samples to {chain_file}")

        print(f"\n[Summary - {mode.upper()}] Posterior percentiles (16, 50, 84):")
        for i, lab in enumerate(like.labels):
            q = np.percentile(flat_samples[:, i], [16, 50, 84])
            print(f"  {lab:>18s} = {q[1]:+.4f}  "
                  f"+{q[2]-q[1]:.4f} / -{q[1]-q[0]:.4f}")

        make_diagnostics(like, flat_samples, png_file)

        if HAS_COSMO_TOOLS:
            print(f"\n[Post] Generating triangle plot: {pdf_file} ...")
            stats = cosmo_tools.calculate_stats(flat_samples, like.labels)
            best_lp = float(np.max(flat_logprob))
            cosmo_tools.print_results(stats, lnL_best=best_lp, num_data=like.n)
            cosmo_tools.plot_getdist_advanced(
                samples       = flat_samples,
                labels        = like.labels,
                stats_list    = stats,
                filename      = pdf_file,
                contour_color = "#00529B",
            )
        else:
            print("\n[Post] Skipping cosmo_tools triangle plot (module missing).")

if __name__ == "__main__":
    main(CONFIG)