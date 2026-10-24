"""First-draft plotting recipe, with declared prior support and fixed errors."""
from pathlib import Path
import os,sys,json,subprocess,hashlib
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
os.environ['MPLBACKEND']='Agg'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor,solve_triangular
import cosmo_tools
import mcmc_sn_evolutionv5m_error10 as model

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures';ANA=ROOT/'baseline'
results=json.loads((ANA/'results.json').read_text())
t=pd.read_csv(ANA/'pairing/matched_cluster_milp_dd_pantheon.csv')
C=np.loadtxt(ANA/'pairing/cov_cluster_milp_dd_pantheon_lens.txt')
for mode,r in results['modes'].items():
    samples=np.load(ANA/f'posterior_{mode}.npy')
    assert np.array_equal(np.percentile(samples,[16,50,84],axis=0).T,np.array(r['quantiles']))
    labels=r['labels'];ranges={'p0':[-21.,-18.],f'p{len(labels)-1}':[5.,20.]}
    if mode=='joint':ranges.update({'p1':[-5.,5.],'p2':[-1/max(t.z),2.]})
    elif mode=='ddr':ranges['p1']=[-1/max(t.z),2.]
    else:ranges['p1']=[-5.,5.]
    plt.rcdefaults()
    filename=OUT/f'sn_evolution_error10_{mode}.pdf'
    cosmo_tools.plot_getdist_advanced(samples=samples,labels=labels,
        stats_list=cosmo_tools.calculate_stats(samples,labels),filename=str(filename),
        contour_color='#00529B',ranges=ranges)
    plt.close('all')
    import pymupdf
    with pymupdf.open(filename) as pdf:
        pdf[0].get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(filename.with_suffix('.png'))
    like=model.SNEvolutionLikelihood(t.z,t.theta,t.dtheta,t.sn_matched_m_b_corr,C,fit_mode=mode,extra_fraction=.1)
    theta=np.array(r['mle']);MB,e1,e2,eta,lm=like._unpack(theta)
    pred=like.model_mu_predicted(theta)+MB+e1*like.z+e2*np.log1p(like.z)
    residual=like.mb-pred;whitened=solve_triangular(like.cho[0],residual,lower=True)
    fig,ax=plt.subplots(1,2,figsize=(9,3.4))
    ax[0].errorbar(like.z,residual,yerr=np.sqrt(np.diag(like.cov_tot)),fmt='o',ms=3,color='#00529B')
    ax[0].axhline(0,color='k',ls='--',lw=1)
    ax[0].set(xlabel='CRS redshift',ylabel='Residual at likelihood maximum (mag)')
    ax[1].plot(np.arange(len(t)),whitened,'o',ms=3,color='#00529B')
    ax[1].axhline(0,color='k',ls='--',lw=1);ax[1].axhspan(-2,2,color='tan',alpha=.25)
    ax[1].set(xlabel='Original pair order',ylabel='Cholesky-whitened residual')
    fig.suptitle(f'{mode}: fixed extra 10%; '+r'$\chi^2/\nu$'+f' = {r["chi2"]:.2f}/{r["nominal_dof"]}')
    fig.tight_layout();fig.savefig(OUT/f'diagnostics_{mode}.png',dpi=150);plt.close(fig)
print('All four original-style figures and full-covariance residual diagnostics created.')
