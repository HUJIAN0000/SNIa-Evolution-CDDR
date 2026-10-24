"""Independent quadrature checks, retaining the original bounded priors."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.special import ndtr
from scipy.integrate import cumulative_trapezoid
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis/baseline_mcmc'
t=pd.read_csv(OUT/'pairing/matched_cluster_milp_dd_pantheon.csv')
z=t.z.to_numpy()
C=np.loadtxt(OUT/'pairing/cov_cluster_milp_dd_pantheon_lens.txt')
C+=np.diag((5/np.log(10))**2*((t.dtheta/t.theta)**2+.1**2))
cf=cho_factor(C,lower=True)
y=t.sn_matched_m_b_corr.to_numpy()-5*np.log10(11.03*206.265/t.theta.to_numpy()*(1+z)**2)-25
def calc(mode,n=3001,lm_order=32,A_order=64):
    hx,hw=hermgauss(lm_order);hw/=np.sqrt(np.pi)
    lm=11.03+np.sqrt(2)*.25*hx;shift=5*np.log10(lm/11.03)
    f=np.log1p(z) if mode=='log_evol' else z
    X=np.ones((len(z),1)) if mode=='ddr' else np.c_[np.ones(len(z)),f]
    CiX=cho_solve(cf,X);V=np.linalg.inv(X.T@CiX)
    if mode in ['evol','log_evol']:
        beta=V@(CiX.T@y)
        std=np.sqrt(V[1,1]);grid=np.linspace(max(-5,beta[1]-9*std),min(5,beta[1]+9*std),n)
        muA=beta[0]+V[0,1]/V[1,1]*(grid-beta[1])
        sdA=np.sqrt(V[0,0]-V[0,1]**2/V[1,1])
        p=(ndtr((-18+shift[None,:]-muA[:,None])/sdA)-ndtr((-21+shift[None,:]-muA[:,None])/sdA))@hw
        density=np.exp(-.5*((grid-beta[1])/std)**2)*p
    else:
        grid=np.linspace(-1/max(z)+1e-9,2,n)
        nx,nw=leggauss(A_order);density=np.zeros(n);chi=np.zeros(n)
        sdA=np.sqrt(V[0,0])
        for start in range(0,n,128):
            ix=slice(start,start+128)
            yy=y[None,:]-5*np.log10(1+grid[ix,None]*z[None,:])
            beta=yy@CiX@V;residual=yy-beta@X.T
            chi[ix]=np.einsum('ij,ji->i',residual,cho_solve(cf,residual.T))
            if mode=='ddr':
                p=(ndtr((-18+shift[None,:]-beta[:,0,None])/sdA)-ndtr((-21+shift[None,:]-beta[:,0,None])/sdA))@hw
            else:
                lo=np.clip((-21+shift[None,:]-beta[:,0,None])/sdA,-10,10)
                hi=np.clip((-18+shift[None,:]-beta[:,0,None])/sdA,-10,10)
                half=(hi-lo)/2;tt=(hi+lo)[:,:,None]/2+half[:,:,None]*nx[None,None,:]
                mu_eps=beta[:,1,None,None]+V[0,1]/sdA*tt
                sd_eps=np.sqrt(V[1,1]-V[0,1]**2/V[0,0])
                pp=ndtr((5-mu_eps)/sd_eps)-ndtr((-5-mu_eps)/sd_eps)
                p=(half*np.sum(nw[None,None,:]*np.exp(-tt**2/2)/np.sqrt(2*np.pi)*pp,axis=2))@hw
            density[ix]=p
        density*=np.exp(-.5*(chi-chi.min()))
    density/=np.trapezoid(density,grid)
    cdf=cumulative_trapezoid(density,grid,initial=0)
    q=np.interp([.16,.5,.84],cdf,grid)
    return q,grid,density

report={'method':'Bounded Gaussian intercept/evolution integration; Gaussian lm calibration integrated; eta grid where needed. No free scatter.', 'modes':{}}
results=json.loads((OUT/'results.json').read_text())
for mode in ['evol','log_evol','ddr','joint']:
    q,_,_=calc(mode)
    fine,grid,density=calc(mode,6001,48,96)
    refine=float(max(abs(q-fine)))
    assert refine<1e-4
    index=2 if mode=='joint' else 1
    mcmc=np.array(results['modes'][mode]['quantiles'][index])
    fraction=float(max(abs(mcmc-fine))/(fine[2]-fine[0]))
    assert fraction<.05,(mode,fraction)
    report['modes'][mode]={'parameter':'eta0' if mode in ['ddr','joint'] else 'epsilon',
        'quantiles':fine.tolist(),'mcmc_quantiles':mcmc.tolist(),
        'grid_refinement_max_change':refine,'max_mcmc_difference_as_fraction_of_68_width':fraction}
    np.savez_compressed(OUT/f'quadrature_{mode}.npz',parameter=grid,density=density)
(ROOT/'audit/baseline_quadrature_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
