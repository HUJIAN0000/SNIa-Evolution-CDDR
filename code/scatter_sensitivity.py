"""Separate, explicitly empirical scatter sensitivity; never replaces baseline.

C(s)=C_fixed10+s^2 I with its full determinant. Half-normal s priors are
integrated, not adjusted until chi-square equals one. This does not identify a
physical source of scatter and cannot correct coherent redshift systematics.
"""
import os,sys
sys.dont_write_bytecode=True
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import numpy as np
from scipy.linalg import eigh,cho_factor,cho_solve
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from complete_review_checks import historical,matched,ROOT,ANA,FITS,K,LM0

def fit_scatter(tag,data,scale=.5,n=4001):
    z,y,C=data;X=np.c_[np.ones(len(z)),z];eig,U=eigh(C)
    Xt=X.T@U;yt=y@U;s=np.linspace(0,2,n);W=1/(eig[None,:]+s[:,None]**2)
    Q=np.einsum('ak,nk,bk->nab',Xt,W,Xt);V=np.linalg.inv(Q)
    b=np.einsum('nab,nb->na',V,np.einsum('ak,nk,k->na',Xt,W,yt))
    r=yt-b@Xt;chi=np.sum(r*r*W,axis=1)
    ld=np.sum(np.log(eig[None,:]+s[:,None]**2),axis=1)
    logw=-.5*(chi+ld+np.linalg.slogdet(Q)[1]+(s/scale)**2)
    hx,hw=hermgauss(40);hw/=np.sqrt(np.pi);shift=5*np.log10((LM0+np.sqrt(2)*.25*hx)/LM0)
    gx,gw=leggauss(80);prob=np.zeros(n)
    for start in range(0,n,128):
        ix=slice(start,start+128);sd=np.sqrt(V[ix,0,0]);lo=np.clip((-21+shift-b[ix,0,None])/sd[:,None],-10,10)
        hi=np.clip((-18+shift-b[ix,0,None])/sd[:,None],-10,10);half=(hi-lo)/2
        t=(hi+lo)[:,:,None]/2+half[:,:,None]*gx
        mu=b[ix,1,None,None]+V[ix,0,1,None,None]/sd[:,None,None]*t
        se=np.sqrt(V[ix,1,1]-V[ix,0,1]**2/V[ix,0,0])[:,None,None]
        ep=ndtr((5-mu)/se)-ndtr((-5-mu)/se)
        prob[ix]=(half*np.sum(gw*np.exp(-t*t/2)/np.sqrt(2*np.pi)*ep,axis=2))@hw
    weights=np.exp(logw-logw.max());density=weights*prob;density/=np.trapezoid(density,s)
    cq=cumulative_trapezoid(density,s,initial=0);sq=np.interp([.16,.5,.84],cq,s)
    coarse=density[::2].copy();coarse/=np.trapezoid(coarse,s[::2]);cc=cumulative_trapezoid(coarse,s[::2],initial=0)
    refinement=float(max(abs(sq-np.interp([.16,.5,.84],cc,s[::2]))));assert refinement<1e-4
    cdf=cumulative_trapezoid(weights,s,initial=0);cdf/=cdf[-1];rng=np.random.default_rng(20261027)
    parts=[];count=0
    while count<100000:
        ss=np.interp(rng.random(100000),cdf,s)
        means=np.column_stack([np.interp(ss,s,b[:,i]) for i in range(2)])
        vv=np.stack([np.interp(ss,s,V[:,i,j]) for i in range(2) for j in range(2)],axis=1).reshape(-1,2,2)
        beta=means+np.einsum('nij,nj->ni',np.linalg.cholesky(vv),rng.normal(size=means.shape))
        lm=rng.normal(LM0,.25,len(ss));MB=beta[:,0]-5*np.log10(lm/LM0)
        keep=(MB>-21)&(MB<-18)&(abs(beta[:,1])<5)&(lm>5)&(lm<20)
        parts.append(np.c_[MB,beta[:,1],ss,lm][keep]);count+=sum(keep)
    samples=np.concatenate(parts)[:100000];q=np.percentile(samples,[2.5,16,50,84,97.5],axis=0).T
    # MLE of s, WITHOUT prior or intercept-volume terms.
    mleidx=int(np.argmin(chi+ld));mles=float(s[mleidx])
    result={'tag':tag,'n':len(z),'half_normal_scale_mag':scale,'s_support_mag':[0,2],
      'quantiles_025_16_50_84_975':dict(zip(['M_B0','epsilon','scatter_mag','lm_pc'],q.tolist())),
      'scatter_integral_quantiles':sq.tolist(),'s_grid_refinement_maxchange':refinement,
      'scatter_mle_mag':mles,'chi2_at_mle':float(chi[mleidx]),'nominal_dof':len(z)-3,
      'covariance':'C_fixed10+s^2 I; determinant included',
      'physical_priors':'same bounded MB/epsilon and Gaussian lm as baseline',
      'interpretation':'empirical sensitivity only; not a validated physical error model or a redshift-bias correction'}
    (FITS/f'{tag}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.save(FITS/f'{tag}_posterior.npy',samples)
    np.savez_compressed(FITS/f'{tag}.npz',scatter_grid=s,density=density)
    print('SCATTER',tag,'epsilon',q[1].round(4),'s',sq.round(4),flush=True)
    return result

def main():
    t,data,zs=historical();fit_scatter('historical_scatter05',data)
    for scale in [.25,1.]:fit_scatter(f'historical_scatter{str(scale).replace(".","")}',data,scale)
    fit_scatter('strict_scatter05',matched('dz005'))
    g=np.load(ANA/'gp_matern_prediction.npz');z=t.z.to_numpy()
    mu=5*np.log10(LM0*206.265/t.theta.to_numpy()*(1+z)**2)+25
    gpdata=(z,g['mean']-mu,g['covariance']+np.diag(K*K*((t.dtheta/t.theta)**2+.1**2)))
    fit_scatter('gp_scatter05',gpdata)
    # Per-source residuals and leave-one-out diagnostics, without data deletion.
    X=np.c_[np.ones(len(z)),z];cf=cho_factor(data[2],lower=True)
    V=np.linalg.inv(X.T@cho_solve(cf,X));beta=V@X.T@cho_solve(cf,data[1]);r=data[1]-X@beta
    tab=t[['source','z','theta','dtheta']].copy();tab['residual_mag']=r
    tab['marginal_standardized_residual']=r/np.sqrt(np.diag(data[2]))
    eps=[];chis=[]
    for i in range(len(z)):
        keep=np.arange(len(z))!=i;subcf=cho_factor(data[2][np.ix_(keep,keep)],lower=True)
        xx=X[keep];yy=data[1][keep];b=np.linalg.solve(xx.T@cho_solve(subcf,xx),xx.T@cho_solve(subcf,yy));rr=yy-xx@b
        eps.append(float(b[1]));chis.append(float(rr@cho_solve(subcf,rr)))
    tab['leave_one_out_epsilon_mle']=eps;tab['leave_one_out_chi2']=chis
    tab.to_csv(ANA/'residual_diagnostics.csv',index=False)
    print('RESIDUAL CHECK',tab.sort_values('marginal_standardized_residual').iloc[[0,-1]].to_string(index=False),flush=True)

if __name__=='__main__':main()
