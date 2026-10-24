"""Fixed-covariance inference with the exact original bounded physical priors.

At fixed eta the identifiable intercept and epsilon have a Gaussian likelihood.
Integrate their rectangular physical bounds, including the Gaussian lm prior,
then integrate eta on a refined grid. Conditional Gaussian rejection draws give
posterior samples without an interacting MCMC convergence requirement.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss

K=5/np.log(10);LM0=11.03

class BoundedFit:
    def __init__(self,z,y,C,ev='linear',dd='none',eta_hi=2.):
        self.z=np.asarray(z);self.y=np.asarray(y);self.C=np.asarray(C)
        self.ev=ev;self.dd=dd;self.eta_hi=eta_hi
        self.f=self.z if ev=='linear' else np.log1p(self.z)
        self.g=self.z if dd=='linear' else np.log1p(self.z)
        self.X=np.c_[np.ones(len(z)),self.f] if ev!='none' else np.ones((len(z),1))
        self.cf=cho_factor(C,lower=True);self.CiX=cho_solve(self.cf,self.X)
        self.V=np.linalg.inv(self.X.T@self.CiX)
        self.L=np.linalg.cholesky(self.V)
        self.eta_lo=max(-2.,-1/max(self.g)+1e-9)
    def regression(self,eta):
        eta=np.atleast_1d(eta)
        yy=self.y[None,:]-5*np.log10(1+eta[:,None]*self.g[None,:])
        beta=yy@self.CiX@self.V;res=yy-beta@self.X.T
        chi=np.einsum('ij,ji->i',res,cho_solve(self.cf,res.T))
        return beta,chi
    def integral(self,n=3001,gh=32,gl=64):
        hx,hw=hermgauss(gh);hw/=np.sqrt(np.pi)
        lm=LM0+np.sqrt(2)*.25*hx
        assert np.all((lm>5)&(lm<20))
        shift=5*np.log10(lm/LM0);gx,gw=leggauss(gl)
        grid=np.linspace(self.eta_lo,self.eta_hi,n) if self.dd!='none' else np.array([0.])
        beta,chi=self.regression(grid);p=np.zeros(len(grid));sdA=np.sqrt(self.V[0,0])
        for start in range(0,len(grid),128):
            ix=slice(start,start+128);b=beta[ix]
            if self.ev=='none':
                p[ix]=(ndtr((-18+shift-b[:,0,None])/sdA)-ndtr((-21+shift-b[:,0,None])/sdA))@hw
            else:
                lo=np.clip((-21+shift-b[:,0,None])/sdA,-10,10)
                hi=np.clip((-18+shift-b[:,0,None])/sdA,-10,10)
                half=(hi-lo)/2;t=(hi+lo)[:,:,None]/2+half[:,:,None]*gx
                mu=b[:,1,None,None]+self.V[0,1]/sdA*t
                sig=np.sqrt(self.V[1,1]-self.V[0,1]**2/self.V[0,0])
                ep=ndtr((5-mu)/sig)-ndtr((-5-mu)/sig)
                p[ix]=(half*np.sum(gw*np.exp(-t*t/2)/np.sqrt(2*np.pi)*ep,axis=2))@hw
        weight=np.exp(-.5*(chi-chi.min()))
        if len(grid)>1:
            density=weight*p;density/=np.trapezoid(density,grid)
            cdf=cumulative_trapezoid(density,grid,initial=0)
            q=np.interp([.16,.5,.84],cdf,grid)
        else:density=np.ones(1);q=np.zeros(3)
        return grid,density,q,weight,beta,chi
    def run(self,path,seed=20261026,nsample=100000):
        path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
        coarse=self.integral(1501,32,64);fine=self.integral(3001,48,96)
        difference=float(max(abs(coarse[2]-fine[2])))
        assert difference<1e-4,(path,difference)
        grid,density,etaq,weight,beta,chi=fine
        np.savez_compressed(path.with_suffix('.npz'),eta=grid,density=density,chi2_profile=chi)
        # Propose eta from UNTRUNCATED-intercept profile density. Reject the
        # joint physical bounds afterwards; using the bounded density here
        # would incorrectly apply the truncation probability twice.
        rng=np.random.default_rng(seed);chunks=[];total=0;proposed=0
        if len(grid)>1:
            cdf=cumulative_trapezoid(weight,grid,initial=0);cdf/=cdf[-1]
        while total<nsample:
            n=max(8192,min(100000,2*(nsample-total)))
            eta=np.interp(rng.random(n),cdf,grid) if len(grid)>1 else np.zeros(n)
            # beta(eta) is smooth; evaluate interpolation at the refined grid.
            b=np.column_stack([np.interp(eta,grid,beta[:,i]) for i in range(beta.shape[1])])
            coeff=b+rng.normal(size=b.shape)@self.L.T
            lm=rng.normal(LM0,.25,n);MB=coeff[:,0]-5*np.log10(lm/LM0)
            keep=(MB>-21)&(MB<-18)&(lm>5)&(lm<20)
            eps=coeff[:,1] if self.ev!='none' else np.zeros(n)
            if self.ev!='none':keep&=(eps>-5)&(eps<5)
            chunks.append(np.c_[MB,eps,eta,lm][keep]);total+=sum(keep);proposed+=n
            if proposed>10000000:raise RuntimeError('Inefficient bounded rejection')
        samples=np.concatenate(chunks)[:nsample]
        np.save(path.with_name(path.name+'_posterior.npy'),samples)
        quant=np.percentile(samples,[16,50,84],axis=0).T
        if self.dd!='none':
            mcerror=float(max(abs(quant[2]-etaq))/(etaq[2]-etaq[0]))
            assert mcerror<.025,(path,mcerror)
        else:mcerror=0.
        if self.dd!='none':
            opt=minimize_scalar(lambda e:self.regression([e])[1][0],bounds=(self.eta_lo,self.eta_hi),method='bounded')
            eta_mle=float(opt.x)
            candidates=[(eta_mle,opt.fun),(self.eta_hi,self.regression([self.eta_hi])[1][0])]
            eta_mle,minchi=min(candidates,key=lambda p:p[1])
        else:eta_mle=0.;minchi=float(chi[0])
        mle,_=self.regression([eta_mle]);dof=len(self.z)-self.X.shape[1]-(self.dd!='none')
        result={'tag':path.name,'n':len(self.z),'evolution':self.ev,'cddr':self.dd,
          'quantiles':dict(zip(['M_B0','epsilon','eta0','lm_pc'],quant.tolist())),
          'eta_integral_quantiles':etaq.tolist() if self.dd!='none' else None,
          'chi2':float(minchi),'nominal_dof':int(dof),'chi2_over_dof':float(minchi/dof),
          'profile_eta_mle':eta_mle,'profile_coefficients':mle[0].tolist(),
          'integration_refinement_max_eta_change':difference,'iid_sample_eta_quantile_check_fraction':mcerror,
          'nsamples':nsample,'seed':seed,'acceptance':float(nsample/proposed),
          'priors':{'M_B0':[-21,-18],'epsilon':[-5,5],'eta0':[-2,self.eta_hi],
                    'positivity':'eta(z)>0 at all fitted anchors','lm_pc':'Normal(11.03,.25), 5<lm<20'},
          'method':'bounded Gaussian integration plus independent rejection samples; fixed covariance, no free scatter',
          'physical_precision_validated':False}
        path.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('FIT',path.name,'epsilon',quant[1].round(4),'eta',quant[2].round(4),'chi2/dof',round(minchi/dof,3),flush=True)
        return result
