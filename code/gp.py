"""Full-covariance empirical-Bayes GP comparator on the same 50 CRS anchors.

The GP learns corrected apparent magnitudes from SNe (not distances from CRS).
It uses no background expansion curve. Hyperparameters are estimated with REML;
their uncertainty is tested by kernel/mean choices, not claimed marginalized.
"""
from pathlib import Path
import sys, json, time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from matching import ROOT, ANA, LM0, K, inputs, dump

def kernel(x,y,logpars,kind):
    amp,length=np.exp(logpars)
    r=abs(x[:,None]-y[None,:])/length
    if kind=='matern52':
        q=np.sqrt(5)*r
        return amp**2*(1+q+q*q/3)*np.exp(-q)
    return amp**2*np.exp(-.5*r*r)

class GP:
    def __init__(self,z,y,C,kind='matern52',degree=1):
        self.x=np.log(np.asarray(z));self.y=np.asarray(y);self.C=np.asarray(C)
        self.kind=kind;self.degree=degree
        self.H=np.vander(self.x,N=degree+1,increasing=True)
    def state(self,p):
        total=kernel(self.x,self.x,p,self.kind)+self.C
        cf=cho_factor(total,lower=True)
        KiH=cho_solve(cf,self.H);A=self.H.T@KiH
        beta=np.linalg.solve(A,self.H.T@cho_solve(cf,self.y))
        r=self.y-self.H@beta;alpha=cho_solve(cf,r)
        nll=.5*(r@alpha+2*np.log(np.diag(cf[0])).sum()+np.linalg.slogdet(A)[1]
                  +(len(r)-self.H.shape[1])*np.log(2*np.pi))
        return float(nll),cf,KiH,A,beta,alpha
    def fit(self):
        opts=[]
        for p in [[.1,.4],[.3,1.],[1.,2.]]:
            opts.append(minimize(lambda t:self.state(t)[0],np.log(p),method='L-BFGS-B',
                                bounds=[(np.log(.001),np.log(5.)),(np.log(.03),np.log(5.))]))
        self.opt=min(opts,key=lambda o:o.fun);self.p=self.opt.x
        return self
    def predict(self,z,C_cross=None,C_test=None):
        x=np.log(np.asarray(z));H=np.vander(x,N=self.degree+1,increasing=True)
        _,cf,KiH,A,beta,alpha=self.state(self.p)
        kt=kernel(x,self.x,self.p,self.kind)
        prior=kernel(x,x,self.p,self.kind)
        if C_cross is not None:kt=kt+C_cross
        if C_test is not None:prior=prior+C_test
        R=H-kt@KiH
        mean=H@beta+kt@alpha
        cov=prior-kt@cho_solve(cf,kt.T)+R@np.linalg.solve(A,R.T)
        return mean,(cov+cov.T)/2

