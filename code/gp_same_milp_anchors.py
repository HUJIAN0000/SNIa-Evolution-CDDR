"""One controlled GP/MILP comparison using the identical 50 CRS anchors."""
import os,sys
sys.dont_write_bytecode=True
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from matching import ROOT,ANA,inputs,K,LM0
from gp import GP
from bounded_inference import BoundedFit

if __name__=='__main__':
    sn,crs,C=inputs();idx=np.flatnonzero(sn.zHD.to_numpy()>=.3)
    gp=GP(sn.zHD.to_numpy()[idx],sn.m_b_corr.to_numpy()[idx],C[np.ix_(idx,idx)]).fit()
    t=pd.read_csv(ANA/'matches_dd05.csv');z=t.z.to_numpy();mean,V=gp.predict(z)
    mu=5*np.log10(LM0*206.265/t.theta.to_numpy()*(1+z)**2)+25
    total=V+np.diag(K*K*((t.dtheta/t.theta)**2+.1**2))
    np.savez_compressed(ANA/'gp_milp50_prediction.npz',z=z,mean=mean,covariance=V)
    BoundedFit(z,mean-mu,total).run(ANA/'fits/gp_milp50_linear_none')
