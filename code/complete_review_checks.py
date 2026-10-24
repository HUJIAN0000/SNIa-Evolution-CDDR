"""Complete referee checks without replacing the historical 48-pair baseline."""
import os,sys
sys.dont_write_bytecode=True
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse import csr_matrix
from matching import ROOT,ANA,K,LM0,inputs,match,distance,dump,cm
from gp import GP
from bounded_inference import BoundedFit

FITS=ANA/'fits';FITS.mkdir(exist_ok=True)

def historical():
    t=pd.read_csv(ROOT/'baseline/pairing/matched_cluster_milp_dd_pantheon.csv')
    y=t.sn_matched_m_b_corr.to_numpy()-5*np.log10(LM0*206.265/t.theta.to_numpy()*(1+t.z.to_numpy())**2)-25
    C=np.loadtxt(ROOT/'baseline/pairing/cov_cluster_milp_dd_pantheon_lens.txt')
    C+=np.diag(K*K*((t.dtheta/t.theta)**2+.1**2))
    zc=t.z.to_numpy();zs=t.sn_matched_zHD.to_numpy()
    return t,(zc,y,C),zs

def matched(tag,transport=True):
    t=pd.read_csv(ANA/f'matches_{tag}.csv');zc=t.z.to_numpy();zs=t.sn_zHD.to_numpy()
    y=t.sn_m_b_corr.to_numpy()-5*np.log10(LM0*206.265/t.theta.to_numpy()*(1+zc)**2)-25
    if transport:y-=t.transport_mag.to_numpy()
    C=np.load(ANA/f'cov_sn_{tag}.npy')+np.diag(K*K*((t.dtheta/t.theta)**2+.1**2))
    return zs,y,C

def fit(tag,data,ev='linear',dd='none',high=2.):
    result=BoundedFit(*data,ev,dd,high).run(FITS/tag)
    return result

def main():
    t,hist,zs=historical();sn,crs,Cfull=inputs();reports={}
    # Retain the original 48 rows, order, covariance and fixed-10% model.
    for ev in ['none','linear','log']:
        for dd in ['none','linear','log']:
            reports[f'historical_{ev}_{dd}']=fit(f'historical_{ev}_{dd}',hist,ev,dd)
    base=json.loads((ROOT/'baseline/results.json').read_text())['modes']
    verified={}
    for mode,ev,dd,param in [('evol','linear','none','epsilon'),('log_evol','log','none','epsilon'),('ddr','none','linear','eta0'),('joint','linear','linear','eta0')]:
        new=reports[f'historical_{ev}_{dd}'];index=2 if mode=='joint' else 1
        old=np.array(base[mode]['quantiles'][index]);q=np.array(new['quantiles'][param])
        fraction=float(max(abs(q-old))/(q[2]-q[0]));assert fraction<.06
        verified[mode]={'maximum_quantile_change_as_68_width':fraction,'same_chi2_difference':abs(base[mode]['chi2']-new['chi2'])}
    dump(ROOT/'audit/baseline_inference_crosscheck.json',verified)
    for high in [3.,4.]:fit(f'historical_wide{int(high)}_linear_linear',hist,'linear','linear',high)
    for ev,dd in [('linear','log'),('log','linear'),('log','log')]:
        fit(f'historical_wide4_{ev}_{dd}',hist,ev,dd,4.)
    # Corrected finite-redshift treatment as a clearly separate sensitivity.
    T=5*np.log10((1+zs)*distance(zs)/((1+t.z.to_numpy())*distance(t.z.to_numpy())))
    transportdata=(zs,hist[1]-T,hist[2])
    fit('historical_signed_transport',transportdata)
    dump(ANA/'historical_offsets.json',{'n':len(t),'median_abs_dz':float(np.median(abs(zs-t.z))),
       'max_abs_dz':float(max(abs(zs-t.z))),'max_abs_transport_mag':float(max(abs(T))),
       'n_with_abs_dz_below005':int(sum(abs(zs-t.z)<=.005))})
    # Actual two-stage MILP only supplements the historical baseline.
    for tag,tol,metric,om,w0,wa in [
       ('dd05',.05,'dd',.3,-1.,0.),('dz005',.005,'dz',.3,-1.,0.),
       ('dd02',.02,'dd',.3,-1.,0.),('dd03',.03,'dd',.3,-1.,0.),('dd07',.07,'dd',.3,-1.,0.),
       ('om028',.05,'dd',.28,-1.,0.),('om032',.05,'dd',.32,-1.,0.),('cpl',.05,'dd',.3,-.8,.5)]:
        match(sn,crs,Cfull,tag,tol,metric,om,w0,wa)
        fit(f'{tag}_linear_none',matched(tag))
    fit('dd05_untransported',matched('dd05',False))
    fit('dz005_untransported',matched('dz005',False))
    ids,_=pd.factorize(sn.CID,sort=True)
    # Candidate graph reconstructed independently of the MILP implementation.
    dc=distance(crs.z);ds=distance(sn.zHD);i,j=np.where(abs(dc[:,None]-ds)/dc[:,None]<=.05)
    graph=csr_matrix((np.ones(len(i)),(i,ids[j])),shape=(len(crs),max(ids)+1))
    maximum=int(sum(maximum_bipartite_matching(graph,perm_type='column')>=0))
    assert maximum==len(pd.read_csv(ANA/'matches_dd05.csv'))
    dump(ROOT/'audit/independent_cardinality.json',{'maximum_pairs':maximum,'candidate_edges':len(i)})
    # Empirical-Bayes GP at EXACTLY the original 48 radio anchors.
    idx=np.flatnonzero(sn.zHD.to_numpy()>=.3);z=sn.zHD.to_numpy()[idx]
    m=sn.m_b_corr.to_numpy()[idx];C=Cfull[np.ix_(idx,idx)];cid=sn.CID.to_numpy()[idx]
    anchors=t.z.to_numpy();mu=5*np.log10(LM0*206.265/t.theta.to_numpy()*(1+anchors)**2)+25
    assert min(anchors)>min(z) and max(anchors)<max(z)
    gp_report={}
    for kind,degree,tag in [('matern52',1,'gp_matern'),('rbf',1,'gp_rbf'),('matern52',2,'gp_quadratic')]:
        gp=GP(z,m,C,kind,degree).fit();mean,V=gp.predict(anchors)
        assert gp.opt.success
        data=(anchors,mean-mu,V+np.diag(K*K*((t.dtheta/t.theta)**2+.1**2)))
        np.savez_compressed(ANA/f'{tag}_prediction.npz',z=anchors,mean=mean,covariance=V,
            training_rows=idx,training_z=z,training_m=m,hyperparameters=np.exp(gp.p))
        fit(f'{tag}_linear_none',data)
        if tag=='gp_matern':
            for ev in ['none','linear','log']:
                for dd in ['none','linear','log']:
                    if (ev,dd)!=('linear','none'):fit(f'{tag}_{ev}_{dd}',data,ev,dd)
        gp_report[tag]={'training_n':len(idx),'anchor_n':len(anchors),'training_z_range':[float(min(z)),float(max(z))],
          'hyperparameters':np.exp(gp.p).tolist(),'mean_degree':degree,'kernel':kind,'optimizer_success':bool(gp.opt.success)}
        if tag=='gp_matern':
            ids=np.unique(cid);rng=np.random.default_rng(20261023);rng.shuffle(ids);cv=[]
            for fold in range(5):
                test=np.isin(cid,ids[fold::5]);train=~test
                g=GP(z[train],m[train],C[np.ix_(train,train)],kind,degree).fit()
                pred,cvC=g.predict(z[test],C_cross=C[np.ix_(test,train)],C_test=C[np.ix_(test,test)])
                r=m[test]-pred
                cv.append({'fold':fold,'n':int(sum(test)),'rmse_mag':float(np.sqrt(np.mean(r*r))),
                  'chi2_over_N':float(r@cho_solve(cho_factor(cvC,lower=True),r)/len(r)),
                  'marginal95_coverage':float(np.mean(abs(r)<1.96*np.sqrt(np.diag(cvC))))})
            gp_report[tag]['grouped_fivefold']=cv
        dump(ANA/'gp_report.json',gp_report)
    print('ALL REFEREE NUMERICAL CHECKS FINISHED',flush=True)

if __name__=='__main__':main()
