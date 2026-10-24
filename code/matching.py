"""Reproducible ApJ revision: raw-data matching and covariance-aware inference.

Supplemental matching only; fixed-error inference is in bounded_inference.py. No photometry is used to choose matches. The supplied CosmoMatcher 1.4.3
solver is used without modification. CPL candidate construction is an explicit
extension. Inference is implemented separately; no photometry enters pairing.
"""
from pathlib import Path
import sys, os, json, importlib.util, hashlib, argparse, time
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import quad
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm, rankdata


ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data'
from prepare_data import ensure_inputs
ANA=ROOT/'analysis'
LM0=11.03
K=5/np.log(10)
def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
cm=module('revision_cosmomatcher',Path(__file__).parent/'vendor/CosmoMatcherv1.4.3.py')

def distance(z,om=.3,w0=-1.,wa=0.):
    def inv_e(x):
        de=(1+x)**(3*(1+w0+wa))*np.exp(-3*wa*x/(1+x))
        return 1/np.sqrt(om*(1+x)**3+(1-om)*de)
    return np.array([quad(inv_e,0,float(v),epsabs=1e-11)[0] for v in np.atleast_1d(z)])

def inputs():
    ensure_inputs(RAW)
    (ROOT/'audit').mkdir(exist_ok=True,parents=True)
    sn=pd.read_csv(RAW/'Pantheon+SH0ES.dat',sep=r'\s+',dtype={'CID':str})
    crs=pd.read_csv(RAW/'ucsdata.CSV')
    cov=cm.load_matrix(RAW/'Pantheon+SH0ES_STAT+SYS.cov',len(sn))
    if cov.shape!=(len(sn),len(sn)):raise ValueError('Catalogue/covariance dimension mismatch')
    cov,normalization=cm.normalize_covariance(cov,'Pantheon+ STAT+SYS')
    dump(ROOT/'audit/covariance_normalization.json',normalization)
    cho_factor(cov)
    if not np.isfinite(crs[['z','theta','dtheta']]).all().all():raise ValueError('Invalid CRS input')
    if np.any(crs[['z','theta','dtheta']].to_numpy()<=0):raise ValueError('Nonpositive CRS input')
    return sn,crs,cov

def match(sn,crs,cov,tag,tol=.05,metric='dd',om=.3,w0=-1.,wa=0.,unique='CID'):
    groups=sn.CID.to_numpy() if unique=='CID' else np.arange(len(sn))
    if (w0,wa)==(-1.,0.):
        p=cm.build_problem(crs.z.to_numpy(),sn.zHD.to_numpy(),tol=tol,tol_type=metric,omega_m=om,groups=groups)
    else:
        ds=distance(sn.zHD,om,w0,wa);dc=distance(crs.z,om,w0,wa)
        err=abs(dc[:,None]-ds[None,:])/dc[:,None]
        i,j=np.where(err<=tol)
        g,labels=pd.factorize(groups,sort=True)
        cols=np.tile(np.arange(len(i)),2)
        rows=np.r_[i,len(crs)+g[j]]
        A=sparse.csc_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(crs)+len(labels),len(i)))
        p=cm._Problem(A,np.full(A.shape[0],-np.inf),np.ones(A.shape[0]),err[i,j]**2,np.ones(len(i)),
                      np.c_[i,j,np.full(len(i),-1)],'bipartite',len(crs),len(i))
    active=np.asarray(p.A.getnnz(axis=1)).ravel()>0
    p.A=p.A[active].tocsc();p.lower=p.lower[active];p.upper=p.upper[active]
    chosen,report=cm.solve_problem(p,log_func=lambda s:None,tie_break='catalogue_order')
    chosen=chosen[np.argsort(chosen[:,0])]
    i,j=chosen[:,0],chosen[:,1]
    assert len(set(i))==len(i) and len(set(groups[j]))==len(j)
    t=crs.iloc[i].reset_index(drop=True).copy();t.insert(0,'obj_index',crs.index.to_numpy()[i])
    t['sn_row']=j
    for col in ['CID','zHD','zHEL','m_b_corr','zHDERR']:t['sn_'+col]=sn.iloc[j][col].to_numpy()
    ds=distance(t.sn_zHD,om,w0,wa);dc=distance(t.z,om,w0,wa)
    t['delta_z']=t.sn_zHD-t.z
    t['relative_dc_error']=abs(ds-dc)/dc
    t['transport_mag']=5*np.log10((1+t.sn_zHD)*ds/((1+t.z)*dc))
    t.to_csv(ANA/f'matches_{tag}.csv',index=False)
    np.save(ANA/f'cov_sn_{tag}.npy',cov[np.ix_(j,j)])
    report.update(tag=tag,n=len(t),unique_by=unique,criterion=metric,tolerance=tol,om=om,w0=w0,wa=wa,
       candidates=len(p.costs),z_range=[float(t.z.min()),float(t.z.max())],
       sn_z_range=[float(t.sn_zHD.min()),float(t.sn_zHD.max())],
       median_abs_dz=float(np.median(abs(t.delta_z))),max_abs_dz=float(max(abs(t.delta_z))),
       max_relative_dc=float(max(t.relative_dc_error)),max_abs_transport=float(max(abs(t.transport_mag))))
    dump(ANA/f'matching_{tag}.json',report)
    print('MATCH',tag,len(t),report['max_abs_dz'],flush=True)
    return t

def dataset(tag,transport=True,extra_fraction=0):
    t=pd.read_csv(ANA/f'matches_{tag}.csv')
    mu=5*np.log10(LM0*206.265/t.theta*(1+t.z)**2)+25
    y=t.sn_m_b_corr.to_numpy()-mu.to_numpy()
    if transport:y-=t.transport_mag.to_numpy()
    C=np.load(ANA/f'cov_sn_{tag}.npy')+np.diag((K*t.dtheta/t.theta)**2+(K*extra_fraction)**2)
    return t.sn_zHD.to_numpy(),y,C

