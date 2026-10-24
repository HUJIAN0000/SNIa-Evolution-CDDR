"""First-draft likelihood + fixed source-prescribed 10% angular uncertainty.

No change to the 48-pair assignment, redshift convention, mean model or priors.
Use two independent original emcee samplers and extend them for diagnostics.
"""
from pathlib import Path
import os,sys,json,hashlib,importlib.util,time
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
os.environ['MPLBACKEND']='Agg'
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from scipy.optimize import minimize_scalar
from scipy.stats import norm,rankdata

ROOT=Path(__file__).resolve().parents[1]
from prepare_data import ensure_inputs
ensure_inputs(ROOT/'data')
(ROOT/'audit').mkdir(exist_ok=True,parents=True)
OUT=ROOT/'analysis/baseline_mcmc';OUT.mkdir(exist_ok=True,parents=True);PAIR=OUT/'pairing';PAIR.mkdir(exist_ok=True)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
original=module('frozen_first_draft_model',ROOT/'original/mcmc_sn_evolutionv5m.py')
corrected=module('error10_model',ROOT/'code/mcmc_sn_evolutionv5m_error10.py')
matcher=module('frozen_first_draft_matcher',ROOT/'original/CosmoMatcher_v1.1.py')
os.chdir(PAIR)
paths={'tracer':str(ROOT/'data/Pantheon+SH0ES.dat'),'obj':str(ROOT/'data/ucsdata.CSV'),
       'cov_pantheon':str(ROOT/'data/Pantheon+SH0ES_STAT+SYS.cov')}
cols={'tracer_z_col':'zHD','z1_col':'z','z2_col':None,
      'output_cols':['CID','IDSURVEY','zHD','zHEL','m_b_corr','m_b_corr_err_DIAG']}
matcher.CosmoBackend.run_main_logic('cluster','pantheon','milp','dd',paths,cols,{'wm':.3,'tol':.05},print)
df=pd.read_csv(PAIR/'matched_cluster_milp_dd_pantheon.csv')
old=pd.read_csv(ROOT/'original/matched_cluster_milp_dd_pantheon.csv')
C=np.loadtxt(PAIR/'cov_cluster_milp_dd_pantheon_lens.txt')
C_old=np.loadtxt(ROOT/'original/cov_cluster_milp_dd_pantheon_lens.txt')
assert list(zip(df.obj_index,df.sn_matched_idx))==list(zip(old.obj_index,old.sn_matched_idx))
assert np.array_equal(C,C_old)
args=(df.z,df.theta,df.dtheta,df.sn_matched_m_b_corr,C)
error_report=df[['obj_index','z','theta','dtheta']].copy()
error_report['extra_10pct_theta']=.1*df.theta
error_report['sigma_theta_effective']=np.hypot(df.dtheta,.1*df.theta)
error_report['sigma_mu_measured']=(5/np.log(10))*df.dtheta/df.theta
error_report['sigma_mu_extra10pct']=5/np.log(10)*.1
error_report['sigma_mu_effective']=(5/np.log(10))*error_report.sigma_theta_effective/df.theta
error_report.to_csv(OUT/'per_source_error_budget.csv',index=False)
checks={'pairs':len(df),'same_pair_order':True,'covariance_exactly_equal':True,
        'error_formula':'sigma_theta_eff^2 = dtheta^2 + (0.10 theta)^2',
        'mu_extra_sigma_mag':float(5/np.log(10)*.1),'baseline_identity':{}}
for mode in ['evol','log_evol','ddr','joint']:
    a=original.SNEvolutionLikelihood(*args,fit_mode=mode)
    b=corrected.SNEvolutionLikelihood(*args,fit_mode=mode,extra_fraction=0)
    samples=np.load(ROOT/f'original/sn_evolution_v4_chain_{mode}.npy')[::41]
    differences=[abs(a.log_probability(p)-b.log_probability(p)) for p in samples]
    assert max(differences)==0
    checks['baseline_identity'][mode]={'points':len(samples),'max_log_probability_difference':max(differences)}
test=corrected.SNEvolutionLikelihood(*args,extra_fraction=.1)
expected=C+np.diag((5/np.log(10))**2*((df.dtheta/df.theta)**2+.1**2))
checks['covariance_formula_max_difference']=float(abs(test.cov_tot-expected).max())
assert checks['covariance_formula_max_difference']<1e-14
dump(ROOT/'audit/baseline_mcmc_input_checks.json',checks)
if '--check-only' in sys.argv:
    print('Raw-catalog baseline matching and zero-extra-error likelihood identity verified.');sys.exit(0)

def rank_split_rhat(chains,tau):
    # Diagnostic only: walkers within an ensemble interact. Also compare the
    # two independent ensemble summaries and a deterministic integration.
    a=chains[:,::max(1,int(tau/5))]
    a=np.transpose(a,(0,2,1,3)).reshape(-1,a.shape[1],a.shape[3])
    half=a.shape[1]//2;a=np.concatenate([a[:,:half],a[:,-half:]],axis=0)
    def calc(b):
        M,N,D=b.shape
        ranked=np.empty_like(b)
        for d in range(D):ranked[:,:,d]=norm.ppf((rankdata(b[:,:,d].ravel()).reshape(M,N)-.375)/(M*N+.25))
        W=np.var(ranked,axis=1,ddof=1).mean(axis=0)
        B=N*np.var(ranked.mean(axis=1),axis=0,ddof=1)
        return np.sqrt(((N-1)/N*W+B/N)/W)
    return np.maximum(calc(a),calc(abs(a-np.median(a,axis=(0,1))))).tolist()

report={'n':len(df),'additional_fractional_angular_uncertainty':.1,
        'scatter_parameter':False,'redshift_mismatch_correction':False,
        'matching':'exact historical 48-pair greedy assignment; legacy output filename contains milp',
        'priors':'unchanged first-draft bounds and Gaussian lm calibration','modes':{}}
for mode in ['evol','log_evol','ddr','joint']:
    start=time.perf_counter();like=corrected.SNEvolutionLikelihood(*args,fit_mode=mode,extra_fraction=.1)
    # The likelihood is linear in intercept and slope at fixed eta.
    f=np.log1p(df.z.to_numpy()) if mode=='log_evol' else df.z.to_numpy()
    if mode=='ddr':X=np.ones((len(df),1))
    else:X=np.c_[np.ones(len(df)),f]
    CiX=cho_solve(like.cho,X);V=np.linalg.inv(X.T@CiX)
    mu0=5*np.log10(11.03*206.265/df.theta.to_numpy()*(1+df.z.to_numpy())**2)+25
    def profile(eta,details=False):
        yy=df.sn_matched_m_b_corr.to_numpy()-mu0-5*np.log10(1+eta*df.z.to_numpy())
        beta=V@(CiX.T@yy);resid=yy-X@beta
        chi=float(resid@cho_solve(like.cho,resid))
        return (chi,beta) if details else chi
    if mode in ['joint','ddr']:
        lo=-1/max(df.z)+1e-9;hi=2-1e-9
        grid=np.linspace(lo+1e-7,hi,2001);values=np.array([profile(v) for v in grid])
        i=int(values.argmin());left=grid[max(0,i-2)];right=grid[min(len(grid)-1,i+2)]
        opt=minimize_scalar(profile,bounds=(left,right),method='bounded',options={'xatol':1e-12})
        eta=float(opt.x)
    else:eta=0.
    chi,beta=profile(eta,True)
    samplers=[]
    for run,seed in enumerate([42,20261023]):
        if run and mode=='joint':
            _,start_beta=profile(1.5,True)
            like.p0=np.array([start_beta[0],start_beta[1],1.5,11.03])
        sampler,*_=corrected.run_mcmc(like,32,8000,seed=seed,auto_extend=True,max_extensions=15)
        samplers.append(sampler)
    # Bring independent ensembles to the same length before comparing them.
    total=max(s.iteration for s in samplers)
    for s in samplers:
        if s.iteration<total:s.run_mcmc(None,total-s.iteration,progress=False)
    while True:
        taus=[s.get_autocorr_time(quiet=True) for s in samplers]
        tau=float(max(max(v) for v in taus));total=samplers[0].iteration
        burn=max(1000,int(5*tau));raw=np.stack([s.get_chain()[burn:] for s in samplers])
        rh=rank_split_rhat(raw,tau)
        print('INDEPENDENT CHECK',mode,total,tau,rh,flush=True)
        passed=(total-burn)>80*tau and max(rh)<1.01
        if passed or total>=160000:break
        for s in samplers:s.run_mcmc(None,8000,progress=False)
    if not passed:raise RuntimeError(f'{mode}: diagnostic targets not met; retain logs rather than publish a result')
    thin=max(1,int(tau/10));flat=raw[:,::thin].reshape(-1,like.ndim)
    lp=np.concatenate([s.get_log_prob()[burn::thin].reshape(-1) for s in samplers])
    np.save(OUT/f'posterior_{mode}.npy',flat)
    ensemble_q=[np.percentile(r[::thin].reshape(-1,like.ndim),[16,50,84],axis=0).T.tolist() for r in raw]
    np.savez_compressed(OUT/f'raw_chain_{mode}.npz',chain=np.stack([s.get_chain() for s in samplers]),
        log_posterior=np.stack([s.get_log_prob() for s in samplers]),burn=burn,thin=thin,tau=np.array(taus),
        acceptance=np.stack([s.acceptance_fraction for s in samplers]),seeds=[42,20261023])
    q=np.percentile(flat,[16,50,84],axis=0).T
    mle=[beta[0],beta[1],eta,11.03] if mode=='joint' else [beta[0],eta if mode=='ddr' else beta[1],11.03]
    ll=float(like.log_likelihood(mle));n_identifiable=like.ndim-1
    report['modes'][mode]={'labels':like.labels,'quantiles':q.tolist(),'n_samples':len(flat),
        'steps_per_ensemble':total,'seeds':[42,20261023],'burn':burn,'thin':thin,
        'tau':np.array(taus).tolist(),'retained_steps_over_tau':float((total-burn)/tau),
        'rank_split_rhat':rh,'ensemble_quantiles':ensemble_q,'convergence_targets_passed':passed,
        'mean_acceptance':[float(s.acceptance_fraction.mean()) for s in samplers],
        'mle':list(map(float,mle)),'loglike_max':ll,'chi2':chi,
        'nominal_dof':len(df)-n_identifiable,'n_identifiable':n_identifiable,
        'aic':2*n_identifiable-2*ll,'bic':n_identifiable*np.log(len(df))-2*ll,
        'information_criterion_note':'Data-identifiable intercept combination counted once; lm and MB are exactly degenerate in likelihood.',
        'seconds':time.perf_counter()-start}
    dump(OUT/'results.json',report)
    print('FINISHED MODE',mode,json.dumps(report['modes'][mode]),flush=True)
print('ALL FOUR FIRST-DRAFT ERROR-CORRECTED FITS COMPLETE',flush=True)
