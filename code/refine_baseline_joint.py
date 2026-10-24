"""Continue original-likelihood joint chains; verify tail quantiles by quadrature."""
from pathlib import Path
import os,sys,json,time
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
os.environ['MPLBACKEND']='Agg';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import emcee
from scipy.stats import norm,rankdata
import mcmc_sn_evolutionv5m_error10 as model
ROOT=Path(__file__).resolve().parents[1];ANA=ROOT/'analysis/baseline_mcmc'
report=json.loads((ANA/'results.json').read_text())
reference=np.array(json.loads((ROOT/'audit/baseline_quadrature_checks.json').read_text())['modes']['joint']['quantiles'])
r=report['modes']['joint']
q=np.array(r['quantiles'])[2]
if max(abs(q-reference))/(reference[2]-reference[0])<.02:
    print('Joint quantiles already pass the integration comparison.');sys.exit(0)
(ROOT/'audit/results_before_joint_refinement.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
old=np.load(ANA/'raw_chain_joint.npz')
chains=old['chain'];logs=old['log_posterior'];acceptance=old['acceptance'];old.close()
t=pd.read_csv(ANA/'pairing/matched_cluster_milp_dd_pantheon.csv')
C=np.loadtxt(ANA/'pairing/cov_cluster_milp_dd_pantheon_lens.txt')
like=model.SNEvolutionLikelihood(t.z,t.theta,t.dtheta,t.sn_matched_m_b_corr,C,fit_mode='joint',extra_fraction=.1)
def rhat(raw,tau):
    a=raw[:,::max(1,int(tau/5))]
    a=np.transpose(a,(0,2,1,3)).reshape(-1,a.shape[1],a.shape[3]);h=a.shape[1]//2
    a=np.concatenate([a[:,:h],a[:,-h:]],axis=0)
    def calc(b):
        M,N,D=b.shape;z=np.empty_like(b)
        for d in range(D):z[:,:,d]=norm.ppf((rankdata(b[:,:,d].ravel()).reshape(M,N)-.375)/(M*N+.25))
        W=np.var(z,axis=1,ddof=1).mean(axis=0);B=N*np.var(z.mean(axis=1),axis=0,ddof=1)
        return np.sqrt(((N-1)*W+B)/(N*W))
    return np.maximum(calc(a),calc(abs(a-np.median(a,axis=(0,1))))).tolist()
records=[];start=time.perf_counter()
for stage in range(4):
    newchains=[];newlogs=[];newaccept=[]
    steps=88000 if stage==0 else 64000
    seeds=[20261024+2*stage,20261025+2*stage]
    for run,seed in enumerate(seeds):
        np.random.seed(seed)
        sampler=emcee.EnsembleSampler(32,4,like.log_probability)
        sampler.run_mcmc(chains[run,-1],steps,progress=False)
        newchains.append(sampler.get_chain());newlogs.append(sampler.get_log_prob());newaccept.append(sampler.acceptance_fraction)
    oldsteps=chains.shape[1]
    acceptance=(acceptance*oldsteps+np.array(newaccept)*steps)/(oldsteps+steps)
    chains=np.concatenate([chains,np.stack(newchains)],axis=1)
    logs=np.concatenate([logs,np.stack(newlogs)],axis=1)
    tau_values=np.array([emcee.autocorr.integrated_time(c,quiet=True) for c in chains])
    tau=float(tau_values.max());burn=max(1000,int(5*tau));thin=max(1,int(tau/10))
    raw=chains[:,burn:];flat=raw[:,::thin].reshape(-1,4)
    quant=np.percentile(flat,[16,50,84],axis=0).T
    rh=rhat(raw,tau)
    diff=float(max(abs(quant[2]-reference))/(reference[2]-reference[0]))
    records.append({'steps_per_ensemble':chains.shape[1],'continuation_seeds':seeds,
        'rank_split_rhat':rh,'eta_quantiles':quant[2].tolist(),
        'eta_quantile_difference_as_fraction_of_68_width':diff})
    print('JOINT REFINEMENT',json.dumps(records[-1]),flush=True)
    if diff<.02 and max(rh)<1.01 and (chains.shape[1]-burn)>80*tau:break
else:raise RuntimeError('Joint refinement did not reach the independent integration target')
np.save(ANA/'posterior_joint.npy',flat)
np.savez_compressed(ANA/'raw_chain_joint_refined.npz',chain=chains,log_posterior=logs,
    burn=burn,thin=thin,tau=tau_values,acceptance=acceptance,seeds=[42,20261023])
# Keep the original saved chain as an audit snapshot; make the refined one the
# active raw-chain file. These are within the new isolated output directory.
(ANA/'raw_chain_joint.npz').replace(ANA/'raw_chain_joint_initial112000.npz')
(ANA/'raw_chain_joint_refined.npz').rename(ANA/'raw_chain_joint.npz')
r.update(quantiles=quant.tolist(),n_samples=len(flat),steps_per_ensemble=chains.shape[1],burn=burn,thin=thin,
    tau=tau_values.tolist(),rank_split_rhat=rh,retained_steps_over_tau=float((chains.shape[1]-burn)/tau),
    ensemble_quantiles=[np.percentile(c[::thin].reshape(-1,4),[16,50,84],axis=0).T.tolist() for c in raw],
    mean_acceptance=acceptance.mean(axis=1).tolist(),refinement=records,seconds=r['seconds']+time.perf_counter()-start)
(ANA/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('JOINT REFINEMENT COMPLETE',flush=True)
