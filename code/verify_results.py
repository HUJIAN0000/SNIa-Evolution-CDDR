"""Check raw indices, likelihood identity, posteriors and optional independent replay."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MPLBACKEND']='Agg'
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor
from prepare_data import ensure_inputs
ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,default=ROOT/'analysis')
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--report',type=Path,default=ROOT/'audit/result_verification.json')
    args=parser.parse_args();ana=args.analysis.resolve()
    inputs=ensure_inputs(ROOT/'data')
    sn=pd.read_csv(ROOT/'data/Pantheon+SH0ES.dat',sep=r'\s+',dtype={'CID':str})
    radio=pd.read_csv(ROOT/'data/ucsdata.CSV')
    c=np.loadtxt(ROOT/'data/Pantheon+SH0ES_STAT+SYS.cov',skiprows=1).reshape(len(sn),len(sn))
    cho_factor((c+c.T)/2)
    t=pd.read_csv(ROOT/'baseline/pairing/matched_cluster_milp_dd_pantheon.csv')
    sub=np.loadtxt(ROOT/'baseline/pairing/cov_cluster_milp_dd_pantheon_lens.txt')
    idx=t.sn_matched_idx.to_numpy(dtype=int)
    assert len(t)==48 and np.array_equal(sub,c[np.ix_(idx,idx)])
    assert np.array_equal(t.sn_matched_m_b_corr,sn.iloc[idx].m_b_corr)
    assert np.array_equal(t.theta,radio.iloc[t.obj_index.astype(int)].theta)
    old=load('old_model_check',ROOT/'original/mcmc_sn_evolutionv5m.py')
    new=load('fixed10_check',ROOT/'code/mcmc_sn_evolutionv5m_error10.py')
    identities={};baseline=json.loads((ROOT/'baseline/results.json').read_text())
    for mode in ['evol','log_evol','ddr','joint']:
        shared=(t.z,t.theta,t.dtheta,t.sn_matched_m_b_corr,sub)
        a=old.SNEvolutionLikelihood(*shared,fit_mode=mode)
        b=new.SNEvolutionLikelihood(*shared,fit_mode=mode,extra_fraction=0)
        chain=np.load(ROOT/f'original/sn_evolution_v4_chain_{mode}.npy')
        delta=max(abs(a.log_probability(p)-b.log_probability(p)) for p in chain[::97])
        assert delta==0;identities[mode]=float(delta)
        fixed=np.load(ROOT/f'baseline/posterior_{mode}.npy')
        assert np.array_equal(np.percentile(fixed,[16,50,84],axis=0).T,baseline['modes'][mode]['quantiles'])
    fit_files=sorted((ana/'fits').glob('*.json'));assert len(fit_files)==42
    comparisons={};exact_count=0
    for p in fit_files:
        result=json.loads(p.read_text());s=np.load(p.with_name(p.stem+'_posterior.npy'))
        assert s.shape==(100000,4) and np.isfinite(s).all(),p.name
        assert np.all((-21<s[:,0])&(s[:,0]<-18)&(-5<s[:,1])&(s[:,1]<5)&(5<s[:,3])&(s[:,3]<20))
        if 'quantiles' in result:
            keys=['M_B0','epsilon','eta0','lm_pc'];q=np.percentile(s,[16,50,84],axis=0).T
            assert result['integration_refinement_max_eta_change']<1e-4
        else:
            keys=['M_B0','epsilon','scatter_mag','lm_pc'];q=np.percentile(s,[2.5,16,50,84,97.5],axis=0).T
            assert np.all((0<s[:,2])&(s[:,2]<2)) and result['s_grid_refinement_maxchange']<1e-4
        actual=result.get('quantiles',result.get('quantiles_025_16_50_84_975'))
        for i,key in enumerate(keys):assert np.array_equal(q[i],actual[key]),(p.name,key)
        if args.reference:
            ref=np.load(args.reference/'fits'/(p.stem+'_posterior.npy'))
            maxdiff=float(np.max(abs(s-ref)));equal=np.array_equal(s,ref)
            exact_count+=int(equal)
            qref=np.percentile(ref,[16,50,84],axis=0).T
            qnew=np.percentile(s,[16,50,84],axis=0).T
            tol=.02*(qref[:,2]-qref[:,0])+1e-8
            assert np.all(np.max(abs(qnew-qref),axis=1)<=tol),f'Reference quantiles differ materially: {p.stem}'
            comparisons[p.stem]={'sample_max_absolute_difference':maxdiff,'samples_elementwise_equal':equal}
    for tag in ['dd05','dz005','dd02','dd03','dd07','om028','om032','cpl']:
        r=json.loads((ana/f'matching_{tag}.json').read_text())
        assert r['optimal'] and all(x['status']==0 and x['mip_gap']==0 for x in r['stages'])
    main=pd.read_csv(ana/'matches_dd05.csv');strict=pd.read_csv(ana/'matches_dz005.csv')
    assert len(main)==50 and len(strict)==27 and max(abs(strict.delta_z))<=.005
    for tag in ['om028','om032','cpl']:
        tt=pd.read_csv(ana/f'matches_{tag}.csv');assert np.array_equal(main[['obj_index','sn_row']],tt[['obj_index','sn_row']])
    if args.reference:
        for tag in ['dd05','dz005','dd02','dd03','dd07','om028','om032','cpl']:
            aa=pd.read_csv(ana/f'matches_{tag}.csv');bb=pd.read_csv(args.reference/f'matches_{tag}.csv')
            assert np.array_equal(aa[['obj_index','sn_row']],bb[['obj_index','sn_row']])
            assert np.array_equal(np.load(ana/f'cov_sn_{tag}.npy'),np.load(args.reference/f'cov_sn_{tag}.npy'))
    delta=baseline['modes']['joint']['chi2']-baseline['modes']['evol']['chi2']
    scatter=json.loads((ana/'fits/historical_scatter05.json').read_text())
    report={'sn_rows':len(sn),'radio_rows':len(radio),'raw_sha256':{k:v['sha256'] for k,v in inputs.items()},
            'historical_pairs':48,'baseline_row_covariance_alignment':True,
            'zero_extra_error_likelihood_identity':identities,'posterior_models_checked':len(fit_files),
            'matching_certificates_checked':8,'MILP_pairs':50,'strict_pairs':27,
            'strict_radio_zmax':float(strict.z.max()),'replay_posteriors_elementwise_identical':exact_count if args.reference else None,
            'replay_comparisons':comparisons,
            'descriptive_fixed_covariance_delta_AIC_joint_minus_linear':float(delta+2),
            'descriptive_fixed_covariance_delta_BIC_joint_minus_linear':float(delta+np.log(48)),
            'baseline_empirical_scatter_epsilon95':[scatter['quantiles_025_16_50_84_975']['epsilon'][i] for i in [0,4]],
            'scope':'Numerical reproducibility, not validation of a physical residual model or independent joint detection.'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Verified 42 posterior models, 8 matching certificates, raw covariance indices and original likelihood identity.')
    if args.reference:print('Independent replay: elementwise-identical posteriors',exact_count,'/ 42.')

if __name__=='__main__':main()
