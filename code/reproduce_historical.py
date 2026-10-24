"""Run the unmodified historical greedy matcher; optionally sample all old models."""
from pathlib import Path
import argparse,importlib.util,json,os,shutil,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MPLBACKEND']='Agg'
ROOT=Path(__file__).resolve().parents[1]
from prepare_data import ensure_inputs

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sample',action='store_true')
    args=parser.parse_args()
    import numpy as np
    import pandas as pd
    ensure_inputs(ROOT/'data')
    work=ROOT/'analysis/historical_mcmc';work.mkdir(parents=True,exist_ok=True)
    for name in ['CosmoMatcher_v1.1.py','mcmc_sn_evolutionv5m.py','cosmo_tools.py']:
        shutil.copy2(ROOT/'original'/name,work/name)
    sys.path.insert(0,str(work));os.chdir(work)
    backend=module('historical_matcher',work/'CosmoMatcher_v1.1.py')
    paths={'tracer':str(ROOT/'data/Pantheon+SH0ES.dat'),'obj':str(ROOT/'data/ucsdata.CSV'),
           'cov_pantheon':str(ROOT/'data/Pantheon+SH0ES_STAT+SYS.cov')}
    cols={'tracer_z_col':'zHD','z1_col':'z','z2_col':None,
          'output_cols':['CID','IDSURVEY','zHD','zHEL','m_b_corr','m_b_corr_err_DIAG']}
    backend.CosmoBackend.run_main_logic('cluster','pantheon','milp','dd',paths,cols,{'wm':.3,'tol':.05},print)
    t=pd.read_csv('matched_cluster_milp_dd_pantheon.csv')
    old=pd.read_csv(ROOT/'original/matched_cluster_milp_dd_pantheon.csv')
    assert np.array_equal(t[['obj_index','sn_matched_idx']],old[['obj_index','sn_matched_idx']])
    c=np.loadtxt('cov_cluster_milp_dd_pantheon_lens.txt')
    assert np.array_equal(c,np.loadtxt(ROOT/'original/cov_cluster_milp_dd_pantheon_lens.txt'))
    result={'pairs':len(t),'pair_indices_and_order_identical':True,'covariance_identical':True,
            'sampled':args.sample,'algorithm':'historical cluster branch is greedy despite the legacy milp option/file names'}
    if args.sample:
        model=module('historical_sampler',work/'mcmc_sn_evolutionv5m.py');model.main(model.CONFIG)
        result['models']={}
        for mode in ['evol','log_evol','ddr','joint']:
            s=np.load(work/f'sn_evolution_v4_chain_{mode}.npy')
            result['models'][mode]={'samples':len(s),'quantiles16_50_84':np.percentile(s,[16,50,84],axis=0).T.tolist()}
    (work/'reproduction_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
