"""Reproduce AAS76998 calculations in a fresh working directory on CPU."""
from pathlib import Path
import argparse,datetime,json,os,shutil,subprocess,sys,uuid
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent

def run(work,script,*args):
    command=[sys.executable,'-X','utf8','-u',str(work/'code'/script),*map(str,args)]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    log=work/'logs'/(script+'.log');print('Running',script,flush=True)
    with log.open('w',encoding='utf-8') as f:
        p=subprocess.run(command,cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:
        print(log.read_text(encoding='utf-8',errors='replace')[-7000:]);raise RuntimeError(f'{script} failed; see {log}')
    print('Completed',script,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Verify the package and frozen numerical products without fitting')
    parser.add_argument('--mode',choices=['full','figures','historical','historical-mcmc','baseline-mcmc'],default='full')
    parser.add_argument('--output',type=Path,help='New empty output directory; default _runs/run_<id>')
    args=parser.parse_args()
    from verify_package import verify
    print(json.dumps(verify(),indent=2),flush=True)
    if args.check:
        target=ROOT/'_runs/check';target.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
        p=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'code/verify_results.py'),
                          '--analysis',str(ROOT/'reference_results/analysis'),'--report',str(target/'checks.json')],env=env)
        raise SystemExit(p.returncode)
    work=args.output.resolve() if args.output else ROOT/'_runs'/('run_'+uuid.uuid4().hex[:8])
    if work==ROOT or (work.exists() and any(work.iterdir())):raise ValueError('Use a new/empty output directory; frozen files will not be overwritten')
    work.mkdir(parents=True,exist_ok=True)
    for folder in ['code','data','baseline','original']:shutil.copytree(ROOT/folder,work/folder,ignore=shutil.ignore_patterns('__pycache__','*.partial_*'))
    for folder in ['analysis','audit','figures','logs']:(work/folder).mkdir(exist_ok=True)
    print('Work directory:',work,flush=True)
    if args.mode in ['historical','historical-mcmc']:
        run(work,'reproduce_historical.py',*(['--sample'] if args.mode.endswith('mcmc') else []));return
    if args.mode=='baseline-mcmc':
        for script in ['recalculate_baseline_mcmc.py','verify_baseline_integrals.py','refine_baseline_joint.py','verify_baseline_integrals.py']:run(work,script)
        return
    if args.mode=='figures':shutil.copytree(ROOT/'reference_results/analysis',work/'analysis',dirs_exist_ok=True)
    else:
        for script in ['reproduce_historical.py','complete_review_checks.py','gp_same_milp_anchors.py','scatter_sensitivity.py']:run(work,script)
    for script in ['plot_baseline.py','make_review_figures.py']:run(work,script)
    run(work,'verify_results.py','--reference',ROOT/'reference_results/analysis')
    (work/'RUN_COMPLETE.json').write_text(json.dumps({'mode':args.mode,'returncode':0,'reference_compared':True,
        'scope':'full regenerates 42 supplemental posteriors and seven plot files; frozen baseline MCMC is a separate optional mode'},indent=2),encoding='utf-8')
    print('Completed. Results, figures and validation are in',work,flush=True)

if __name__=='__main__':main()
