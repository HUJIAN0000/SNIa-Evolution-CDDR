"""Publication figures: preserve original triangles; add readable 2D diagnostics."""
import os,sys
sys.dont_write_bytecode=True;os.environ['MPLBACKEND']='Agg'
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matching import ROOT,ANA,K,LM0,inputs
from gp import GP

FIG=ROOT/'figures';MAN=ROOT/'manuscript';FITS=ANA/'fits'
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':11,
 'legend.fontsize':8,'xtick.labelsize':9,'ytick.labelsize':9,'axes.linewidth':.8,
 'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
 'savefig.bbox':'tight'})

def finish(fig,name):
    fig.savefig(FIG/(name+'.pdf'));fig.savefig(FIG/(name+'.png'),dpi=160)
    plt.close(fig)

def main():
    sn,_,C=inputs();idx=np.flatnonzero(sn.zHD.to_numpy()>=.3)
    gp=GP(sn.zHD.to_numpy()[idx],sn.m_b_corr.to_numpy()[idx],C[np.ix_(idx,idx)]).fit()
    z=np.linspace(.32,2.25,300);mean,V=gp.predict(z);sd=np.sqrt(np.diag(V))
    p=np.load(ANA/'gp_matern_prediction.npz');t=pd.read_csv(ROOT/'baseline/pairing/matched_cluster_milp_dd_pantheon.csv')
    mu=5*np.log10(LM0*206.265/t.theta.to_numpy()*(1+t.z.to_numpy())**2)+25
    fit=json.loads((FITS/'gp_matern_linear_none.json').read_text())
    a,e=fit['profile_coefficients'];radio=mu+a+e*t.z.to_numpy()
    fig,(ax,bx)=plt.subplots(2,1,figsize=(7,6.2),sharex=True,gridspec_kw={'height_ratios':[2.1,1],'hspace':.06})
    ax.scatter(sn.zHD.to_numpy()[idx],sn.m_b_corr.to_numpy()[idx],s=6,color='.55',alpha=.35,label='Pantheon+ training rows',rasterized=True)
    ax.fill_between(z,mean-1.96*sd,mean+1.96*sd,color='#d6b898',alpha=.55,label='GP 95% marginal band')
    ax.plot(z,mean,color='#234d7d',lw=1.4,label='GP mean')
    radioerr=K*np.sqrt((t.dtheta/t.theta)**2+.1**2)
    ax.errorbar(t.z,radio,yerr=radioerr,fmt='o',ms=3,color='#234d7d',ecolor='#7798b4',elinewidth=.7,capsize=0,label='CRS prediction at conditional best fit')
    ax.set_ylabel(r'Corrected apparent magnitude');ax.legend(loc='lower right',frameon=False)
    bx.errorbar(t.z,radio-p['mean'],yerr=np.sqrt(radioerr**2+np.diag(p['covariance'])),fmt='o',ms=3,color='#234d7d',elinewidth=.8)
    bx.axhline(0,color='.35',lw=.8);bx.set_ylabel(r'CRS $-$ GP [mag]');bx.set_xlabel(r'Redshift $z$');bx.set_xlim(.38,2.3)
    finish(fig,'review_gp')
    labels=[('historical_linear_none','Original 48; fixed 10%'),('historical_signed_transport','Same 48; signed transport'),
      ('dd05_linear_none','MILP 50; signed transport'),('gp_milp50_linear_none','GP at same 50 CRS anchors'),
      ('dz005_linear_none',r'Strict $|\Delta z|\leq0.005$; 27 pairs'),('gp_matern_linear_none','GP at original 48 anchors'),
      ('gp_rbf_linear_none','GP 48; squared-exponential kernel'),('gp_quadratic_linear_none','GP 48; quadratic mean'),
      ('historical_scatter05','Original 48; inferred extra scatter'),('strict_scatter05','Strict 27; inferred extra scatter'),
      ('gp_scatter05','GP 48; inferred extra scatter')]
    fig,ax=plt.subplots(figsize=(7.2,5.7))
    for i,(tag,label) in enumerate(labels):
        r=json.loads((FITS/(tag+'.json')).read_text())
        if 'quantiles_025_16_50_84_975' in r:q=np.array(r['quantiles_025_16_50_84_975']['epsilon']);color='#9b613c'
        else:
            q=np.percentile(np.load(FITS/(tag+'_posterior.npy'))[:,1],[2.5,16,50,84,97.5]);color='#234d7d'
        y=len(labels)-1-i;ax.plot(q[[0,4]],[y,y],color=color,lw=1);ax.plot(q[[1,3]],[y,y],color=color,lw=3)
        ax.plot(q[2],y,'o',color=color,ms=4)
    ax.axvline(0,color='.4',lw=.8,ls='--');ax.set_yticks(range(len(labels)),[l for _,l in labels][::-1])
    ax.axhline(2.5,color='.75',ls=':',lw=.8);ax.set_xlabel(r'Conditional linear evolution $\epsilon_1$ [mag]')
    ax.set_ylim(-.7,len(labels)-.3);ax.set_xlim(-.8,1.15);ax.grid(axis='x',alpha=.15)
    fig.subplots_adjust(left=.45,right=.98,top=.98,bottom=.11);finish(fig,'review_comparison')
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,2.8),gridspec_kw={'wspace':.32})
    for high,color in [(2,'#234d7d'),(3,'#9b613c'),(4,'#705182')]:
        tag='historical_linear_linear' if high==2 else f'historical_wide{high}_linear_linear'
        p=np.load(FITS/(tag+'.npz'));ax.plot(p['eta'],p['density'],color=color,lw=1.5,label=rf'$\eta_0<{high}$')
        if high==4:bx.plot(p['eta'],p['chi2_profile']-min(p['chi2_profile']),color='#234d7d',lw=1.5)
    ax.set_ylabel('Marginal posterior density');ax.set_xlabel(r'$\eta_0$');ax.legend(frameon=False)
    bx.set_ylabel(r'$\chi^2(\eta_0)-\chi^2_{\min}$');bx.set_xlabel(r'$\eta_0$')
    for a in [ax,bx]:a.set_xlim(-.45,4)
    bx.set_ylim(0,6);finish(fig,'review_prior')
    print('Three additional figures generated; original four triangles preserved.')

if __name__=='__main__':main()
