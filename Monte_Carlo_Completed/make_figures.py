"""Regenerate five figures from the corrected run outputs; no hard-coded results."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mplconfig'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':140,'savefig.bbox':'tight','axes.titlepad':14})
BLUE,ORANGE,OLIVE,PURPLE = '#285a91','#b96c25','#63784c','#796593'


def save(fig,path):
    fig.savefig(path,facecolor='white')
    plt.close(fig)


def main(out):
    R = pd.read_csv(out/'mc_run_metrics.csv')
    S = json.loads((out/'summary.json').read_text())
    A = json.loads((out/'data_audit.json').read_text())
    EX = json.loads((out/'example_patient_full.json').read_text())
    # Observed coverage is read before any imputation.
    fig,ax = plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    age = A['age_histogram']
    ax[0].bar([int(x) for x in age],list(age.values()),width=.9,color=BLUE)
    ax[0].set(title=f'Eligible ages (n={A["cohort_rows"]:,})',xlabel='Age (years)',ylabel='Patients')
    fields = ['nremhr','rmssd','spo2','stress_score','sleep_points_percentage']
    ax[1].barh(['Resting HR','HRV (RMSSD)','SpO2','Stress score','Sleep points'],[A['observed_coverage'][k] for k in fields],color=ORANGE)
    ax[1].set(title='Observed wearable-summary coverage',xlabel='Non-missing fraction',xlim=(0,1))
    ax[1].xaxis.set_major_formatter(PercentFormatter(1))
    ax[1].invert_yaxis()
    conditions = A['condition_prevalence']
    ax[2].barh([x.replace('_',' ').title() for x in conditions],list(conditions.values()),color=OLIVE)
    ax[2].set(title='Source condition flags',xlabel='Fraction of eligible cohort')
    ax[2].xaxis.set_major_formatter(PercentFormatter(1))
    ax[2].invert_yaxis()
    fig.suptitle('Data fusion inputs: one summary row per patient',fontsize=16)
    save(fig,out/'fig1_data_fusion.png')

    fig,ax = plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    cols = ['static_auc','wearable_no_cgm_auc','fusion_auc','full_auc']
    ax[0].boxplot([R[k].dropna() for k in cols],tick_labels=['Clinical','+ Wearables','+ CGM*','+ Psych'],patch_artist=True,
                  boxprops={'facecolor':'#c3d4e5'},medianprops={'color':BLUE,'linewidth':2})
    ax[0].set(title='AUROC for the same 2-hour target',ylabel='AUROC',ylim=(.5,1))
    ax[0].tick_params(axis='x',rotation=20)
    z = np.load(out/'heldout_predictions_run0.npz')
    from sklearn.metrics import precision_recall_curve
    precision,recall,_ = precision_recall_curve(z['y'],z['p'])
    ax[1].plot(recall,precision,color=BLUE)
    ax[1].axhline(z['y'].mean(),color='grey',ls='--',label=f"Run-0 prevalence: {z['y'].mean():.2%}")
    ax[1].set(title='Precision–recall, held-out run 0',xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1))
    ax[1].legend(loc='upper right',fontsize=9)
    ax[2].scatter(R.full_ece,R.full_brier,color=OLIVE,alpha=.7)
    ax[2].axhline(R.brier_constant_baseline.mean(),ls='--',color='grey',label='Mean constant-risk Brier')
    ax[2].set(title='Calibration and prediction error',xlabel='ECE (10 fixed-width bins)',ylabel='Brier score')
    ax[2].legend(fontsize=9)
    fig.suptitle(f'{len(R)} Monte Carlo runs; held-out original patient identities',fontsize=16)
    fig.supxlabel('*CGM and all longitudinal physiology are simulated. Weekly Model A has a different target and is excluded.',fontsize=10)
    save(fig,out/'fig2_model_performance.png')

    fig,ax = plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    names = ['lo','mid','hi']
    for i,k in enumerate(names):
        v = R['beta_'+k]
        low,high = v.quantile([.025,.975])
        ax[0].plot([i,i],[low,high],color=BLUE,lw=2)
        ax[0].scatter([i],[v.mean()],color=BLUE,s=45,zorder=3)
        truth = R['beta_true_'+k].mean()
        ax[0].plot([i-.12,i+.12],[truth,truth],color='black',lw=2)
    ax[0].set(xticks=[0,1,2],xticklabels=['Low baseline risk','Mid baseline risk','High baseline risk'],
              ylabel='Next-day peak reduction (mg/dL)',title='Randomized simulator effect recovery')
    ax[0].plot([],[],color='black',lw=2,label='Mean effect imposed in simulator')
    ax[0].plot([],[],'o',color=BLUE,label='Estimate and central 95% MC range')
    ax[0].legend(fontsize=9)
    ax[1].hist(R.spike_rate*100,bins=10,color=ORANGE,edgecolor='white')
    ax[1].set(title='Synthetic event prevalence',xlabel='Positive held-out 2-hour windows (%)',ylabel='Monte Carlo runs')
    fig.supxlabel('Cluster-robust trial intervals are saved per run. Imposed effects are not evidence of clinical benefit.',fontsize=10)
    save(fig,out/'fig3_causal_effect.png')

    fig,ax = plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    v = [R[k].mean() for k in cols]
    ax[0].bar(['Clinical','+ Wearables','+ CGM*','+ Psych'],v,color=BLUE)
    ax[0].set(title='Same-target predictive performance',ylabel='Mean AUROC',ylim=(0,1))
    for i,x in enumerate(v): ax[0].text(i,x+.02,f'{x:.3f}',ha='center')
    dcols = ['delta_wearable','delta_cgm','delta_psych']
    for i,k in enumerate(dcols):
        mean,lo,hi = R[k].mean(),R[k].quantile(.025),R[k].quantile(.975)
        ax[1].plot([lo,hi],[i,i],color=BLUE,lw=2)
        ax[1].plot(mean,i,'o',color=BLUE)
        ax[1].annotate(f'{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]',(mean,i),xytext=(0,14),textcoords='offset points',ha='center',fontsize=9)
    ax[1].axvline(0,ls='--',color='grey')
    ax[1].set(yticks=[0,1,2],yticklabels=['Add wearables','Add simulated CGM','Add psych context'],
              xlabel='Paired change in AUROC',title='Incremental model comparisons',ylim=(-.6,2.8))
    ax[1].invert_yaxis()
    fig.supxlabel('Central 95% simulation ranges across paired runs. Predictive increments are not causal effects.',fontsize=10)
    save(fig,out/'fig4_value_decomposition.png')

    fig,ax = plt.subplots(5,1,figsize=(13,11),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,1,1,1,1]})
    series,dec = EX['series'],EX['decision']
    h = np.array(series['hours'])
    ax[0].plot(h,series['G'],color=BLUE,label='Simulated glucose')
    ax[0].axhline(180,ls='--',color=ORANGE,label='Modeled spike threshold')
    ax[0].set(ylabel='Glucose (mg/dL)',title=f"{EX['profile']['virtual_patient_id']}: held-out example selected by baseline risk")
    ax[0].legend(loc='upper right',fontsize=9)
    ax[1].plot(dec['hours'],dec['p'],color=BLUE)
    ax[1].axhline(dec['threshold'],ls='--',color=ORANGE,label=f"Validation-selected threshold ({dec['threshold']:.3f})")
    ax[1].set(ylabel='2-h risk',ylim=(0,1))
    ax[1].legend(fontsize=9)
    for a,key,label,color in [(ax[2],'HR','HR (bpm)',PURPLE),(ax[3],'SBP','SBP (mmHg)',OLIVE),(ax[4],'RM','HRV (ms)',BLUE)]:
        a.plot(h,series[key],color=color,lw=.9)
        a.set(ylabel=label)
    for a in ax:
        a.axvspan(0,24,color='grey',alpha=.12)
        a.grid(axis='x',alpha=.2)
    ax[-1].set(xlabel='Simulation hour',xlim=(0,168))
    fig.suptitle('Virtual patient: source-derived baseline, synthetic 15-minute trajectories',fontsize=15)
    fig.supxlabel('Grey region: 24-hour initialization burn-in. Every displayed prediction uses data available at that decision time.',fontsize=10)
    save(fig,out/'fig5_virtual_patient.png')
    print('Saved five corrected figures.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'outputs')
    main(p.parse_args().out)
