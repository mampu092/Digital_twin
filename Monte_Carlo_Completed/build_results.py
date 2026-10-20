"""Create a readable numerical report from completed outputs."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main(out):
    S = json.loads((out/'summary.json').read_text())
    A = json.loads((out/'data_audit.json').read_text())
    R = pd.read_csv(out/'mc_run_metrics.csv')
    M = S['metrics']
    def stat(k,d=3,pct=False):
        z=M[k]
        scale=100 if pct else 1
        unit='%' if pct else ''
        return f"{z['mean']*scale:.{d}f}{unit} [{z['low']*scale:.{d}f}{unit}, {z['high']*scale:.{d}f}{unit}]"
    rows=[]
    for label,prefix in [('Clinical baseline','static'),('Clinical + wearables / simulated BP','wearable_no_cgm'),
                         ('Add simulated CGM','fusion'),('Add psychological context','full')]:
        rows.append(f'| {label} | {stat(prefix+"_auc")} | {stat(prefix+"_ap")} | {stat(prefix+"_brier",4)} |')
    coverage='\n'.join(f'| {k} | {v:.1%} |' for k,v in A['observed_coverage'].items())
    p=A['privacy']
    text=f'''# Completed Monte Carlo outputs

The completed run used **{S['cohort_n']:,} eligible source patients aged 18–60**, {S['mc_runs']} independent seeded simulations, {S['patients_per_run']:,} bootstrap trajectories per simulation, seven days and 15-minute steps. The prediction target is a **synthetic glucose event in the next two hours** after a 24-hour burn-in.

All bracketed ranges below are central 95% Monte Carlo ranges across runs, not clinical confidence intervals. The CSV contains summary data; **all longitudinal observations and event labels are simulated**. No real-world predictive accuracy or treatment benefit is established.

## Predictive model outputs

| Model, same held-out 2-hour target | AUROC | Average precision | Brier score |
|---|---:|---:|---:|
{chr(10).join(rows)}

- Test positive-window prevalence: {stat('spike_rate',2,True)}.
- Constant training-prevalence predictor Brier score: {stat('brier_constant_baseline',4)}. Lower Brier is better.
- Full-model ECE, ten fixed-width probability bins: {stat('full_ece',4)}. This coarse ECE can be small with rare events and does not alone establish good calibration.
- Two-hour SBP MAE: {stat('mae_sbp_mmHg',2)} mmHg.
- Two-hour stress-index MAE: {stat('mae_stress',4)} on the 0–1 scale.
- Weekly clinical-only Model A AUROC: {stat('auc_A_weekly')}. Its target is any modeled weekly glucose decision-window event or the synthetic cardiovascular rule, so it is not ranked against Model B.
- Synthetic cardiovascular rule prevalence: {stat('cardio_rate',2,True)}. Any modeled weekly event: {stat('any_event_rate',2,True)}.

![Model evaluation](fig2_model_performance.png)

## Incremental predictive value

| Addition to the preceding model | Paired AUROC change |
|---|---:|
| Wearable summaries and simulated wearable/BP history | {stat('delta_wearable')} |
| Simulated CGM | {stat('delta_cgm')} |
| Psychological context and simulated stress | {stat('delta_psych')} |

These are paired predictive ablations. A central simulation range spanning zero does not support a consistent gain under the tested generator. Their sum is the mean full-minus-clinical AUROC difference. The simulator is partly constructed from clinical features, so its ablation results are specific to these design assumptions.

![Value comparisons](fig4_value_decomposition.png)

## Alert outputs on test patients

| Operating point | Precision | Window recall | Alerts per 100 decision windows |
|---|---:|---:|---:|
| Threshold selected by validation F2 | {stat('selected_precision',1,True)} | {stat('selected_recall',1,True)} | {stat('selected_alerts_per_100_windows',2)} |
| Original fixed threshold 0.30 | {stat('fixed_precision',1,True)} | {stat('fixed_recall',1,True)} | {stat('fixed_alerts_per_100_windows',2)} |

The validation-selected threshold is {stat('selected_threshold',3)}. Thresholds vary by run. Precision is undefined when no alert is emitted; valid-run counts are retained in summary.json. Metric means are means across runs, not pooled patient estimates. Rare-event prevalence is the relevant reference for average precision.

Mean lead among detected eligible threshold crossings: **{stat('alert_lead_min',1)} minutes**. Eligible crossing recall: {stat('crossing_recall',1,True)}. Only alerts strictly before the crossing and at most 120 minutes before it are counted. The maximum counted lead across runs is {R.max_lead_min.max():.0f} minutes. These crossing metrics differ from the window-label definition, which also permits a rise of 70 mg/dL.

## Randomized simulator outputs

Daily availability probability is 0.8, and prompt randomization probability is 0.5. The next-day peak benefit is imposed as a risk-dependent 4–12 mg/dL reduction. Treatment-centered regression uses only observed randomized outcomes and prior history. Per-run standard errors account for repeated days within each simulated trajectory.

| Baseline-risk stratum | Estimated reduction (mg/dL), MC range | Mean imposed effect (mg/dL) | Per-run cluster 95% interval coverage |
|---|---:|---:|---:|
| Low | {stat('beta_lo',2)} | {M['beta_true_lo']['mean']:.2f} | {M['beta_covered_lo']['mean']:.1%} |
| Mid | {stat('beta_mid',2)} | {M['beta_true_mid']['mean']:.2f} | {M['beta_covered_mid']['mean']:.1%} |
| High | {stat('beta_hi',2)} | {M['beta_true_hi']['mean']:.2f} | {M['beta_covered_hi']['mean']:.1%} |

This is a recovery check for a known synthetic intervention, not evidence that an activity prompt lowers a person's glucose by these amounts. Coverage estimates are coarse with only {S['mc_runs']} runs.

![Simulated effects](fig3_causal_effect.png)

## Data fusion and privacy outputs

Input: {A['source_rows']:,} rows, {A['source_columns']} columns, one source row per unique patient ID. The age rule excludes {A['excluded_age_rows']:,} rows outside 18–60 or with unavailable ages. Selected-cohort unique IDs: {A['unique_patient_ids']:,}. Duplicate IDs: {A['duplicate_ids']}.

There are only **{A['wearable_profile_reuse']['distinct_triplets']} distinct complete HR/HRV/breathing summary triplets across {A['wearable_profile_reuse']['complete_hr_hrv_breathing_rows']:,} rows**. The most repeated triplet occurs {A['wearable_profile_reuse']['largest_repeat_count']:,} times. Repeated values do not establish the same physical wearer, but prevent treating these rows as verified independent Fitbit participants. Evaluation separates supplied patient IDs; it cannot validate unprovided original wearable-user identities.

Observed, pre-imputation coverage:

| Input summary field | Observed fraction |
|---|---:|
{coverage}

![Data inputs](fig1_data_fusion.png)

There are no source timestamps to test real-time alignment, missing intervals or forecast labels. Train-fitted imputation and missingness flags support this prototype, but cannot recover a measured stream. Original linkage and source licensing require documentation before reuse outside this local research setting.

For age, gender, rounded BMI and smoking quasi-identifiers, **{p['unique_record_pct']:.2f}% of records are unique**, minimum k is {p['min_k']}, and {p['record_k_le_3_pct']:.2f}% of records lie in classes with k≤3. The fraction of classes with k≤3 is {p['classes_k_le_3_pct']:.2f}%, a different denominator. These diagnostics do not certify anonymization. No legal-compliance claim is made. The dashboard excludes source IDs and coarsens example ages; local replay vectors still contain source-derived features.

## Virtual-patient output

Five held-out examples are selected at baseline-risk quantiles independently of future events. Open dashboard.py through Streamlit to choose a patient, replay available history and inspect probabilities. Future trajectories require an explicit retrospective setting.

![Virtual patient](fig5_virtual_patient.png)

## Reproducibility and checks

- Original identity overlap across train, validation and test: {int(R.identity_overlap.max())} in every run.
- Completed run rows: {len(R)}. Each run uses a distinct recorded seed.
- Aggregated held-out windows: {int(R.test_windows.sum()):,}; this sum includes independent simulated trajectories and is not a count of distinct real people.
- Frozen run-0 model: model_full_run0.npz. Its saved scaling and weights can replay held-out feature vectors with predict.py.
- Input SHA-256, parameters and versions: provenance.json.
- Behavioral tests cover alignment, grouped bootstrap splits, train-only imputation, no-future rolling features, event-horizon boundaries, tied scores, scalar bias learning, confusion counts, model replay, input validation, and trial effect recovery. Check verification.json for the recorded test result and UI checks.

See ../README.md for formulas, limitations, repair details, launch commands, privacy references and interpretation. Results supersede the supplied JSON metrics and five original figures.
'''
    (out/'RESULTS.md').write_text(text,encoding='utf-8')
    print(f'Wrote {out / "RESULTS.md"}')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'outputs')
    main(parser.parse_args().out)
