# Enhancement audit — version 0.2

The project now implements the screenshot concepts as explicit calculations, model components, data contracts and research protocols. **Share as a research prototype with caveats. It does not establish loneliness-caused hypertension or cardiovascular damage.**

Reviewed 25 supplied attachments, 17 distinct binary images and 12 conceptual groups. Duplicates are mapped by content hash. Raw screenshots, including video-call faces, are excluded from the package. Source CSV remains external.

## Real-data findings

The new comparison uses all 82,767 source rows under the original grouped partitions, with 16,546 held out for evaluation. This test set was already examined during the initial build, so the extension is exploratory. The clinical-context set excludes the three psychological proxies and all wearable columns; all four feature sets use the same fixed HGB settings. Preprocessing, calibration and threshold selection use their designated partitions.

| target | domain | roc_auc | pr_auc | brier |
| --- | --- | --- | --- | --- |
| heart_disease | clinical_context | 0.889329 | 0.272733 | 0.036954 |
| heart_disease | clinical_psych_proxies | 0.889685 | 0.271684 | 0.036962 |
| heart_disease | clinical_wearable | 0.889646 | 0.271051 | 0.036986 |
| heart_disease | clinical_psych_wearable | 0.889781 | 0.269670 | 0.036987 |
| hypertension | clinical_context | 0.781276 | 0.229640 | 0.075757 |
| hypertension | clinical_psych_proxies | 0.781071 | 0.231490 | 0.075721 |
| hypertension | clinical_wearable | 0.779330 | 0.227048 | 0.075877 |
| hypertension | clinical_psych_wearable | 0.779461 | 0.228799 | 0.075847 |

Paired profile-cluster bootstrap differences (200 repetitions):

| target | contrast | delta_auroc | ci_low | ci_high | delta_ap |
| --- | --- | --- | --- | --- | --- |
| heart_disease | psych_vs_clinical | 0.000356 | -0.000782 | 0.001678 | -0.001048 |
| heart_disease | wearable_vs_clinical | 0.000317 | -0.000829 | 0.001282 | -0.001681 |
| heart_disease | psych_added_to_wearables | 0.000135 | -0.000865 | 0.001320 | -0.001381 |
| hypertension | psych_vs_clinical | -0.000205 | -0.001397 | 0.000787 | 0.001850 |
| hypertension | wearable_vs_clinical | -0.001946 | -0.003262 | -0.000466 | -0.002592 |
| hypertension | psych_added_to_wearables | 0.000131 | -0.000602 | 0.001028 | 0.001751 |

Psychological addition has AUROC and average-precision intervals spanning zero for both recorded targets. This dataset does not demonstrate a reliable incremental predictive gain from these proxies. Sleep and activity remain in clinical context, so the contrast is conditional on them.

Adjusted cross-sectional associations:

| target | exposure | term | n | odds_ratio | ci95_low | ci95_high | q_bh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hypertension | bad_mental_days | exposure_per_observed_sd | 82767 | 1.004782 | 0.977860 | 1.032445 | 0.882871 |
| heart_disease | bad_mental_days | exposure_per_observed_sd | 82767 | 1.318382 | 1.270795 | 1.367751 | 0.000000 |
| heart_disease | bad_mental_days | recorded_hypertension | 82767 | 1.528435 | 1.399175 | 1.669635 | 0.000000 |
| heart_disease | bad_mental_days | exposure_x_hypertension | 82767 | 0.982134 | 0.902356 | 1.068964 | 0.882871 |
| hypertension | TENSE/ANXIOUS | exposure_per_observed_sd | 73313 | 1.001993 | 0.975803 | 1.028886 | 0.882871 |
| heart_disease | TENSE/ANXIOUS | exposure_per_observed_sd | 73313 | 1.017855 | 0.977559 | 1.059812 | 0.780989 |
| heart_disease | TENSE/ANXIOUS | recorded_hypertension | 73313 | 1.502525 | 1.369104 | 1.648949 | 0.000000 |
| heart_disease | TENSE/ANXIOUS | exposure_x_hypertension | 73313 | 1.009192 | 0.922609 | 1.103901 | 0.882871 |

`bad_mental_days` is associated with recorded heart disease in the specified model. `TENSE/ANXIOUS` does not show a clear association with either target. These statements concern available raw proxies, not diagnosed anxiety or loneliness. For the heart-disease models, the exposure main effect is at hypertension=0, and the interaction is a ratio of odds ratios. Residual confounding, reverse causation and uncertain source linkage remain unresolved.

## Executed synthetic components

The entirely synthetic four-modality network used 800 training, 200 validation and 200 test participants; checkpoint selection used validation loss over 30 epochs.

- Next-week mean SBP MAE: 2.692 mmHg; DBP MAE: 1.978 mmHg.
- On 131 matched cases with both last BP readings observed, model SBP MAE: 2.702; persistence MAE: 3.764 mmHg.
- Noisy next-night acute-state AUROC: 0.620; Brier: 0.243. This does not validate minute/hour detection.
- Right-censored event likelihood loss: 0.654. This is not a clinical CVD accuracy score.
- Five-coordinate Kalman filter and descriptive BP summaries executed on 84 invented days. HPA/immune axes have no assay support.
- Synthetic 12-week trial injected effect: −4.000 mmHg. Centered HT estimate: -4.319, 95% participant-bootstrap interval [-5.250, -3.476] mmHg.
- Restricted proximal-trial effects, two prespecified state strata, fixed-policy IPW value, effective sample size and safe-utility comparison executed. These calculations demonstrate code under invented assumptions, not intervention efficacy.

The synthetic MRT estimate is -3.120 mmHg (95% interval -3.746 to -2.441); the realized eligible mean injected effect is -3.973. The interval does not contain that injected mean in this run. A single realization is not a coverage study. The seed and estimator were not selected to make the interval contain the truth; repeated-simulation coverage validation remains outstanding.

The raw, uncentered HT estimate is retained in the JSON to expose finite-sample weight variance. No reported synthetic number is transferred to real patients. The continuous-time Cox formulation from the references is documented; the executed event head uses a discrete-time likelihood instead.

## Audit findings and repairs

| severity | area | finding | resolution_or_limit | evidence |
| --- | --- | --- | --- | --- |
| High | Scientific scope | Loneliness and measured BP are absent; causation is not identifiable | Resolved in presentation and data gates; missing-data requirement remains | reports/research_capabilities.csv |
| Medium | Clinical baseline definition | Historical baseline contained bad_mental_days, obscuring psychological addition | Added explicit domain partitions and matched settings; retained historical results | scripts/analyze_psychosocial.py |
| High | Enrollment timing | Seventh observation previously entered its own scoring reference | Fixed: seven enrollment readings remain unscored; first score on night eight | src/twin/digital_twin.py |
| Medium | Missing-value handling | Legacy GRU trusted callers to zero masked readings | Fixed: both encoder implementations explicitly zero unobserved values | src/models/wearable_encoder.py |
| Medium | Empty neural input | Legacy inference could produce a score for an entirely missing trajectory | Fixed: no-observation inference now returns unavailable | src/models/fusion_training.py |
| Medium | Architecture diagram | An arrow implied experimental fusion updated the live twin state | Removed misleading link; added separate version 0.2 research architecture page | docs/architecture.pdf |
| High | Temporal alignment | Source has no timestamps or availability times | Added as-of contract and regression tests; real-source forecasting remains unavailable | src/research/longitudinal.py |
| High | Censoring and missing targets | Incident-event and multitask training require distinct label handling | Added censored interval likelihood and per-head label masks; exercised synthetically | src/research/multimodal.py |
| High | Causal policy evaluation | Observational classification score changes cannot estimate treatment effects | Added restricted trial estimators with positivity, eligibility, clustering and overlap checks | src/research/causal.py |
| High | Latent physiology | Biological axis names alone cannot identify ANS/HPA/immune mechanisms | Explicitly label synthetic assumed matrices and measurement support per axis | src/research/state_space.py |
| High | Study validity | Previously inspected test set, unknown wearable linkage, no external validation | Unresolved: new comparisons are exploratory; external prospective study required | docs/research_extension.md |

## Verification

Latest full pytest run: 45 tests, 0 failures, 0 errors, 0 skipped.
Tests include independently computed BP formulas, exact balanced-trial IPW/ATE values, manual survival likelihood, no post-event loss, missing-label exclusion, masked-value invariance, future/late-result exclusion, invalid-propensity rejection, overlapping decision rejection, state covariance and enrollment timing. All dashboard pages are exercised by Streamlit AppTest.

Visual browser checks and final ZIP extraction verification are recorded separately in `outputs/metrics/enhancement_visual_check.json` and the ZIP's adjacent verification JSON. Their presence must be checked before claiming those stages passed.

## Run and review

```powershell
python scripts/analyze_psychosocial.py --data "PATH/FINAL_dataset_with_fitbit.csv"
python scripts/run_research_demo.py
python -m pytest tests/ -q --junitxml=outputs/metrics/final_tests.xml
python scripts/build_enhancement_report.py
python -m streamlit run dashboard/app.py
python scripts/package_project.py --verify
```

Open **Loneliness and BP Research** in the dashboard. See `docs/research_extension.md` for the full equations, biomarker priorities, four research questions, 12-week protocol, assumptions and primary references. The four empty `data/templates/` CSVs define future collection formats; they contain no fabricated source measurements.

Original model results are preserved. New synthetic checkpoint: `models/saved/longitudinal_synthetic.pt`. Source-derived model weights still require source-rights review before redistribution; raw source data is not included.
