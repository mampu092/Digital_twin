# Loneliness, anxiety and cardiovascular physiology — version 0.2

The research objective is to test whether persistent loneliness and anxiety precede changes in autonomic regulation and sustained blood pressure, and whether a defined psychosocial intervention improves an objectively measured cardiovascular outcome. **The supplied CSV cannot establish that causal chain.** The implemented extension separates cross-sectional calculations on the CSV from entirely synthetic longitudinal and randomized-trial exercises.

## What each reference became

All 25 attachments are recorded by filename, content hash and duplicate mapping in `reference_image_manifest.json`. Twelve conceptual groups cover the unique ideas. Several images show cropped formulas; the equations below specify the implementation rather than inventing the missing pixels.

| ID | Reference concept | Implemented component | Current evidence |
|---|---|---|---|
| R01 | Acute, intermediate and long-term horizons | Dashboard horizon registry, data capability checks, separate model heads | Acute head exercised on next-night synthetic labels; minute/hour stress validation still requires suitable sampling |
| R02 | History encoder, BP forecast and survival probability | Four modality encoders → GRU → BP and discrete-hazard heads; as-of collection/availability filtering | Synthetic training with participant separation; no real BP/event model |
| R03 | Prioritized biomarkers | Collection table below, long-format measurement and BP templates | Missing biomarkers remain missing |
| R04 | Four research questions | Association analysis, synthetic state transitions, utility and trial estimators; study protocol | Q1 partly exploratory; Q2–Q4 require new longitudinal/intervention data |
| R05 | Average causal effect | Horvitz–Thompson estimator with participant bootstrap; 12-week synthetic trial | Known injected effect tested; no source-data causal estimate |
| R06 | IPW policy value | Fixed-policy value with positivity validation, effective sample size and maximum weight | Synthetic logged decisions; no learned clinical policy |
| R07 | Separate physiological, psychological, assay and context encoders | `src/research/multimodal.py` with masks, elapsed time and history-only GRU | Synthetic checkpoint `longitudinal_synthetic.pt` |
| R08 | Incremental value vs clinical baseline | Four matched-domain HGB models for each recorded outcome; paired cluster-bootstrap differences | Actual CSV calculations; post hoc reuse of existing test set |
| R09 | ANS/HPA/immune/vascular/behavior state-space | Five-coordinate linear Gaussian filter, missing-observation updates, process and measurement covariance | Invented parameters; HPA and immune coordinates explicitly lack measurement support |
| R10 | Safe action utility | Feasibility-filtered utility argmax, rejecting unsupported benefits | Caller-supplied synthetic assumptions; no medication or treatment recommendation |
| R11 | BP, psychological, event and calibration-related objective | Huber BP/psych losses, acute BCE, censored event likelihood, acute Brier term | Disabled heads for missing labels; Brier loss does not establish clinical calibration |
| R12 | Microrandomized eligible decisions and proximal effects | Known assignment probabilities; participant-cluster uncertainty; prespecified state strata | Synthetic nonoverlapping 12-hour outcomes with no carryover |

## Measurement priorities and source availability

| Priority | Measurements | Status and use |
|---|---|---|
| Essential | Repeated validated cuff BP / ABPM, resting HR, age, medications, smoking, BMI, diabetes | Age, smoking, BMI and diabetes present. Hypertension is a diagnosis flag, not BP. NREM HR is a source column with unverified units/linkage, not a verified resting-HR protocol. Medication data absent. |
| Essential | Validated loneliness, anxiety and perceived-stress scores; sleep/activity | No loneliness instrument or validated anxiety score. `bad_mental_days`, `TENSE/ANXIOUS`, `TIRED` are available proxies. Sleep and activity exist but have no timestamps. Keep instrument name, version, scoring range and collection time for new data. |
| High | ECG or quality-controlled PPG, RMSSD, respiration, activity-adjusted HR | Aggregated RMSSD and respiration columns exist. Raw signals, quality annotations and synchronized activity do not. HRV alone does not establish an arrhythmia, anxiety disorder or structural cardiac abnormality. |
| High | Lipids, HbA1c, renal function | HbA1c is present. Lipids and renal assays absent; a kidney-disease flag is not renal function. |
| Research | hs-CRP, IL-6, timed diurnal cortisol | Absent; do not infer them from HRV, stress score or a latent coordinate. |
| Advanced | ADM, FABP4/proteomics, arterial stiffness, endothelial function | Absent. Reserve for a separately justified and adequately powered protocol. |

`ENTERTAINMENT` is never used as a loneliness surrogate in the extension. The raw `stress_score` direction is unknown. Unitless invented loneliness in the synthetic generator is not a UCLA scale, and invented anxiety is not GAD-7. No copyrighted questionnaire items are reproduced; collect properly administered and scored instruments appropriate to the population.

## Equations and executable scope

**History.** At a prediction landmark `t`, include a measurement only when both its collection timestamp and its availability timestamp are at or before `t`. Late-arriving assays cannot leak into earlier histories. Normalize using training data or a prior enrollment period; retain value masks, units, device, instrument and provenance. A daily average cannot be relabeled a minute-resolution stress signal.

The encoder computes `h_i(t) = f_theta(X_i[0:t], masks, elapsed_time)`. Separate modality encoders avoid treating absent assays as measured zeros. The current neural demonstrator uses equally spaced 21-night histories with artificial missingness and explicit time features. It does not claim robustness to irregular real-device streams. Variable-length histories require an explicit padding/length contract before clinical use.

**BP.** The synthetic BP head predicts the mean of seven invented paired readings in the next week, in SBP/DBP units. Descriptive calculations use:

- Pulse pressure = SBP − DBP.
- Approximate resting MAP = DBP + (SBP − DBP)/3.
- Sleep SBP dip (%) = 100 × (mean awake SBP − mean asleep SBP) / mean awake SBP.

The software requires valid paired readings, validated-device flags, distinct timestamps and observed sleep state. Its minimum of two readings per subgroup is an arithmetic guard, not an ABPM adequacy or diagnostic guideline. Uneven sampling can bias equally weighted means. No sleep-state inference from UTC clock time is performed.

**Survival.** The screenshot's continuous-time proportional-hazards formulation is `lambda_i(u|h) = lambda_0(u) exp(g(h))`; with survival `S`, the conditional event probability is `1 − S(t+Delta)/S(t)`, for survival to `t`. The executed neural alternative uses interval-specific hazards `q_k = sigmoid(logit_k)`, `S_k = product_(j<=k)(1−q_j)` and `P(event by k)=1−S_k`. The loss includes event-free intervals and the event interval, then stops. A censored subject contributes only through fully observed intervals. Censoring inside an interval needs upstream interval handling; the synthetic generator censors at interval ends. No separate Cox model is fitted.

Do not turn a prevalent `heart_disease` flag into an event time, count censored subjects as event-free beyond their follow-up, or call the plotted synthetic curve a patient-specific clinical risk. The demonstration assumes conditionally independent censoring, one event type, no competing risks and no left truncation. A real CVD study must handle death/competing events and adjudication explicitly.

**Multitask loss.** The implemented fixed objective is
`L = L_acute_BCE + L_BP_Huber + 0.2 L_psych_Huber + 0.5 L_event_NLL + 0.1 L_acute_Brier`.
BP targets are standardized using declared synthetic scales before Huber loss. A head contributes only for participants with its label available. Hyperparameters are demonstration choices. Brier is a proper probability loss, not proof of calibration; there is no fitted external calibrator or calibrated individual uncertainty interval for this network.

**State.** `z_(t+1)=A(dt)z_t+B x_t+D a_t+C c_t+epsilon`, `y_t=H z_t+eta`, with `z=[ANS,HPA,immune,vascular,behavior]`. The filter propagates covariance across missing observations and larger gaps and uses the Joseph covariance update. All matrices and noise terms are assumed. HPA/immune have no measurements in the saved example; their values remain prior/model driven and are marked as such. Named coordinates do not establish physiological identifiability. The original twin's smoothed deviation index remains separate from this five-coordinate demonstration.

**Causal effect.** For one baseline assignment per eligible participant, `psi = E[Y(1)−Y(0)]`. The synthetic endpoint is mean SBP in the last week of a 12-week trial. The raw HT estimate is `mean[A Y/p − (1−A)Y/(1−p)]`. A prespecified constant of 130 mmHg is subtracted for a second, variance-reduced version; with correct randomization this has the same expectation, but a different finite-sample value. Both are saved. Intervals resample participants, not decision rows.

**MRT.** `beta(s)=E[Y_(t+Delta)(1)−Y_(t+Delta)(0)|S_t=s,I_t=1]`. The example reports overall and two prespecified state-stratum effects, weighting eligible decisions equally. Participants with more eligible decisions therefore contribute more to the estimand; cluster bootstrap handles dependence but does not change that weighting. The implementation is the restricted case with outcomes before the next decision, no carryover and no future intervention in the proximal window. General excursion effects under treatment feedback require a longitudinal method such as WCLS with the specified future-treatment protocol.

**Policy.** `V_IPW(pi)=mean[1(A=pi(S)) R / p(A|S)]`. The fixed policy is specified before evaluation and is not optimized on test rewards. The reward `130−SBP−0.5*action` is an invented utility. Strict positivity is enforced, and effective sample size plus maximum weights are reported; no silent weight clipping. Unknown propensities, missing eligible outcomes and overlapping windows fail validation. A low effective sample size or uncontrolled confounding would limit interpretation even if arithmetic succeeds.

**Utility.** `a*=argmax_(a in A_safe) E[physiological benefit + psychological benefit − burden − risk | history]`. The executable comparison accepts only caller-marked safe actions with all four finite utility components. It does not estimate treatment response from the CSV. No action is selected when benefit information is absent.

## What the real-data analysis means

The new clinical-context model excludes `bad_mental_days`, `TENSE/ANXIOUS` and `TIRED`. Psychological addition, wearable addition and their combination are compared using identical fixed HGB settings and the existing train/tune/calibration/test partitions. Sleep and activity stay in clinical context: the contrast tests conditional predictive addition and cannot estimate the total effect of psychosocial distress. The earlier project's historical “clinical” model retained `bad_mental_days`; its old and new tables answer different questions.

The descriptive association model adjusts for age, gender, BMI, diabetes and smoking. For recorded heart disease it additionally includes hypertension and the exposure×hypertension interaction. The exposure coefficient is then conditional on hypertension=0; the hypertension coefficient is at the centered exposure mean. The interaction odds ratio is a ratio of odds ratios, not proof of biological synergy. Complete exposure cases are used; missing `TENSE/ANXIOUS` values are not assigned a symptom level. Cluster sandwich intervals and Benjamini–Hochberg q-values are reported for the eight coefficients. These models are exploratory and assume a linear log-odds association for standardized raw proxies.

Residual confounding, reverse causation, outcome timing, unverified source joins and ascertainment can explain associations. Conditioning on hypertension can block part of a hypothesized causal pathway, so this model cannot estimate mediation or the total loneliness effect. No loneliness variable is observed at all. Do not conclude that a nonsignificant raw proxy means loneliness has no effect in people.

## Prospective study specification

1. **Association question:** prospectively collect repeated validated loneliness/anxiety/stress scores, standardized resting BP and HR, quality-controlled HRV, sleep/activity and medication changes. Record site, socioeconomic context, depression, illness and baseline cardiovascular status as potential confounders. Prespecify exposure windows and lagged outcomes; examine reverse direction and missingness.
2. **Transition question:** compare persistent changes with each participant's prior enrollment distribution, retaining signal quality, measurement times and device changes. Define sustained BP elevation using the selected clinical protocol and independent outcome assessment. Use patient/site separation and temporal validation; assess false alerts per patient-time and detection delay.
3. **Personalization question:** begin with a defined, acceptable intervention such as an optional social-contact support program and a specified comparator. Record availability, eligibility, assignment, delivered intervention, adherence, burden and adverse events. Estimate proximal effects before optimizing policy. An MRT is appropriate only with a clear decision schedule and future-treatment protocol.
4. **Benefit question:** use a prespecified randomized 12-week comparison of personalized support versus conventional/nonpersonalized support, alongside usual clinical care. Primary endpoint: protocol-measured final-week mean SBP; analyze baseline adjustment, missingness and intention to treat. Prespecify secondary psychological outcomes and multiplicity. Determine sample size from a justified clinically meaningful difference and variance; no power claim is made here.

Long-term incident hypertension/CVD needs years of follow-up, verified baseline freedom from the endpoint, event adjudication, censoring and competing-risk handling. External validation should assess calibration, discrimination, subgroup performance, net benefit, device/site transportability and burden. Research demonstrations are not deployable clinical monitoring or treatment-selection tools.

## Primary references

- AHA, *Effects of Objective and Perceived Social Isolation on Cardiovascular and Brain Health* (2022), [doi:10.1161/JAHA.122.026493](https://www.ahajournals.org/doi/10.1161/JAHA.122.026493). Motivates the hypothesis; observational evidence and causal uncertainty do not validate this dataset.
- AHA/ACC, [2025 high blood pressure guideline summary](https://professional.heart.org/en/science-news/2025-high-blood-pressure-guideline/top-things-to-know). Supports standardized validated BP measurement; wearable aggregates are not substitutes for measured BP.
- Qian et al., [The Micro-Randomized Trial for Developing Digital Interventions: Data Analysis Methods](https://arxiv.org/abs/2004.10241). Defines causal excursion effects and WCLS; the implemented demo is a restricted no-carryover case.
- Swaminathan & Joachims, [Batch Learning from Logged Bandit Feedback through Counterfactual Risk Minimization](https://jmlr.org/papers/v16/swaminathan15a.html). Supports the propensity-weighting framework and its variance concerns.

These sources support methodological framing, not the synthetic effect sizes or model coefficients.
