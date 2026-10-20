# 24-minute competition demonstration

Team: [TEAM NAME] | College: [COLLEGE] | Video: [DEMO VIDEO LINK]

**0:00-2:00 - Problem and scope.** Describe preventive monitoring and the need to
integrate clinical context with change within a person. State that this prototype
classifies recorded diagnoses and demonstrates synthetic physiological deviations.
It has no validated short-term event forecast.

**2:00-4:00 - Biological rationale.** Open clinical_hypotheses.md. Explain the
stress/autonomic, sleep, metabolic and respiratory hypotheses. Identify mechanisms
the dataset does not measure. Explicitly state that original device units and
stress-score direction have not been verified.

**4:00-6:30 - Dataset audit.** Open reports/data_audit.html. Show 82,767 records,
32 fields, unique patient IDs, missingness, label imbalance, and low wearable
cardinality. Explain why those findings constrain claims and why the CSV is omitted
from the distribution. Do not scroll private row-level data.

**6:30-8:00 - Architecture.** Show docs/architecture.pdf. Distinguish the real-data
classification track, synthetic temporal track, and system behavior tests.
Describe train/tune/calibration/test isolation and duplicate-profile grouping.

**8:00-10:30 - Actual results.** Open Research and Model Validation. Read results
from the displayed saved outputs, not a memorized metric. Compare clinical-only
and combined models, confidence intervals, calibration, and class prevalence.
Show source-signature holdout and missingness sensitivity. Discuss the lack of
external validation even if internal AUC is high.

**10:30-13:00 - Patient overview and replay.** Open Patient Overview. Identify the
invented patient. Set replay to 3 to show baseline learning, then 7 and 10. Explain
that baseline classification remains distinct from physiological state. Point to
completeness and last update.

**13:00-15:30 - Anomaly scenarios.** Choose Sustained Stress; move replay to 18,
then 30. Open Physiological Monitoring. Show NREM HR, RMSSD, sleep and respiration.
Explain fixed enrollment bands and multisensor deviation markers. Select Stable
to demonstrate ordinary variation. No cardiac-event diagnosis is inferred.

**15:30-17:00 - Fusion experiment.** Open Digital Twin Simulation and evaluate the
experimental network. Explain that the GRU trained on synthetic sequences paired
independently with real static labels. Its excellent synthetic separation measures
the simulator's perturbations, not disease detection from real sensor streams.
The state uncertainty band is heuristic and has not been clinically calibrated.

**17:00-19:00 - Explainability.** Open Cardiovascular Risk Analysis. Explain top
reference substitutions and their non-additive interpretation. Compare clinical
and combined scores. Point to held-out calibration and OOD/missingness warnings.
State that SHAP, feature importance and attention do not establish causality.

**19:00-21:00 - Intervention explorer.** Change assumed sleep, activity and recovery.
Read the hypothetical-scenario label. A score decrease does not prove treatment
effectiveness; an increase is also a legitimate model output. Do not interpret
the original stress score as clinical stress without its codebook.

**21:00-22:00 - Patient PDF.** Open Patient Report, download and inspect it. Show
the synthetic trajectory summary, classification outputs, explanations and
limitations. The report contains no real patient identity.

**22:00-24:00 - Validation and next steps.** Show test results and reproducible
commands in README. Discuss source verification, prospective device-linked
cohorts, validated event timestamps, multi-site external validation, calibration
and alert burden, clinical review, consent, security and regulatory assessment.
Finish with the demonstrated capability: reproducible state updates with explicit
evidence boundaries, ready for further research.
