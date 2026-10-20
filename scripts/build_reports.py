import _bootstrap
import json
import platform
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from src.common import ROOT,dump_json


def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))


def markdown_table(frame):
    columns=list(frame.columns)
    rows=["| "+" | ".join(columns)+" |","| "+" | ".join(["---"]*len(columns))+" |"]
    for _,r in frame.iterrows():
        rows.append("| "+" | ".join(f"{v:.4f}" if isinstance(v,float) else str(v) for v in r)+" |")
    return "\n".join(rows)


def write(name,text):
    (ROOT/name).write_text(text.strip()+"\n",encoding="utf-8")


def notebook(path,title,cells):
    result=[{"cell_type":"markdown","metadata":{},"source":[f"# {title}\n","Inspectable companion to executable source scripts; outputs are generated locally."]}]
    for code in cells:
        result.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code.splitlines(keepends=True)})
    for index,cell in enumerate(result):
        cell["id"]=f"cell-{index:02d}"
    dump_json({"cells":result,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
              "language_info":{"name":"python","version":platform.python_version()}},"nbformat":4,"nbformat_minor":5},ROOT/path)


def build():
    audit=read("reports/audit_summary.json")
    real=pd.read_csv(ROOT/"outputs/metrics/test_metrics.csv")
    selection=pd.read_csv(ROOT/"outputs/metrics/model_selection.csv")
    synthetic=read("outputs/metrics/synthetic_evaluation.json")
    fusion=read("outputs/metrics/fusion_evaluation.json")
    cis=read("outputs/metrics/bootstrap_ci95.json")
    ablation=read("outputs/metrics/ablation.json")
    matched=read("outputs/metrics/matched_family_ablation.json")
    splits=read("outputs/metrics/split_counts.json")
    tests={}
    for name in ["core_tests","dashboard_tests","final_tests","extracted_tests"]:
        p=ROOT/f"outputs/metrics/{name}.xml"
        if p.exists():
            suite=ET.parse(p).getroot()
            if suite.tag=="testsuites":suite=suite.find("testsuite")
            tests[name]={k:suite.attrib.get(k) for k in ["tests","failures","errors","skipped","time"]}
    dump_json({"runs":tests,"environment":{"python":platform.python_version(),"system":platform.system()},
               "not_tested":["macOS execution","Linux/Docker execution","CUDA execution","Python 3.11 execution"]},
               ROOT/"outputs/metrics/test_summary.json")
    columns=["target","features","model","roc_auc","pr_auc","sensitivity","specificity","precision","f1","brier"]
    table=markdown_table(real[columns])
    uncertainty=[]
    for _,r in real.iterrows():
        ci=cis[r.target+"_"+r.features]
        uncertainty.append(f"- {r.target}, {r.features}: ROC-AUC {r.roc_auc:.4f} "
            f"(95% CI {ci['roc_auc'][0]:.4f}-{ci['roc_auc'][1]:.4f}); "
            f"average precision {r.pr_auc:.4f} (95% CI {ci['pr_auc'][0]:.4f}-{ci['pr_auc'][1]:.4f}).")
    match_lines=[]
    for target,item in matched.items():
        diff=item["paired_difference"]
        match_lines.append(f"- {target}, fixed {item['fixed_family']}: combined-minus-clinical ROC-AUC "
            f"{diff['roc_auc_delta']:+.4f}, 95% CI {diff['delta_ci95'][0][0]:+.4f} to {diff['delta_ci95'][0][1]:+.4f}; "
            f"AP {diff['pr_auc_delta']:+.4f}, 95% CI {diff['delta_ci95'][1][0]:+.4f} to {diff['delta_ci95'][1][1]:+.4f}.")
    write("reports/model_comparison.md",f"""# Model comparison

Five successfully trained families: logistic regression, random forest,
HistGradientBoosting, multilayer perceptron, and LightGBM. Each was trained for
both targets with clinical-only and combined features (20 searches, two candidates
and two grouped CV folds each). Only the four selected calibrated pipelines are
retained as deployable baseline artifacts. The best fusion checkpoint is separate.

## Held-out test results

{table}

PR-AUC is implemented as average precision, not trapezoidal interpolation.
Every reported score is for recorded disease status, not a future event.
Test prevalence: heart disease {real[real.target.eq('heart_disease')].iloc[0].prevalence:.2%};
hypertension {real[real.target.eq('hypertension')].iloc[0].prevalence:.2%}.
Thresholds optimize tuning balanced accuracy, which creates substantial false-positive
burden and low precision at this prevalence. They are not clinically approved cutoffs.

## Uncertainty

{chr(10).join(uncertainty)}

Intervals use 200 percentile bootstrap resamples of patient/profile clusters,
conditional on the fitted model and fixed threshold. They do not capture training
variation, source uncertainty, donor reuse or transportability. Individual
prediction confidence intervals are not available.

## Ablation

Separately optimized pipelines give the results in `outputs/metrics/ablation.json`.
Their choice of algorithm differs and can confound a feature-only comparison.
The secondary matched-family analysis holds algorithm and hyperparameters fixed:

{chr(10).join(match_lines)}

No reliable clinical benefit follows from these small internal differences. The
source-linkage and semantic concerns remain even when an interval excludes zero.
See source_signature_holdout.csv for a separate 43-signature sensitivity analysis.
That secondary analysis reuses primary hyperparameters; it is not an external or
nested donor-validation result. No labels were fabricated for these tests.

## Tuning results

{markdown_table(selection[['target','features','model','validation_pr_auc','training_cv_pr_auc']])}

## Calibration and explainability

Independent sigmoid calibration follows model selection. Calibration plots and
Brier scores are saved for every selected feature-set/target pair.
Decision curves use illustrative probability thresholds and recorded labels; no
clinical net-benefit claim is warranted. Permutation importance uses 1,500 held-out
rows and three repeats without retuning. Tree SHAP uses 150 rows where supported.
SHAP explains raw tree output, not calibrated probability. Patient explanations
use one-at-a-time training-reference replacement and are non-additive/noncausal.
""")
    write("reports/clinical_validation.md",f"""# Validation findings and clinical status

## Track A - Cross-sectional classification

{table}

Actual partitions: {splits}. Test patients remain untouched by preprocessing,
model selection, calibration and operating-threshold choice. Exact clinical
predictor profiles and repeated patient IDs are grouped. Unknown source donor
overlap cannot be eliminated without source metadata.

Additional evidence: bootstrap_ci95.json; subgroup_metrics.csv (age/gender, at least
100 rows and 20 positives); missingness_sensitivity.csv; train_test_distribution.csv;
source_signature_holdout.csv; matched_family_ablation.json.
Single-split evaluation and small hyperparameter search limit precision and stability.

## Track B - Synthetic temporal evaluation

{synthetic['patients']} independent synthetic patients, {synthetic['evaluated_observations']}
scored observations after enrollment, coverage {synthetic['evaluation_coverage']:.2%}.
ROC-AUC {synthetic['metrics']['roc_auc']:.4f}; sensitivity
{synthetic['metrics']['sensitivity']:.4f}; specificity {synthetic['metrics']['specificity']:.4f};
F1 {synthetic['metrics']['f1']:.4f}.
The fixed index threshold is 2.5. Bootstrap resampling uses whole synthetic patients.
The monotonic index-to-[0,1] mapping is not a calibrated risk probability.
The synthetic positive prevalence is designed and unrelated to clinical prevalence.

The fusion network was trained on {fusion['partition_sizes']['train']} real static
rows paired with independent synthetic sequences; tune {fusion['partition_sizes']['tune']},
test {fusion['partition_sizes']['test']}. Its disease-head ROC-AUC is
{fusion['classification']['roc_auc']:.4f} on this subset; synthetic sequence-anomaly
ROC-AUC is {fusion['synthetic_sequence_anomaly']['roc_auc']:.6f}.
Near-perfect sequence discrimination reflects strong engineered shifts across long
windows and an easier sequence-level label. It is not comparable to clinical event
prediction or to per-observation anomaly detection. The disease head is uncalibrated.

## Track C - System behavior

Test runs (counts are parsed from actual JUnit outputs): {json.dumps(tests)}.
The suite covers training-only imputation, grouping, artifact inference, neural
gradients, missing sensors, invalid chronology/identity/ranges, idempotent state
updates, SQLite round trips, intervention isolation, PDF content and all seven UI pages.
PDF rendering and live-browser visual inspection complement automated assertions.

## Not established

No external, prospective, interventional, multi-site or real wearable temporal
validation has been performed. No medical-device certification, clinical utility,
causal effect, real-world alert burden or individual uncertainty guarantee is claimed.
Python 3.12 on Windows CPU was executed. Python 3.11, Linux/macOS, Docker and CUDA
are documented paths but have not been run in this session.
""")
    write("docs/model_card.md",f"""# Model card

Intended use: local research and competition demonstration of recorded-status
classification, synthetic anomaly detection and hypothetical input scenarios.
Prohibited interpretation: imminent-event prediction, clinical diagnosis, causal
treatment benefit, medication selection or autonomous patient triage.

Primary target: heart_disease; secondary target: hypertension. Binary recorded
labels lack timing. Features exclude both targets, patient_id and bmi_gap.
Median/mode imputation, numeric indicators, standardization and one-hot encoding
are fit in training pipelines. Class weighting is used for logistic, random forest,
HistGradientBoosting and LightGBM. MLP uses its native unweighted loss with early
stopping; AP selection and thresholding address its imbalance tradeoff.

{table}

See reports/model_comparison.md for intervals, threshold tradeoffs and matched
ablation. Stored artifacts: four calibrated baseline bundles, fusion.pt,
fusion_transform.joblib. Only load trusted joblib files; they can execute code.

Limitations: unknown label timing, observational confounding, low-cardinality
wearables, unverified patient linkage, missingness, unknown scales, no external
cohort, limited gender representation, single split and modest search budget.
Training references and min/max checks are not a full multivariate OOD detector.
Data-rights verification remains required before publishing weights.
""")
    write("reports/final_research_report.md",f"""# Cardiovascular Digital Twin - executed research report

Built for the requested Happiest Health Digital Twin Challenge demonstration.
Team and institutional attribution remain user-editable placeholders.

## Objective and evidence

Integrate static clinical context, wearable-associated variables and changing
physiological observations while keeping associations, synthetic deviations and
clinical outcome evidence distinct. All {audit['rows']:,} rows of the source CSV
were audited; actual width is {audit['columns']}. No genuine longitudinal source
data or future-event labels were available.

## Major data-quality findings

No missing patient IDs, repeated IDs or exact duplicate full rows. There are
{audit['duplicate_clinical_profiles']} duplicate clinical profiles under the audit
definition, and {audit['complete_cases']:,} complete cases. SpO2 is 60.14% missing;
stress/sleep scores 47.99%; NREM HR/RMSSD/respiration 39.27%.
Only 43 NREM HR/RMSSD signatures are observed. No physiological or score unit
was certified from names alone. Source provenance and license were not supplied.
The raw CSV is omitted from the distribution.

## Methods and actual classification results

{table}

Both targets are recorded-status classification. Train-only preprocessing,
patient/profile splits, grouped randomized CV, independent calibration and untouched
test evaluation implement the real-data track. Balanced-accuracy thresholding gives
high sensitivity at low precision; a clinical alert policy has not been validated.

## Wearable incremental value

{chr(10).join(match_lines)}

The small internal changes do not establish prospective or clinical value. A
source-signature holdout, missingness-only model and complete-case comparison
investigate sensitivity; none repairs absent provenance.

## Synthetic temporal and fusion experiments

The robust patient-relative detector uses first-seven-night medians and MAD scales,
with minimum noise floors, then scores the two largest absolute standardized
deviations. Synthetic scenarios create correlated multivariate shifts, autoregressive
noise, random missingness and varied baselines.
Independent synthetic test performance: AUC {synthetic['metrics']['roc_auc']:.4f},
F1 {synthetic['metrics']['f1']:.4f}, specificity {synthetic['metrics']['specificity']:.4f}.
These quantify recovery of injected perturbations, not cardiac events.

The EHR embedding + GRU/attention + gated fusion model executes on CPU, supports
CUDA when available, and has two heads. Ten training epochs were run with an
independent tuning subset; the checkpoint was selected by tuning loss. High
synthetic sequence AUC ({fusion['synthetic_sequence_anomaly']['roc_auc']:.6f}) is
expected for strong simulator shifts and is not clinical evidence.

## Working system

The twin validates observation identity, provenance, ranges and strict chronology;
preserves a static profile and personal reference; updates a smoothed state;
tracks missingness and heuristic process uncertainty; persists to SQLite;
executes noncausal scenarios; explains predictions; and exports a patient PDF.
Streamlit provides eight pages and Plotly charts, including the version 0.2 loneliness and BP research extension. Screenshots are captured from
the real running application using an invented patient.

## Interpretation and next research steps

Biological hypotheses are in docs/clinical_hypotheses.md with primary NHLBI sources.
Neither attention, SHAP nor reference substitutions identify causal pathways.
First verify source cohorts, consent/license, source donor IDs, device units,
score direction and merge code. Then collect prospective linked sensor/EHR data
with event adjudication and timestamps; pre-register endpoints and horizons;
use patient/donor/time-separated development; test independent sites and devices;
assess calibration, sensitivity, specificity, alert burden and subgroup utility;
validate uncertainty and interventions with appropriate study designs; and obtain
clinical, privacy, security and regulatory review before any patient-care use.

The current system is an executable research prototype. Genuine event forecasting,
causal treatment modeling, secure clinical deployment and external validation
remain unimplemented because the necessary data and evidence are unavailable.

Source and methods receipts: reports/audit_summary.json,
reports/feature_dictionary.csv, outputs/metrics/*.json and *.csv, source scripts,
and three inspectable companion notebooks.
""")
    write("README.md",f"""# Cardio Twin

**Recorded cardiovascular disease classification + synthetic physiological monitoring**

Happiest Health Digital Twin Challenge 2026, Bengaluru - research prototype.
Team: **[TEAM NAME]** | College: **[COLLEGE NAME]** | Video: **[DEMO VIDEO LINK]**

This is a runnable Python application with trained artifacts, a seven-page
Streamlit dashboard, patient-specific state updates, intervention input scenarios,
PDF reports, reproducible scripts and automated tests. It does not predict
imminent cardiac events or claim causal treatment effectiveness.

## Launch the bundled demo

Tested: Python {platform.python_version()}, Windows, CPU. Python 3.11 is allowed by
the package metadata but was not tested here. Use Python 3.12 to match execution.
No proprietary API, account, paid service or raw patient data is required for the demo.

Windows PowerShell, from this project directory:

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe scripts/run_demo.py
.\\.venv\\Scripts\\python.exe -m streamlit run dashboard/app.py
```

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_demo.py
python -m streamlit run dashboard/app.py
```

Open http://127.0.0.1:8501. If LightGBM needs OpenMP on macOS, install the system
OpenMP runtime. Docker installs libgomp1 on Linux:
`docker build -t cardio-twin .` then `docker run --rm -p 8501:8501 cardio-twin`.
Docker, macOS, Linux and CUDA paths are provided but not executed in this build.
For GPU use a compatible official PyTorch installation and `--device cuda`;
the included tested environment uses CPU. `environment.yml` is an alternate
Conda environment. `requirements-lock-windows-py312.txt` records the exact tested
transitive package set; the top-level requirements are portable version pins.

## Dataset and permissions

The local audit processed **{audit['rows']:,} rows and {audit['columns']} columns**.
The supplied brief said 33 columns; the CSV has 32. Patient IDs are unique,
timestamps are absent and no incident-event label is available.
Targets are `heart_disease` and `hypertension`, both recorded disease status.
Wearable provenance, units, score direction and patient linkage are unverified.
Missingness is high and repeated wearable values suggest aggregation or reuse.

The source CSV is intentionally **not included** because redistribution permission
was not established. Place an authorized copy at `data/raw/FINAL_dataset_with_fitbit.csv`,
or pass `--data PATH` to each data-dependent script. No real patient records appear
in screenshots or demo reports. Review source terms before publishing model weights.

## Executed results

{table}

Test set: {splits['test']:,} rows. PR-AUC denotes average precision.
All five requested baseline families trained in both feature modes for each target;
the best four calibrated baseline pipelines are saved. Model choice uses tune data,
calibration uses its own partition, and thresholds maximize tune balanced accuracy.
Low positive predictive value means many positive classifications are false
positives at these cutoffs. No operating threshold is clinically validated.

{chr(10).join(match_lines)}

Synthetic per-observation anomaly evaluation: **ROC-AUC
{synthetic['metrics']['roc_auc']:.3f}, sensitivity {synthetic['metrics']['sensitivity']:.3f},
specificity {synthetic['metrics']['specificity']:.3f}** across {synthetic['patients']} simulated
patients. This detects injected perturbations and cannot establish clinical efficacy.
The PyTorch fusion branch executes and is separately labeled as a mixed-source
experiment. Its synthetic sequence result must not be promoted as cardiac prediction.

## Architecture and scientific motivation

Clinical variables are encoded with training-only transformations. The experimental
neural branch combines numerical normalization/categorical embeddings with a
masked GRU, temporal attention and gated fusion. Baseline classifiers and the
patient-relative temporal detector remain separately interpretable.

Stress/autonomic, sleep, metabolic and respiratory mechanisms motivate the project,
but they are untested hypotheses in these data. See
[clinical hypotheses](docs/clinical_hypotheses.md) for primary source references.

![Architecture](docs/screenshots/architecture.png)

[Architecture PDF](docs/architecture.pdf) · [Technical design](docs/architecture.md)

## Dashboard

![Running patient overview](docs/screenshots/patient_overview.png)

Pages: Patient Overview; Physiological Monitoring; Cardiovascular Risk Analysis;
Digital Twin Simulation; Intervention Explorer; Research and Model Validation;
Patient Report. Use the sidebar scenario and replay sliders to show baseline
learning, stable measurements and emerging synthetic deviations.
The intervention explorer displays **Hypothetical model scenario - not a validated
treatment recommendation.** Duration is descriptive, not a learned response horizon.

## Reproduce the entire analysis

Activate the environment and supply the authorized CSV before these commands:

```bash
python scripts/audit_dataset.py --data PATH_TO_CSV
python scripts/train_baselines.py --data PATH_TO_CSV
python scripts/generate_synthetic_data.py
python scripts/train_fusion.py --data PATH_TO_CSV
python scripts/evaluate_models.py --data PATH_TO_CSV
python scripts/matched_ablation.py --data PATH_TO_CSV
python scripts/run_demo.py
python -m pytest tests/ -v --junitxml=outputs/metrics/final_tests.xml
python scripts/build_reports.py
python scripts/package_project.py
```

Set `CARDIO_DATA_PATH` to omit `--data`, except the audit CLI takes its own explicit
default path. Random seed is 42. Available flags are described by each script's
`--help`. YAML files document the tested defaults; supported overrides are CLI flags.
Auditing/tuning/evaluation need the original CSV. Inference, dashboard, tests and
the synthetic demonstration use bundled artifacts and run without it.

## Inspectable outputs

- `reports/data_audit.html`: complete schema, completeness and label audit.
- `reports/model_comparison.md`: actual comparisons, CIs, calibration and ablations.
- `reports/clinical_validation.md`: three validation tracks and exact limitations.
- `reports/final_research_report.md`: research narrative and future study requirements.
- `outputs/metrics/`: machine-readable results, bootstrap CIs, subgroup and sensitivity analyses.
- `outputs/patient_reports/synthetic_patient_report.pdf`: invented-patient example.
- `docs/demo_script.md`: detailed 24-minute presentation guide.
- `notebooks/`: EDA, model comparison and twin companions, with executable source cells.
- `docs/model_card.md`, `docs/data_card.md`, `docs/ethics_and_privacy.md`.

Tests are recorded in `outputs/metrics/test_summary.json`; the final delivery also
checks ZIP CRC integrity and clean extraction. The tests cover chronology, invalid
data, state persistence, leakage guards, models, simulation, reporting and all UI pages.
Notebook cells are provided for inspection and reruns, not claimed as executed notebooks.

## Limitations and license

This is not a certified medical device. No real temporal outcomes, independent
external validation, causal interventions or clinically calibrated individual
uncertainty exist. The source sensor linkage and scales remain unknown. Classifiers
identify associations with existing labels. Synthetic detection results reflect the
generator. SQLite and the local UI do not implement secure clinical deployment.
Only trained artifacts you trust should be loaded.

Original software is MIT licensed; source dataset rights and trained-weight
redistribution terms are not verified. See LICENSE and data/raw/README.md.
The package is prepared for review; no public repository has been published.
""")
    root_cell="from pathlib import Path\nimport sys\nroot = Path.cwd()\nif root.name == 'notebooks': root = root.parent\nsys.path.insert(0, str(root))\nimport pandas as pd\n"
    notebook("notebooks/01_exploratory_analysis.ipynb","Dataset audit and provenance",[
        root_cell,"pd.read_csv(root / 'reports/feature_dictionary.csv')\n",
        "pd.read_csv(root / 'reports/missingness_by_subgroup.csv')\n",
        "# Authorized source only: define a local path to re-run the audit.\n# from src.data.loader import load_dataset\n# from src.data.validation import audit_dataset\n# df = load_dataset('YOUR_AUTHORIZED_CSV.csv')\n# audit_dataset(df, Path('YOUR_AUTHORIZED_CSV.csv'))\n",
        "pd.read_csv(root / 'reports/wearable_label_associations.csv')\n"])
    notebook("notebooks/02_model_comparison.ipynb","Classification and synthetic evidence",[
        root_cell,"pd.read_csv(root / 'outputs/metrics/test_metrics.csv')\n",
        "import json\njson.loads((root / 'outputs/metrics/matched_family_ablation.json').read_text())\n",
        "from IPython.display import Image, display\ndisplay(Image(filename=str(root / 'outputs/figures/heart_disease_combined_validation.png')))\n",
        "pd.read_csv(root / 'outputs/metrics/source_signature_holdout.csv')\n"])
    notebook("notebooks/03_digital_twin_demonstration.ipynb","Synthetic state demonstration",[
        root_cell,"import numpy as np\nfrom src.data.simulation import simulate, demo_profile\nfrom src.models.baselines import load_bundle\nfrom src.twin.digital_twin import CardiovascularDigitalTwin\nprofile=demo_profile()\ntwin=CardiovascularDigitalTwin(profile, {'heart_disease':load_bundle()})\n",
        "trajectory=simulate(scenario='sustained_stress')\nfor record in trajectory.replace({np.nan:None}).to_dict('records'):\n    twin.ingest_wearable_data(record)\npd.DataFrame(twin.state.anomalies)[['timestamp','status','latent_state']]\n",
        "twin.simulate_intervention({'intervention_type':'sleep_regularity','duration_days':14,'assumed_feature_changes':{'sleep_hours':7.5}})\n",
        "twin.export_patient_report(root/'outputs/patient_reports/notebook_synthetic_patient.pdf')\n"])
    print("README, research reports, model card and three companion notebooks generated from saved results.")


if __name__=="__main__":
    build()
