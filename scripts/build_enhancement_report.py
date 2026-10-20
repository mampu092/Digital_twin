"""Build a result-backed audit, standalone HTML readout and reproducible companion."""
import _bootstrap
import html
import json
import xml.etree.ElementTree as ET
import pandas as pd
from src.common import ROOT, dump_json


def read(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8"))


def table(frame):
    cells=[[str(c) for c in frame.columns]]
    cells.extend([[f"{v:.6f}" if isinstance(v,float) else str(v) for v in row] for row in frame.to_numpy()])
    return "\n".join(["| "+" | ".join(cells[0])+" |","| "+" | ".join(["---"]*len(cells[0]))+" |"]+
                     ["| "+" | ".join(row)+" |" for row in cells[1:]])


def build():
    domains=pd.read_csv(ROOT/"outputs/metrics/psychosocial_domain_metrics.csv")
    associations=pd.read_csv(ROOT/"outputs/metrics/psychosocial_associations.csv")
    comparisons=read("outputs/metrics/psychosocial_ablation.json")
    neural=read("outputs/metrics/longitudinal_synthetic_evaluation.json")
    causal=read("outputs/metrics/synthetic_causal_evaluation.json")
    images=read("docs/reference_image_manifest.json")
    delta_rows=[]
    for target,contrasts in comparisons["contrasts"].items():
        for name,result in contrasts.items():
            delta_rows.append(dict(target=target,contrast=name,delta_auroc=result["roc_auc_delta"],
                                   ci_low=result["delta_ci95"][0][0],ci_high=result["delta_ci95"][0][1],
                                   delta_ap=result["pr_auc_delta"]))
    deltas=pd.DataFrame(delta_rows)
    deltas.to_csv(ROOT/"outputs/metrics/psychosocial_paired_differences.csv",index=False)
    tests={}
    path=ROOT/"outputs/metrics/final_tests.xml"
    if path.exists():
        suite=ET.parse(path).getroot().find("testsuite")
        tests={k:int(suite.attrib[k]) for k in ["tests","failures","errors","skipped"]}
    findings=[
        ("High","Scientific scope","Loneliness and measured BP are absent; causation is not identifiable",
         "Resolved in presentation and data gates; missing-data requirement remains","reports/research_capabilities.csv"),
        ("Medium","Clinical baseline definition","Historical baseline contained bad_mental_days, obscuring psychological addition",
         "Added explicit domain partitions and matched settings; retained historical results","scripts/analyze_psychosocial.py"),
        ("High","Enrollment timing","Seventh observation previously entered its own scoring reference",
         "Fixed: seven enrollment readings remain unscored; first score on night eight","src/twin/digital_twin.py"),
        ("Medium","Missing-value handling","Legacy GRU trusted callers to zero masked readings",
         "Fixed: both encoder implementations explicitly zero unobserved values","src/models/wearable_encoder.py"),
        ("Medium","Empty neural input","Legacy inference could produce a score for an entirely missing trajectory",
         "Fixed: no-observation inference now returns unavailable","src/models/fusion_training.py"),
        ("Medium","Architecture diagram","An arrow implied experimental fusion updated the live twin state",
         "Removed misleading link; added separate version 0.2 research architecture page","docs/architecture.pdf"),
        ("High","Temporal alignment","Source has no timestamps or availability times",
         "Added as-of contract and regression tests; real-source forecasting remains unavailable","src/research/longitudinal.py"),
        ("High","Censoring and missing targets","Incident-event and multitask training require distinct label handling",
         "Added censored interval likelihood and per-head label masks; exercised synthetically","src/research/multimodal.py"),
        ("High","Causal policy evaluation","Observational classification score changes cannot estimate treatment effects",
         "Added restricted trial estimators with positivity, eligibility, clustering and overlap checks","src/research/causal.py"),
        ("High","Latent physiology","Biological axis names alone cannot identify ANS/HPA/immune mechanisms",
         "Explicitly label synthetic assumed matrices and measurement support per axis","src/research/state_space.py"),
        ("High","Study validity","Previously inspected test set, unknown wearable linkage, no external validation",
         "Unresolved: new comparisons are exploratory; external prospective study required","docs/research_extension.md"),
    ]
    audit=pd.DataFrame(findings,columns=["severity","area","finding","resolution_or_limit","evidence"])
    audit.to_csv(ROOT/"reports/enhancement_findings.csv",index=False)
    ate=causal["parallel_trial_centered"]
    mrt=causal["mrt_proximal_centered"]
    mrt_truth=causal["known_mrt_eligible_mean_effect"]
    mrt_covered=mrt["ci95"][0]<=mrt_truth<=mrt["ci95"][1]
    text=f"""# Enhancement audit — version 0.2

The project now implements the screenshot concepts as explicit calculations, model components, data contracts and research protocols. **Share as a research prototype with caveats. It does not establish loneliness-caused hypertension or cardiovascular damage.**

Reviewed {images['attachment_count']} supplied attachments, {images['unique_binary_images']} distinct binary images and 12 conceptual groups. Duplicates are mapped by content hash. Raw screenshots, including video-call faces, are excluded from the package. Source CSV remains external.

## Real-data findings

The new comparison uses all 82,767 source rows under the original grouped partitions, with 16,546 held out for evaluation. This test set was already examined during the initial build, so the extension is exploratory. The clinical-context set excludes the three psychological proxies and all wearable columns; all four feature sets use the same fixed HGB settings. Preprocessing, calibration and threshold selection use their designated partitions.

{table(domains[['target','domain','roc_auc','pr_auc','brier']])}

Paired profile-cluster bootstrap differences (200 repetitions):

{table(deltas)}

Psychological addition has AUROC and average-precision intervals spanning zero for both recorded targets. This dataset does not demonstrate a reliable incremental predictive gain from these proxies. Sleep and activity remain in clinical context, so the contrast is conditional on them.

Adjusted cross-sectional associations:

{table(associations[['target','exposure','term','n','odds_ratio','ci95_low','ci95_high','q_bh']])}

`bad_mental_days` is associated with recorded heart disease in the specified model. `TENSE/ANXIOUS` does not show a clear association with either target. These statements concern available raw proxies, not diagnosed anxiety or loneliness. For the heart-disease models, the exposure main effect is at hypertension=0, and the interaction is a ratio of odds ratios. Residual confounding, reverse causation and uncertain source linkage remain unresolved.

## Executed synthetic components

The entirely synthetic four-modality network used 800 training, 200 validation and 200 test participants; checkpoint selection used validation loss over {neural['epochs']} epochs.

- Next-week mean SBP MAE: {neural['bp']['sbp_mae_mmhg']:.3f} mmHg; DBP MAE: {neural['bp']['dbp_mae_mmhg']:.3f} mmHg.
- On {neural['bp']['matched_persistence_n']} matched cases with both last BP readings observed, model SBP MAE: {neural['bp']['matched_model_sbp_mae']:.3f}; persistence MAE: {neural['bp']['matched_persistence_sbp_mae']:.3f} mmHg.
- Noisy next-night acute-state AUROC: {neural['acute']['roc_auc']:.3f}; Brier: {neural['acute']['brier']:.3f}. This does not validate minute/hour detection.
- Right-censored event likelihood loss: {neural['survival']['test_likelihood_loss']:.3f}. This is not a clinical CVD accuracy score.
- Five-coordinate Kalman filter and descriptive BP summaries executed on 84 invented days. HPA/immune axes have no assay support.
- Synthetic 12-week trial injected effect: −4.000 mmHg. Centered HT estimate: {ate['estimate']:.3f}, 95% participant-bootstrap interval [{ate['ci95'][0]:.3f}, {ate['ci95'][1]:.3f}] mmHg.
- Restricted proximal-trial effects, two prespecified state strata, fixed-policy IPW value, effective sample size and safe-utility comparison executed. These calculations demonstrate code under invented assumptions, not intervention efficacy.

The synthetic MRT estimate is {mrt['estimate']:.3f} mmHg (95% interval {mrt['ci95'][0]:.3f} to {mrt['ci95'][1]:.3f}); the realized eligible mean injected effect is {mrt_truth:.3f}. The interval {'contains' if mrt_covered else 'does not contain'} that injected mean in this run. A single realization is not a coverage study. The seed and estimator were not selected to make the interval contain the truth; repeated-simulation coverage validation remains outstanding.

The raw, uncentered HT estimate is retained in the JSON to expose finite-sample weight variance. No reported synthetic number is transferred to real patients. The continuous-time Cox formulation from the references is documented; the executed event head uses a discrete-time likelihood instead.

## Audit findings and repairs

{table(audit)}

## Verification

Latest full pytest run: {tests.get('tests','not recorded')} tests, {tests.get('failures','not recorded')} failures, {tests.get('errors','not recorded')} errors, {tests.get('skipped','not recorded')} skipped.
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
"""
    (ROOT/"reports/enhancement_audit.md").write_text(text,encoding="utf-8")
    html_text=f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cardio Twin · Enhancement audit</title><style>
body{{font:16px/1.6 system-ui,sans-serif;background:#f3f7fa;color:#16354b;margin:0}}
main{{max-width:1200px;margin:auto;padding:40px 28px}}h1{{font-size:38px;line-height:1.12}}
h2{{font-size:24px;margin-top:34px}}.eyebrow{{letter-spacing:.15em;color:#067e83;font-weight:700;font-size:12px}}
.lead{{font-size:20px;max-width:950px}}.panel{{background:white;padding:24px;border:1px solid #dae5eb;border-radius:14px;margin:20px 0}}
.warning{{border-left:5px solid #ba7b30;background:#fff5e6;padding:18px}}.tables{{overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #dae5eb;padding:10px;text-align:left}}
th{{background:#eaf2f5}}a{{color:#047e83}}.muted{{color:#526c7e}}code{{background:#eaf2f5;padding:2px 5px}}</style>
<main><div class="eyebrow">CARDIOVASCULAR DIGITAL TWIN · VERSION 0.2</div>
<h1>Loneliness, anxiety and blood-pressure research</h1>
<p class="lead">An executable research extension with {images['attachment_count']} reference attachments mapped to 12 concepts, matched real-data comparisons, and separate synthetic longitudinal and trial demonstrations.</p>
<div class="warning"><b>The causal chain is still a hypothesis.</b> The source has no measured loneliness, paired BP, timestamps, trial assignments or incident-event follow-up.</div>
<section class="panel"><h2>Actual source: incremental prediction</h2><p>Psychological proxies did not demonstrate a clear gain over clinical context. All intervals for psychological addition span zero. These are exploratory comparisons on a previously inspected test partition.</p>
<div class="tables">{deltas.to_html(index=False,float_format=lambda x:f'{x:.6f}',border=0)}</div></section>
<section class="panel"><h2>Entirely synthetic execution</h2>
<p>Next-week SBP MAE <b>{neural['bp']['sbp_mae_mmhg']:.2f} mmHg</b>; noisy next-night acute AUROC <b>{neural['acute']['roc_auc']:.3f}</b>. These are software demonstration results.</p>
<p>Injected trial effect −4.00 mmHg; centered estimate <b>{ate['estimate']:.2f}</b> (95% interval {ate['ci95'][0]:.2f} to {ate['ci95'][1]:.2f}). No intervention efficacy in real people is established.</p></section>
<section class="panel"><h2>Audit and repairs</h2><div class="tables">{audit.to_html(index=False,border=0)}</div></section>
<section class="panel"><h2>Validation and review files</h2><p>Full automated suite: {tests.get('tests','not recorded')} tests; {tests.get('failures','not recorded')} failures; {tests.get('errors','not recorded')} errors.</p>
<p><a href="enhancement_audit.md">Detailed result-backed audit</a> · <a href="../docs/research_extension.md">Equations and prospective protocol</a> · <a href="../docs/reference_image_manifest.json">Attachment traceability</a></p>
<p class="muted">Launch the local dashboard and select “Loneliness and BP Research” for interactive inspection. The ZIP verification JSON records the final extraction test.</p></section></main></html>"""
    (ROOT/"reports/enhancement_audit.html").write_text(html_text,encoding="utf-8")
    marker="\n## Version 0.2 — loneliness and BP research extension\n"
    readme=ROOT/"README.md"
    original=readme.read_text(encoding="utf-8").split(marker)[0]
    readme.write_text(original+marker+"""
See [enhancement audit](reports/enhancement_audit.md) and [research protocol](docs/research_extension.md).
The dashboard adds **Loneliness and BP Research**. New executable components cover matched psychological/wearable ablation, adjusted associations, measurement availability, BP summaries, five-state filtering, four-modality neural prediction, censored survival and restricted randomized-trial/IPW estimation.

```powershell
python scripts/analyze_psychosocial.py --data "PATH/FINAL_dataset_with_fitbit.csv"
python scripts/run_research_demo.py
python scripts/build_enhancement_report.py
```

All longitudinal and causal demonstrations are explicitly synthetic. No loneliness or BP measurement is manufactured for the original CSV. An unmeasured biological axis remains unsupported, even if the simulator can plot it. The new clinical-context comparison differs from the historical clinical baseline, which included `bad_mental_days`. Both sets of results are retained.
""",encoding="utf-8")
    cells=[{"cell_type":"markdown","metadata":{},"id":"intro","source":[
        "# Loneliness and BP extension — reproducible companion\n",
        "The CLI scripts have been executed. These notebook cells are provided for inspection and rerunning; this notebook itself has not been executed.\n",
        "All longitudinal and causal data here are synthetic. Real-source computations require the separately authorized CSV."]}]
    for i,code in enumerate([
        "from pathlib import Path\nimport json, pandas as pd\nroot = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\npd.read_csv(root/'outputs/metrics/psychosocial_domain_metrics.csv')",
        "pd.read_csv(root/'outputs/metrics/psychosocial_associations.csv')",
        "json.loads((root/'outputs/metrics/longitudinal_synthetic_evaluation.json').read_text())",
        "json.loads((root/'outputs/metrics/synthetic_causal_evaluation.json').read_text())",
        "import sys\nsys.path.insert(0,str(root))\nfrom src.research.longitudinal import bp_summary\nbp = pd.read_csv(root/'data/synthetic/research_bp_readings.csv')\nbp_summary(bp)"
    ]):
        cells.append(dict(cell_type="code",metadata={},id=f"research-{i}",execution_count=None,outputs=[],source=code.splitlines(keepends=True)))
    dump_json({"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},
               "nbformat":4,"nbformat_minor":5},ROOT/"notebooks/04_loneliness_bp_research.ipynb")
    dump_json({"version":"0.2.0","attachments":images["attachment_count"],"unique_images":images["unique_binary_images"],
               "tests":tests,"scope":"Research prototype with limitations; no clinical causal validation",
               "real_data_outputs":["psychosocial_domain_metrics.csv","psychosocial_associations.csv","psychosocial_paired_differences.csv"],
               "synthetic_outputs":["longitudinal_synthetic_evaluation.json","synthetic_causal_evaluation.json","synthetic_bp_summary.json"]},
              ROOT/"outputs/metrics/enhancement_summary.json")
    print("Built enhancement audit, HTML readout, README entry and notebook companion.")


if __name__=="__main__":
    build()
