"""Build a privacy-filtered ZIP, verify CRC/hashes, and test it in a fresh directory."""
import _bootstrap
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from src.common import ROOT,dump_json

SKIP_DIRS={".git",".venv","__pycache__",".pytest_cache",".cache","pdf_review"}
REQUIRED=["README.md","LICENSE","requirements.txt","pyproject.toml","environment.yml","Dockerfile",
    "dashboard/app.py","src/twin/digital_twin.py","models/saved/heart_disease_combined.joblib",
    "models/saved/hypertension_combined.joblib","models/saved/fusion.pt","data/synthetic/demo_trajectories.csv",
    "reports/data_audit.html","reports/final_research_report.md","docs/architecture.pdf",
    "outputs/patient_reports/synthetic_patient_report.pdf","outputs/metrics/test_metrics.csv",
    "docs/screenshots/patient_overview.png","docs/screenshots/architecture.png",
    "notebooks/01_exploratory_analysis.ipynb","notebooks/02_model_comparison.ipynb",
    "notebooks/03_digital_twin_demonstration.ipynb",
    "reports/enhancement_audit.md","reports/enhancement_audit.html","docs/research_extension.md",
    "docs/reference_image_manifest.json","models/saved/longitudinal_synthetic.pt",
    "outputs/metrics/psychosocial_domain_metrics.csv","outputs/metrics/synthetic_causal_evaluation.json"]


def eligible(path):
    relative=path.relative_to(ROOT)
    if any(part in SKIP_DIRS for part in relative.parts):return False
    if path.suffix in {".pyc",".log",".sqlite",".zip",".xml"}:return False
    if path.name==".env" or path.name.startswith("private"):return False
    if relative.as_posix().startswith("data/raw/") and path.suffix==".csv":return False
    if path.name=="package_manifest.json":return False
    return True


def package(output=None,verify=False):
    output=Path(output or ROOT.parent/"Cardiovascular_Digital_Twin_Python_Environment.zip").resolve()
    for name in REQUIRED:
        if not (ROOT/name).is_file():
            raise FileNotFoundError(f"Required deliverable missing: {name}")
    files=sorted(p for p in ROOT.rglob("*") if p.is_file() and eligible(p))
    manifest={"files":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "raw_dataset_included":False,"row_level_source_predictions_included":False,
        "source_license":"unverified","model_artifact_redistribution":"requires source-rights review",
        "demo_data":"invented profile and synthetic trajectories",
        "format":"project source, models and environment specifications; no virtual environment"}
    dump_json(manifest,ROOT/"package_manifest.json")
    files.append(ROOT/"package_manifest.json")
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,arcname="Cardiovascular_Digital_Twin/"+p.relative_to(ROOT).as_posix())
    report={"zip":str(output),"bytes":output.stat().st_size,"files":len(files),
        "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"raw_csv_included":False}
    with zipfile.ZipFile(output) as z:
        bad=z.testzip()
        if bad:raise ValueError(f"CRC failure: {bad}")
        report["crc_check"]="passed"
        for rel,expected in manifest["files"].items():
            actual=hashlib.sha256(z.read("Cardiovascular_Digital_Twin/"+rel)).hexdigest()
            if actual!=expected:raise ValueError(f"Hash mismatch: {rel}")
        report["manifest_hash_check"]="passed"
        if verify:
            clean=Path(tempfile.mkdtemp(prefix="cardio_twin_extracted_")).resolve()
            for member in z.infolist():
                target=(clean/member.filename).resolve()
                if not target.is_relative_to(clean):raise ValueError("Unsafe archive member")
            z.extractall(clean)
    if verify:
        extracted=clean/"Cardiovascular_Digital_Twin"
        checks=[["-m","pytest","tests/","-q","--junitxml=outputs/metrics/extracted_tests.xml"],
                ["scripts/run_demo.py"],["scripts/check_research_artifacts.py"]]
        report["extraction_directory"]=str(extracted)
        for args in checks:
            proc=subprocess.run([sys.executable,*args],cwd=extracted,capture_output=True,text=True,encoding="utf-8",errors="replace")
            label="tests" if args[0]=="-m" else ("research" if "research" in args[0] else "demo")
            (ROOT/f"outputs/extracted_{label}.log").write_text(proc.stdout+"\n"+proc.stderr,encoding="utf-8")
            report[f"extracted_{label}_returncode"]=proc.returncode
            if proc.returncode:raise RuntimeError(f"Clean extraction {label} failed; see outputs/extracted_{label}.log")
        suite=ET.parse(extracted/"outputs/metrics/extracted_tests.xml").getroot().find("testsuite")
        report["extracted_test_counts"]={k:suite.attrib[k] for k in ["tests","failures","errors","skipped"]}
        report["environment_validation"]="fresh project extraction using the tested installed Python environment; not a second dependency installation"
    dump_json(report,output.with_suffix(".verification.json"))
    print(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default=None)
    parser.add_argument("--verify",action="store_true",help="Extract to a new temp directory and run all tests plus demo")
    args=parser.parse_args()
    package(args.output,args.verify)
