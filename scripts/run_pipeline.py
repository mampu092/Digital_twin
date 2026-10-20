"""Run the reproducible stages; source rights and clinical validity are not assumed."""
import _bootstrap
import argparse
import subprocess
import sys
from src.common import ROOT

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,help="Authorized source CSV")
    parser.add_argument("--skip-training",action="store_true",help="Reuse compatible source-specific artifacts and private split manifest")
    args=parser.parse_args()
    stages=[["scripts/audit_dataset.py","--data",args.data]]
    if not args.skip_training:
        stages += [["scripts/train_baselines.py","--data",args.data]]
    stages += [["scripts/generate_synthetic_data.py"]]
    if not args.skip_training:
        stages += [["scripts/train_fusion.py","--data",args.data]]
    stages += [["scripts/evaluate_models.py","--data",args.data],
               ["scripts/matched_ablation.py","--data",args.data],
               ["scripts/analyze_psychosocial.py","--data",args.data],
               ["scripts/run_research_demo.py"],
               ["scripts/run_demo.py"],
               ["-m","pytest","tests/","-q","--junitxml=outputs/metrics/final_tests.xml"],
               ["scripts/build_reports.py"],
               ["scripts/build_enhancement_report.py"]]
    for stage in stages:
        subprocess.run([sys.executable,*stage],cwd=ROOT,check=True)
