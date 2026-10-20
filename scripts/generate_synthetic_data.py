import _bootstrap
import argparse
import pandas as pd
from src.common import ROOT,dump_json
from src.data.simulation import simulate,SCENARIOS,UNITS

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Generate clearly marked synthetic nightly demo trajectories.")
    parser.add_argument("--days",type=int,default=30)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    frames=[simulate(scenario=s,days=args.days,seed=args.seed) for s in SCENARIOS]
    pd.concat(frames).to_csv(ROOT/"data/synthetic/demo_trajectories.csv",index=False)
    dump_json({"provenance":"entirely synthetic","units":UNITS,"scenarios":SCENARIOS,
               "seed":args.seed,"days":args.days,"cadence":"one nocturnal summary daily; circadian dynamics not modeled",
               "missing_probability":.08,"autocorrelation":.65,"onset_day_zero_index":10,
               "original_stress_score_direction":"unknown; synthetic convention must not be transferred"},
              ROOT/"data/synthetic/provenance.json")
    print(f"Generated {len(frames)*args.days} synthetic observations.")
