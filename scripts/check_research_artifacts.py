"""Small executable check of the packaged research model and saved calculations."""
import _bootstrap
import json
from src.common import ROOT
from src.research.inference import synthetic_prediction
from src.research.causal import estimate_ate
import pandas as pd

if __name__=="__main__":
    output=synthetic_prediction(1000)
    assert 50<output["next_week_mean_sbp_mmhg"]<260
    trial=pd.read_csv(ROOT/"data/synthetic/parallel_trial.csv")
    trial["outcome"]-=130
    result=estimate_ate(trial)
    saved=json.loads((ROOT/"outputs/metrics/synthetic_causal_evaluation.json").read_text())
    assert abs(result["estimate"]-saved["parallel_trial_centered"]["estimate"])<1e-9
    print(json.dumps({"research_checkpoint_reload":"passed","saved_trial_recalculation":"passed",
                      "synthetic_prediction":output},indent=2))
