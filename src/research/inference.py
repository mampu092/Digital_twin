"""Reload and exercise the saved synthetic checkpoint without training."""
import numpy as np
import torch
from src.common import ROOT
from src.research.synthetic_cohort import cohort
from src.research.multimodal import LongitudinalMultimodal, cumulative_event_probability


def synthetic_prediction(participant=1000):
    if not isinstance(participant,int) or not 1000<=participant<1200:
        raise ValueError("Choose an invented held-out participant from 1000 to 1199")
    checkpoint=torch.load(ROOT/"models/saved/longitudinal_synthetic.pt",map_location="cpu",weights_only=True)
    if checkpoint["provenance"]!="synthetic_only":
        raise ValueError("Unexpected model provenance")
    model=LongitudinalMultimodal(checkpoint["event_intervals"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    values,masks,time,_,_=cohort()
    ix=slice(participant,participant+1)
    with torch.no_grad():
        output=model({k:v[ix] for k,v in values.items()},{k:v[ix] for k,v in masks.items()},time[ix])
    bp=output["bp"][0].numpy()*[15,10]+[120,75]
    return {"participant":f"SYN-LONG-{participant}","provenance":"ENTIRELY SYNTHETIC",
            "next_week_mean_sbp_mmhg":float(bp[0]),"next_week_mean_dbp_mmhg":float(bp[1]),
            "next_night_acute_state_score":float(torch.sigmoid(output["acute"])[0]),
            "cumulative_event_probability_by_illustrative_year":cumulative_event_probability(output["event"])[0].tolist(),
            "modality_observed_fraction":{k:float(mask[ix].mean()) for k,mask in masks.items()},
            "scope":"Saved synthetic model demonstration; no real patient forecast or treatment recommendation"}
