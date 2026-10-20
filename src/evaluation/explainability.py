import numpy as np
import pandas as pd
from src.models.baselines import predict


def explain_patient(bundle,profile):
    """One-at-a-time reference substitutions; NOT additive SHAP attributions."""
    frame=pd.DataFrame([profile]).reindex(columns=bundle["features"])
    base=float(predict(bundle,frame)[0])
    rows=[]
    for c in bundle["features"]:
        changed=frame.copy()
        changed[c]=bundle["reference"][c]
        rows.append({"feature":c,"score_difference":base-float(predict(bundle,changed)[0]),
                     "method":"one-feature replacement; non-additive and noncausal"})
    return sorted(rows,key=lambda r:abs(r["score_difference"]),reverse=True)


def data_quality(bundle,profile):
    missing=[c for c in bundle["features"] if profile.get(c) is None or pd.isna(profile.get(c))]
    ood=[c for c,(lo,hi) in bundle["bounds"].items()
         if profile.get(c) is not None and not pd.isna(profile.get(c)) and not lo<=profile[c]<=hi]
    for c in bundle["features"]:
        if c not in bundle["bounds"] and profile.get(c) is not None:
            # Learned categorical vocabulary, with unknowns explicitly warned.
            pre=bundle["pipeline"].named_steps["preprocess"]
            cats=pre.transformers_[1][2]
            if c in cats:
                encoder=pre.named_transformers_["category"].named_steps["encode"]
                if profile[c] not in encoder.categories_[cats.index(c)]:
                    ood.append(c)
    return dict(completeness=1-len(missing)/len(bundle["features"]),imputed_fields=missing,
        outside_training_range=ood,uncertainty="No validated individual confidence interval. "
        "Aggregate bootstrap CIs quantify held-out performance only; range checks cannot prove in-distribution status.")
