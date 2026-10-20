import numpy as np
import pandas as pd
import torch
from src.models.baselines import predict
from src.models.fusion_model import FusionModel
from src.evaluation.metrics import metrics,bootstrap
from src.evaluation.explainability import explain_patient


def test_saved_models_load_predict_and_explain(models,profile):
    for model in models.values():
        p=predict(model,pd.DataFrame([profile]))[0]
        assert np.isfinite(p) and 0<=p<=1
        assert len(explain_patient(model,profile))==len(model["features"])


def test_fusion_all_missing_mask_has_finite_gradients():
    model=FusionModel(4,[2,3])
    out=model(torch.randn(3,4),torch.zeros(3,2,dtype=torch.long),
              torch.zeros(3,14,6),torch.zeros(3,14,6),torch.zeros(3,14,2))
    assert out["state"].shape==(3,64)
    assert torch.isfinite(out["state"]).all()
    out["disease"].sum().backward()
    assert model.disease.weight.grad is not None
    torch.testing.assert_close(out["attention"].sum(1),torch.ones(3))


def test_metric_reference_cases():
    result=metrics([0,0,1,1],[.1,.2,.8,.9])
    assert result["roc_auc"]==1 and result["sensitivity"]==1 and result["specificity"]==1
    assert result["tn"]==2 and result["tp"]==2
    assert metrics([0,0],[.1,.2])["roc_auc"] is None


def test_cluster_bootstrap_reproducible():
    y=np.tile([0,1],20);p=np.tile([.2,.7],20);g=np.repeat(np.arange(20),2)
    assert bootstrap(y,p,g,repeats=20)==bootstrap(y,p,g,repeats=20)
