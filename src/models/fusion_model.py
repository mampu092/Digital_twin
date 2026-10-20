"""Experimental EHR + synthetic sensor multitask network; no forecasting head."""
import torch
from torch import nn
from src.models.ehr_encoder import EHREncoder
from src.models.wearable_encoder import WearableEncoder


class FusionModel(nn.Module):
    def __init__(self,n_numeric,cardinalities):
        super().__init__()
        self.ehr=EHREncoder(n_numeric,cardinalities)
        self.wearable=WearableEncoder()
        self.gate=nn.Sequential(nn.Linear(128,128),nn.Sigmoid())
        self.fusion=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Dropout(.15))
        self.disease=nn.Linear(64,1)
        self.anomaly=nn.Linear(64,1)

    def forward(self,numeric,categorical,values,mask,time):
        static=self.ehr(numeric,categorical)
        dynamic,attention=self.wearable(values,mask,time)
        joined=torch.cat([static,dynamic],dim=1)
        state=self.fusion(joined*self.gate(joined))
        return {"disease":self.disease(state).squeeze(-1),"anomaly":self.anomaly(state).squeeze(-1),
                "state":state,"attention":attention}
