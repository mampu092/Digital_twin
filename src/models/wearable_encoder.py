import torch
from torch import nn


class WearableEncoder(nn.Module):
    def __init__(self,sensors=6):
        super().__init__()
        self.gru=nn.GRU(sensors*2+2,64,batch_first=True)
        self.attention=nn.Linear(64,1)

    def forward(self,values,mask,time):
        values=torch.where(mask.bool(),values,torch.zeros_like(values))
        hidden,_=self.gru(torch.cat([values,mask,time],dim=-1))
        # Avoid attending all-missing time steps when any observed step exists.
        valid=mask.sum(-1)>0
        valid=torch.where(valid.any(dim=1,keepdim=True),valid,torch.ones_like(valid))
        weights=torch.softmax(self.attention(hidden).squeeze(-1).masked_fill(~valid,-1e4),dim=1)
        return (hidden*weights.unsqueeze(-1)).sum(1),weights
