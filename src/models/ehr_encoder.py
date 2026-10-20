import numpy as np
import pandas as pd
import torch
from torch import nn
from src.data.loader import CATEGORICAL


class StaticTransform:
    def fit(self,frame):
        self.cats=[c for c in frame if c in CATEGORICAL]
        self.nums=[c for c in frame if c not in self.cats]
        self.vocab={c:{v:i+1 for i,v in enumerate(sorted(frame[c].dropna().astype(str).unique()))} for c in self.cats}
        self.medians=frame[self.nums].median().fillna(0)
        self.mean=frame[self.nums].fillna(self.medians).mean()
        self.std=frame[self.nums].fillna(self.medians).std().replace(0,1).fillna(1)
        return self

    def transform(self,frame):
        frame=frame.reindex(columns=self.nums+self.cats)
        num=((frame[self.nums].fillna(self.medians)-self.mean)/self.std).to_numpy(dtype=np.float32)
        cat=np.column_stack([frame[c].astype(str).map(self.vocab[c]).fillna(0).astype(int) for c in self.cats])
        return torch.tensor(num),torch.tensor(cat,dtype=torch.long)


class EHREncoder(nn.Module):
    def __init__(self,n_numeric,cardinalities):
        super().__init__()
        self.embeddings=nn.ModuleList([nn.Embedding(n+1,min(8,n+1)) for n in cardinalities])
        self.network=nn.Sequential(nn.Linear(n_numeric+sum(min(8,n+1) for n in cardinalities),128),
            nn.LayerNorm(128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,64),nn.GELU())

    def forward(self,numeric,categorical):
        return self.network(torch.cat([numeric]+[e(categorical[:,i]) for i,e in enumerate(self.embeddings)],dim=1))
