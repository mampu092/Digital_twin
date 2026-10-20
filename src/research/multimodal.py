"""Separate modality encoders, causal history GRU and label-masked prediction heads.

The supplied CSV cannot train these longitudinal heads. The bundled checkpoint
uses entirely synthetic participants, exposures, assays, BP and future outcomes.
"""
import torch
from torch import nn
import torch.nn.functional as F

MODALITIES = {"physiology": 6, "psychology": 3, "assays": 3, "context": 5}


class LongitudinalMultimodal(nn.Module):
    def __init__(self, event_intervals=5):
        super().__init__()
        self.encoders = nn.ModuleDict({name: nn.Sequential(nn.Linear(size*2, 16), nn.GELU())
                                      for name, size in MODALITIES.items()})
        self.temporal = nn.GRU(16*4+2, 48, batch_first=True)
        self.heads = nn.ModuleDict({"acute": nn.Linear(48, 1), "bp": nn.Linear(48, 2),
                                   "psych": nn.Linear(48, 3), "event": nn.Linear(48, event_intervals)})

    def forward(self, values, masks, time):
        encoded = []
        for name in MODALITIES:
            mask = masks[name]
            # Hidden values must not affect the output; mask and value are distinct.
            safe = torch.where(mask.bool(), values[name], torch.zeros_like(values[name]))
            if not torch.isfinite(safe).all():
                raise ValueError("Observed modality values must be finite")
            encoded.append(self.encoders[name](torch.cat([safe, mask], dim=-1)))
        if not torch.isfinite(time).all():
            raise ValueError("Finite elapsed-time inputs required")
        state, _ = self.temporal(torch.cat([*encoded, time], dim=-1))
        history = state[:, -1]
        return {name: head(history).squeeze(-1) if name == "acute" else head(history)
                for name, head in self.heads.items()}


def discrete_survival_loss(logits, observed_bins, event):
    """Right-censor at the END of the last fully observed interval.

    For event=1, the event occurs in observed_bins-1. For event=0 all bins up to
    observed_bins are known event-free. Partial intervals are excluded upstream.
    """
    if logits.ndim != 2 or observed_bins.shape != event.shape or len(event) != len(logits):
        raise ValueError("Invalid survival shapes")
    if not torch.isfinite(observed_bins).all() or not torch.isfinite(event).all():
        raise ValueError("Finite follow-up and event indicators required")
    if not ((event == 0) | (event == 1)).all():
        raise ValueError("Binary event indicator required")
    if not ((observed_bins >= 1) & (observed_bins <= logits.shape[1]) &
            (observed_bins == observed_bins.round())).all():
        raise ValueError("Follow-up must contain 1..K complete/event intervals")
    bins = torch.arange(logits.shape[1], device=logits.device)[None, :]
    at_risk = bins < observed_bins[:, None]
    event_target = ((bins == observed_bins[:, None]-1) & event[:, None].bool()).float()
    return (F.binary_cross_entropy_with_logits(logits, event_target, reduction="none")*at_risk).sum(1).mean()


def cumulative_event_probability(logits):
    return 1-torch.cumprod(1-torch.sigmoid(logits), dim=1)


def multitask_loss(outputs, labels, available):
    """Missing labels are not zero/negative outcomes. All heads can be disabled."""
    components = {}
    for head in ["acute", "bp", "psych", "event"]:
        selected = available[head].bool()
        if not selected.any():
            continue
        if head == "event":
            components[head] = discrete_survival_loss(outputs[head][selected],
                                                      labels["observed_bins"][selected], labels["event"][selected])
        elif head == "acute":
            target = labels[head][selected]
            components[head] = F.binary_cross_entropy_with_logits(outputs[head][selected], target)
            components["brier"] = ((torch.sigmoid(outputs[head][selected])-target)**2).mean()
        else:
            components[head] = F.huber_loss(outputs[head][selected], labels[head][selected], delta=1.0)
    if not components:
        raise ValueError("No observed labels: no training objective")
    weights = {"acute": 1.0, "bp": 1.0, "psych": .2, "event": .5, "brier": .1}
    return sum(weights[name]*loss for name, loss in components.items()), components
