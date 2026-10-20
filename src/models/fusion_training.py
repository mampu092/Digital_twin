import copy
import logging
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from src.common import ROOT,dump_json
from src.data.loader import feature_columns,SENSORS
from src.data.simulation import simulate,SCENARIOS,NOISE
from src.models.ehr_encoder import StaticTransform
from src.models.fusion_model import FusionModel
from src.evaluation.metrics import metrics,bootstrap


def encode_trajectory(frame):
    values=frame[SENSORS].to_numpy(dtype=np.float32)
    mask=np.isfinite(values)
    # Baseline is past enrollment data only; normalization never sees future rows.
    centers=np.nanmedian(values[:7],axis=0)
    centers=np.where(np.isfinite(centers),centers,np.array([62,45,96.5,15,7.2,45]))
    scale=np.maximum(np.nanstd(values[:7],axis=0),NOISE)
    scale=np.where(np.isfinite(scale),scale,NOISE)
    normalized=np.nan_to_num((values-centers)/scale).clip(-10,10)
    days=np.arange(len(frame),dtype=np.float32)
    time=np.column_stack([days/30,np.sin(2*np.pi*days/7)]).astype(np.float32)
    return normalized.astype(np.float32),mask.astype(np.float32),time


def make_sensors(n,seed):
    rng=np.random.default_rng(seed)
    arrays=[]
    labels=[]
    for i in range(n):
        # Scenario assignment independent of clinical target; balanced stable/perturbed.
        s="stable" if rng.random()<.5 else rng.choice(SCENARIOS[1:])
        frame=simulate(f"SYNTH-{seed}-{i}",s,days=30,seed=seed*100000+i)
        arrays.append(encode_trajectory(frame))
        labels.append(float(s!="stable"))
    return [torch.tensor(np.stack([a[j] for a in arrays])) for j in range(3)]+[torch.tensor(labels)]


def fit_fusion(df,epochs=10,max_train=4096,device="auto"):
    from src.models.baselines import verify_training_dataset
    verify_training_dataset(df)
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device=torch.device("cuda" if device=="auto" and torch.cuda.is_available() else "cpu" if device=="auto" else device)
    manifest=pd.read_csv(ROOT/"data/processed/private_split_manifest.csv")
    rng=np.random.default_rng(42)
    selection={}
    for name,cap in [("train",max_train),("tune",768),("test",1024)]:
        idx=np.flatnonzero(manifest.split.eq(name))
        selection[name]=rng.choice(idx,min(cap,len(idx)),replace=False)
    cols=feature_columns(df,"heart_disease","clinical")
    transform=StaticTransform().fit(df.iloc[selection["train"]][cols])
    datasets={}
    for i,(name,ix) in enumerate(selection.items()):
        logging.info("Generating independent synthetic temporal branch for fusion %s (%d)",name,len(ix))
        numeric,cat=transform.transform(df.iloc[ix][cols])
        values,mask,time,anomaly=make_sensors(len(ix),101+i)
        disease=torch.tensor(df.iloc[ix].heart_disease.to_numpy(dtype=np.float32))
        datasets[name]=TensorDataset(numeric,cat,values,mask,time,disease,anomaly)
    architecture=dict(n_numeric=len(transform.nums),cardinalities=[len(transform.vocab[c]) for c in transform.cats])
    model=FusionModel(**architecture).to(device)
    pos=float(datasets["train"].tensors[-2].mean())
    disease_loss=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-pos)/pos,device=device))
    anomaly_loss=torch.nn.BCEWithLogitsLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    best_loss=float("inf"); best=None; history=[]
    def evaluate(name):
        model.eval(); losses=[]; truths=[]; probs=[]
        with torch.no_grad():
            for batch in DataLoader(datasets[name],batch_size=256):
                num,cat,v,m,t,y,a=[x.to(device) for x in batch]
                out=model(num,cat,v,m,t)
                losses.append(float(disease_loss(out["disease"],y)+anomaly_loss(out["anomaly"],a)))
                truths.append(torch.stack([y,a],dim=1).cpu().numpy())
                probs.append(torch.sigmoid(torch.stack([out["disease"],out["anomaly"]],dim=1)).cpu().numpy())
        return float(np.mean(losses)),np.concatenate(truths),np.concatenate(probs)
    for epoch in range(epochs):
        model.train(); total=[]
        for batch in DataLoader(datasets["train"],batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(42+epoch)):
            num,cat,v,m,t,y,a=[x.to(device) for x in batch]
            optimizer.zero_grad()
            out=model(num,cat,v,m,t)
            # AdamW supplies explicit parameter regularization.
            loss=disease_loss(out["disease"],y)+anomaly_loss(out["anomaly"],a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step(); total.append(float(loss.detach()))
        val,_,_=evaluate("tune")
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(total)),validation_loss=val))
        logging.info("Fusion epoch %d validation loss %.4f",epoch+1,val)
        if val<best_loss:
            best_loss=val;best=copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    _,truth,prob=evaluate("test")
    report={"scope":"MIXED-SOURCE experiment: real static labels with independent SYNTHETIC sequences",
            "clinical_status":"not validated for real longitudinal use; probabilities uncalibrated",
            "classification":metrics(truth[:,0],prob[:,0]),
            "synthetic_sequence_anomaly":metrics(truth[:,1],prob[:,1]),
            "classification_ci95":bootstrap(truth[:,0],prob[:,0],repeats=100),
            "synthetic_sequence_ci95":bootstrap(truth[:,1],prob[:,1],repeats=100),
            "partition_sizes":{n:len(d) for n,d in datasets.items()},"epochs":epochs,"device":str(device)}
    torch.save({"state_dict":{k:v.cpu() for k,v in best.items()},"architecture":architecture},
               ROOT/"models/saved/fusion.pt")
    joblib.dump(transform,ROOT/"models/saved/fusion_transform.joblib")
    pd.DataFrame(history).to_csv(ROOT/"outputs/metrics/fusion_training_history.csv",index=False)
    dump_json(report,ROOT/"outputs/metrics/fusion_evaluation.json")
    return report


def infer_fusion(profile,trajectory):
    if len(trajectory)<7:
        return {"status":"requires_at_least_7_synthetic_observations"}
    if trajectory[SENSORS].notna().sum().sum()==0:
        return {"status":"insufficient_observed_sensor_data"}
    checkpoint=torch.load(ROOT/"models/saved/fusion.pt",map_location="cpu",weights_only=True)
    model=FusionModel(**checkpoint["architecture"]); model.load_state_dict(checkpoint["state_dict"]); model.eval()
    transform=joblib.load(ROOT/"models/saved/fusion_transform.joblib")
    numeric,cat=transform.transform(pd.DataFrame([profile]))
    v,m,t=encode_trajectory(trajectory)
    with torch.no_grad():
        out=model(numeric,cat,*[torch.tensor(x).unsqueeze(0) for x in [v,m,t]])
    return {"experimental_disease_score":float(torch.sigmoid(out["disease"])[0]),
            "synthetic_anomaly_score":float(torch.sigmoid(out["anomaly"])[0]),
            "scope":"experimental mixed-source network; uncalibrated, not clinical probabilities"}
