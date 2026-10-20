from dataclasses import dataclass,field,asdict
import json
import sqlite3
from pathlib import Path


@dataclass
class PatientState:
    profile:dict
    observations:list=field(default_factory=list)
    anomalies:list=field(default_factory=list)
    interventions:list=field(default_factory=list)
    latent_state:float=0.0
    process_variance:float=1.0
    baseline:dict|None=None

    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS twin_state (patient_id TEXT PRIMARY KEY,payload TEXT NOT NULL)")
            db.execute("INSERT OR REPLACE INTO twin_state VALUES (?,?)",
                       (str(self.profile["patient_id"]),json.dumps(asdict(self),allow_nan=False)))

    @classmethod
    def load(cls,path,patient_id):
        with sqlite3.connect(path) as db:
            row=db.execute("SELECT payload FROM twin_state WHERE patient_id=?",(str(patient_id),)).fetchone()
        if row is None:
            raise KeyError(patient_id)
        return cls(**json.loads(row[0]))
