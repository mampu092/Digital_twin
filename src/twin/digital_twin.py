import copy
import numpy as np
import pandas as pd
from src.twin.patient_state import PatientState
from src.models.anomaly_detector import PatientAnomalyDetector
from src.models.baselines import predict
from src.data.loader import SENSORS
from src.data.simulation import BOUNDS
from src.evaluation.explainability import explain_patient,data_quality
from src.twin.intervention_engine import simulate_intervention


class CardiovascularDigitalTwin:
    def __init__(self,patient_profile,models,storage_path=None):
        if "patient_id" not in patient_profile:
            raise ValueError("patient_id required")
        self.state=PatientState(copy.deepcopy(patient_profile))
        self.models=models
        self.storage_path=storage_path
        self.detector=PatientAnomalyDetector()

    def ingest_clinical_data(self,ehr_record):
        if str(ehr_record.get("patient_id",self.state.profile["patient_id"]))!=str(self.state.profile["patient_id"]):
            raise ValueError("Patient identity mismatch")
        self.state.profile.update(copy.deepcopy(ehr_record))
        self._save()

    def ingest_wearable_data(self,sensor_record):
        row=copy.deepcopy(sensor_record)
        if str(row.get("patient_id"))!=str(self.state.profile["patient_id"]):
            raise ValueError("Patient identity mismatch")
        if row.get("provenance")!="synthetic":
            raise ValueError("This prototype accepts explicitly synthetic sensors only; real-device units are unverified.")
        time=pd.Timestamp(row.get("timestamp"))
        if pd.isna(time) or time.tzinfo is None:
            raise ValueError("A timezone-aware timestamp is required")
        if self.state.observations and time<=pd.Timestamp(self.state.observations[-1]["timestamp"]):
            raise ValueError("Observations must be strictly chronological; duplicates rejected")
        row["timestamp"]=time.isoformat()
        for c in SENSORS:
            value=row.get(c)
            if value is not None and pd.isna(value):
                value=None
            if value is not None:
                if not isinstance(value,(int,float,np.number)) or not np.isfinite(value) or not BOUNDS[c][0]<=value<=BOUNDS[c][1]:
                    raise ValueError(f"{c}: invalid simulated measurement")
                value=float(value)
            row[c]=value
        # Keep only the input schema; ground truth/scenario metadata cannot influence detector.
        row={k:row[k] for k in ["patient_id","timestamp","provenance",*SENSORS]}
        self.state.observations.append(row)
        return self.update_patient_state()

    def update_patient_state(self):
        if not self.state.observations:
            return {"status":"no_observations"}
        if self.detector.reference is None:
            # Current reading must not set its own reference. Seven enrollment
            # nights are reserved; the first scored observation is night eight.
            self.detector.fit(self.state.observations[:-1])
            self.state.baseline=self.detector.reference
        row=self.state.observations[-1]
        result=self.detect_physiological_anomalies()
        # Idempotent updates: one transition per timestamp.
        if self.state.anomalies and self.state.anomalies[-1]["timestamp"]==row["timestamp"]:
            return self.state.anomalies[-1]
        previous=self.state.latent_state
        if result["score"] is not None:
            self.state.latent_state=.7*previous+.3*result["score"]
            self.state.process_variance=.7**2*self.state.process_variance+.1+.2*(1-result["coverage"])
        else:
            self.state.process_variance+=.2
        result.update(timestamp=row["timestamp"],latent_state=self.state.latent_state,
                      process_sd=float(np.sqrt(self.state.process_variance)),
                      state_interpretation="smoothed deviation index; not event probability")
        self.state.anomalies.append(result)
        self._save()
        return result

    def estimate_baseline_risk(self):
        return {t:dict(score=float(predict(m,pd.DataFrame([self.state.profile]))[0]),
                       label=m["label"],quality=data_quality(m,self.state.profile))
                for t,m in self.models.items()}

    def detect_physiological_anomalies(self):
        return self.detector.score(self.state.observations[-1]) if self.state.observations else {"status":"no_observations"}

    def estimate_current_scenario(self):
        """Apply latest synthetic readings to a copy, preserving the clinical baseline."""
        if not self.state.observations:
            return {"status":"no_observations"}
        snapshot=copy.deepcopy(self.state.profile)
        # Missing latest readings remain missing, rather than silently carrying stale readings.
        snapshot.update({c:self.state.observations[-1].get(c) for c in SENSORS})
        return {t:{"score":float(predict(m,pd.DataFrame([snapshot]))[0]),
                   "scope":"synthetic-input classification scenario; not a change in established diagnosis or event risk",
                   "quality":data_quality(m,snapshot)}
                for t,m in self.models.items()}

    def simulate_intervention(self,scenario):
        result=simulate_intervention(self.state.profile["patient_id"],profile=self.state.profile,
                                    models=self.models,**scenario)
        self.state.interventions.append(result)
        self._save()
        return result

    def explain_prediction(self):
        return {t:explain_patient(m,self.state.profile) for t,m in self.models.items()}

    def export_patient_report(self,path=None):
        from src.reporting.pdf_report import patient_pdf
        return patient_pdf(self,path)

    def _save(self):
        if self.storage_path:
            self.state.save(self.storage_path)

    @classmethod
    def restore(cls,path,patient_id,models):
        state=PatientState.load(path,patient_id)
        twin=cls(state.profile,models,path)
        twin.state=state
        twin.detector.reference=state.baseline
        return twin
