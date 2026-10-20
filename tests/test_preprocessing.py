import numpy as np
import pandas as pd
from src.data.preprocessing import make_preprocessor
from src.data.loader import feature_columns
from src.models.baselines import split_dataset,profile_groups


def test_imputer_uses_training_only():
    train=pd.DataFrame({"age":[20.,40.,np.nan],"gender":["Female","Male","Female"]})
    pp=make_preprocessor(list(train)).fit(train)
    before=pp.named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
    pp.transform(pd.DataFrame({"age":[1000.,np.nan],"gender":["unknown","Female"]}))
    assert before[0]==30
    np.testing.assert_array_equal(before,pp.named_transformers_["numeric"].named_steps["impute"].statistics_)


def test_target_and_identifier_excluded(profile):
    df=pd.DataFrame([profile])
    for target in ["heart_disease","hypertension"]:
        assert not {"patient_id","heart_disease","hypertension","bmi_gap"}&set(feature_columns(df,target))


def test_groups_keep_conflicting_labels_and_duplicate_ids_together():
    rng=np.random.default_rng(1)
    df=pd.DataFrame({"patient_id":np.arange(200),"age":rng.uniform(20,80,200),"gender":["Female","Male"]*100,
                     "heart_disease":np.tile([0,0,0,1],50),"hypertension":np.tile([0,1],100)})
    df.loc[1,["age","gender"]]=df.loc[0,["age","gender"]]
    df.loc[2,"patient_id"]=df.loc[1,"patient_id"]
    split,groups=split_dataset(df)
    assert groups[0]==groups[1]==groups[2]
    assert split[0]==split[1]==split[2]
    assert set(split)=={"train","tune","calibration","test"}
