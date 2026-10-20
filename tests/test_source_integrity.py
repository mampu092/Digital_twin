import json
import pandas as pd
import pytest
import src.models.baselines as baselines


def test_wrong_source_fingerprint_is_rejected(tmp_path,monkeypatch):
    root=tmp_path
    saved=root/"models/saved";saved.mkdir(parents=True)
    (saved/"training_provenance.json").write_text(json.dumps({"source_sha256":"expected","rows":1,"columns":["age"]}))
    monkeypatch.setattr(baselines,"ROOT",root)
    df=pd.DataFrame({"age":[30]});df.attrs["source_sha256"]="wrong"
    with pytest.raises(ValueError,match="differs"):
        baselines.verify_training_dataset(df)
    df.attrs["source_sha256"]="expected"
    baselines.verify_training_dataset(df)
