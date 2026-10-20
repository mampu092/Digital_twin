import pytest
from src.data.simulation import demo_profile
from src.models.baselines import load_bundle

@pytest.fixture
def profile():
    return demo_profile()

@pytest.fixture
def models():
    return {t:load_bundle(t) for t in ["heart_disease","hypertension"]}
