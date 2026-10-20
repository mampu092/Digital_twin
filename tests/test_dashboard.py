from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_all_dashboard_pages():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"dashboard/app.py"),default_timeout=90)
    app.run()
    assert not app.exception
    pages=["Physiological Monitoring","Cardiovascular Risk Analysis","Digital Twin Simulation",
           "Intervention Explorer","Research and Model Validation","Loneliness and BP Research","Patient Report"]
    for page in pages:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception,f"{page}: {app.exception}"


def test_dashboard_baseline_learning_and_replay():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"dashboard/app.py"),default_timeout=90).run()
    app.sidebar.slider[0].set_value(3).run()
    assert not app.exception
    assert any("Learning baseline" in element.value for element in app.info)
