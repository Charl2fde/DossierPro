"""Tests for the Streamlit dashboard, using Streamlit's headless AppTest runner.

The "Prediction en direct" and "Analyse d'un dataset" tabs are not covered
here: they call the live FastAPI server over HTTP (or require a file
upload), which is out of scope for an automated unit test run without that
dependency. They were manually verified instead (see project notes).
"""

from streamlit.testing.v1 import AppTest


def test_app_loads_without_exception() -> None:
    at = AppTest.from_file("src/dashboard/app.py")
    at.run(timeout=60)
    assert not at.exception


def test_stats_tab_renders_dataset_metrics() -> None:
    at = AppTest.from_file("src/dashboard/app.py")
    at.run(timeout=60)

    stats_tab = at.tabs[2]
    metrics = {m.label: m.value for m in stats_tab.metric}
    assert metrics["Businesses"] == "5"
    assert int(metrics["Total avis"].replace(",", "")) > 0


def test_models_tab_renders_comparison_and_warning() -> None:
    at = AppTest.from_file("src/dashboard/app.py")
    at.run(timeout=60)

    models_tab = at.tabs[3]
    assert len(models_tab.dataframe) == 6  # comparison table + 5 confusion matrices
    assert len(models_tab.warning) == 1
