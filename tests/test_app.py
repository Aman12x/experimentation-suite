"""
End-to-end tests for the Streamlit app, driven headlessly with AppTest
"""

import os
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def load_app(sample):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox[0].select(sample).run()
    assert not at.exception
    return at


def select(at, label, value):
    next(s for s in at.selectbox if s.label == label).select(value)
    return at


def click(at, label):
    next(b for b in at.button if label in b.label).click()
    return at.run()


def metrics(at):
    return {m.label: m.value for m in at.metric}


@pytest.mark.integration
def test_welcome_screen_renders():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception


@pytest.mark.integration
def test_control_group_is_control_even_when_treatment_rows_come_first():
    """ecommerce_ab_test.csv starts with a treatment row; lift must still be treatment vs control"""
    at = load_app("E-commerce A/B test (funnel)")
    
    control = next(s for s in at.selectbox if s.label == "Control Group")
    treatment = next(s for s in at.selectbox if s.label == "Treatment Group")
    assert control.value == "control"
    assert treatment.value == "treatment"
    
    select(at, "Metric Column", "order_value").run()
    at = click(at, "Run A/B Test")
    assert not at.exception
    
    import pandas as pd
    df = pd.read_csv(os.path.join(os.path.dirname(APP), "data", "ecommerce_ab_test.csv"))
    means = df.groupby("variant").order_value.mean()
    shown = metrics(at)
    assert shown["Control Mean"] == f"{means['control']:.4f}"
    assert shown["Treatment Mean"] == f"{means['treatment']:.4f}"


@pytest.mark.integration
@pytest.mark.parametrize("test_type", ["T-Test", "Z-Test", "Chi-Squared", "Bayesian"])
def test_every_ab_test_type_runs(test_type):
    at = load_app("A/B test (revenue, conversion)")
    select(at, "Metric Column", "revenue")
    select(at, "Test Type", test_type).run()
    at = click(at, "Run A/B Test")
    assert not at.exception
    assert not at.error


@pytest.mark.integration
def test_psm_runs_from_the_ui_on_string_labels():
    at = load_app("A/B test (revenue, conversion)")
    select(at, "Treatment Column", "group").run()
    select(at, "Treated Value", "treatment")
    select(at, "Outcome Column", "revenue").run()
    next(m for m in at.multiselect if m.label.startswith("Covariate")).select("age").select("tenure_days").run()
    at = click(at, "Run PSM Analysis")
    
    assert not at.exception
    assert not at.error
    assert "ATT (Treatment Effect)" in metrics(at)


@pytest.mark.integration
def test_did_runs_with_clustered_errors():
    at = load_app("Difference-in-differences (store sales)")
    select(at, "Select Causal Method", "Difference-in-Differences (DiD)").run()
    select(at, "Group Column", "region")
    select(at, "Time Period Column", "period")
    select(at, "Outcome Column", "sales").run()
    select(at, "Treatment Group Value", "treatment")
    select(at, "Post-Treatment Period Value", "post")
    select(at, "Cluster Standard Errors By (Optional)", "store_id").run()
    at = click(at, "Run DiD Analysis")
    
    assert not at.exception
    assert not at.error
    assert "DiD Estimate" in metrics(at)
    assert any("Not testable" in m.value for m in at.markdown)


@pytest.mark.integration
def test_iv_runs_and_reports_interpretation():
    at = load_app("A/B test (revenue, conversion)")
    select(at, "Select Causal Method", "Instrumental Variables (IV)").run()
    select(at, "Outcome Variable", "revenue")
    select(at, "Treatment Variable (Endogenous)", "page_views")
    select(at, "Instrumental Variable", "sessions").run()
    at = click(at, "Run IV Analysis")
    
    assert not at.exception
    assert not at.error
    assert "IV Estimate" in metrics(at)
