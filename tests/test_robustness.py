"""
Regression tests for robustness fixes: bad baselines, non-finite input,
arrival order, and the app surviving datasets it cannot use for A/B testing
"""

import json
import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from modules.ab_testing import ABTestingEngine
from modules.ab_advanced import format_change
from modules.experiment_design import to_unit_level
from utils.decision import ship_decision, SHIP, DO_NOT_SHIP
from utils.interpreters import StatisticalInterpreter

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@pytest.fixture
def engine():
    return ABTestingEngine()


# =============== NON-FINITE INPUT ===============

@pytest.mark.unit
@pytest.mark.parametrize("method", ["t_test", "z_test", "mann_whitney_test", "bootstrap_test"])
def test_infinite_values_are_rejected(engine, method):
    """An inf used to flow through to p=1.0, 'not significant'"""
    control = np.array([1.0, 2.0, 3.0, 4.0])
    treatment = np.array([2.0, 3.0, np.inf, 5.0])
    with pytest.raises(ValueError, match="infinite"):
        getattr(engine, method)(control, treatment)


# =============== RELATIVE LIFT ON A BAD BASELINE ===============

@pytest.mark.unit
def test_lift_is_undefined_when_control_mean_is_near_zero(engine):
    rng = np.random.default_rng(0)
    r = engine.t_test(rng.normal(0.0, 1, 5000), rng.normal(0.3, 1, 5000))
    
    assert r['relative_lift'] is None
    assert 'lift_ci_lower' not in r
    assert "control mean" in r['relative_lift_note']
    assert r['significant'] and r['mean_difference'] == pytest.approx(0.3, abs=0.06)
    json.dumps(r, allow_nan=False)


@pytest.mark.unit
def test_lift_is_undefined_for_a_negative_baseline_and_direction_still_right(engine):
    """Metric rises from -10 to -9: a naive ratio calls that a -10% lift"""
    rng = np.random.default_rng(1)
    r = engine.t_test(rng.normal(-10, 1, 5000), rng.normal(-9, 1, 5000))
    
    assert r['relative_lift'] is None
    assert ship_decision(r)['decision'] == SHIP
    assert ship_decision(r, higher_is_better=False)['decision'] == DO_NOT_SHIP


@pytest.mark.unit
def test_lift_stays_defined_for_a_healthy_baseline(engine):
    rng = np.random.default_rng(2)
    r = engine.t_test(rng.normal(100, 10, 2000), rng.normal(104, 10, 2000))
    assert r['relative_lift'] == pytest.approx(4.0, abs=1.0)
    assert r['lift_ci_lower'] < r['relative_lift'] < r['lift_ci_upper']


@pytest.mark.unit
def test_zero_control_rate_has_no_relative_lift(engine):
    for r in (engine.proportions_test(0, 1000, 10, 1000), engine.chi_squared_test(0, 1000, 10, 1000)):
        assert r['relative_lift'] is None


@pytest.mark.unit
@pytest.mark.parametrize("method", ["t_test", "z_test", "mann_whitney_test", "bootstrap_test", "sequential_test"])
def test_every_method_agrees_on_undefined_lift(engine, method):
    rng = np.random.default_rng(3)
    r = getattr(engine, method)(rng.normal(0, 1, 400), rng.normal(0.5, 1, 400))
    assert r['relative_lift'] is None
    json.dumps({k: v for k, v in r.items() if k != 'looks'}, allow_nan=False)


@pytest.mark.unit
def test_decision_and_interpretation_fall_back_to_absolute_change(engine):
    rng = np.random.default_rng(4)
    r = engine.t_test(rng.normal(0.0, 1, 5000), rng.normal(0.3, 1, 5000))
    
    assert "absolute" in format_change(r)
    decision = ship_decision(r, guardrails={'g': r}, guardrail_higher_is_better={'g': False})
    text = " ".join(decision['reasons']) + StatisticalInterpreter.interpret_ab_test_results(r)
    assert "%," not in " ".join(decision['reasons'])      # no percentage quoted anywhere
    assert "Absolute Change" in text
    assert "11" not in format_change(r)                   # the old +11203% is gone
    assert "RECOMMEND" in text


# =============== ARRIVAL ORDER ===============

@pytest.mark.unit
def test_unit_aggregation_keeps_first_appearance_order():
    df = pd.DataFrame({
        'user_id': [30, 10, 30, 20, 10], 'variant': ['a', 'b', 'a', 'a', 'b'], 'y': [1.0, 2.0, 3.0, 4.0, 6.0],
    })
    unit_df, _ = to_unit_level(df, 'user_id', 'variant', ['y'])
    assert unit_df.user_id.tolist() == [30, 10, 20]        # not sorted by id


@pytest.mark.unit
def test_order_column_defines_arrival_order():
    df = pd.DataFrame({
        'user_id': [1, 2, 3, 1], 'variant': ['a', 'b', 'a', 'a'], 'y': [1.0, 2.0, 3.0, 5.0],
        'ts': pd.to_datetime(['2026-01-03', '2026-01-01', '2026-01-02', '2026-01-04']),
    })
    unit_df, _ = to_unit_level(df, 'user_id', 'variant', ['y'], order_col='ts')
    assert unit_df.user_id.tolist() == [2, 3, 1]
    
    single_rows, _ = to_unit_level(df.iloc[:3], 'user_id', 'variant', ['y'], order_col='ts')
    assert single_rows.user_id.tolist() == [2, 3, 1]


# =============== APP SURVIVES UNUSABLE DATA ===============

def load_with(frame_or_error):
    """Run the app with the sample loader replaced, since AppTest cannot upload a file"""
    from streamlit.testing.v1 import AppTest
    
    def fake_load(self, path):
        if isinstance(frame_or_error, Exception):
            raise frame_or_error
        self.data = frame_or_error
        self._identify_column_types()
        return self.data
    
    with mock.patch("modules.data_handler.DataHandler.load_path", fake_load):
        at = AppTest.from_file(APP, default_timeout=60).run()
        at.sidebar.selectbox[0].select("A/B test (revenue, conversion)").run()
    return at


@pytest.mark.integration
def test_dataset_without_an_assignment_column_still_gets_the_other_tabs():
    """An IV-only table has nothing to A/B test; that must not blank the Causal tab"""
    rng = np.random.default_rng(5)
    at = load_with(pd.DataFrame({'y': rng.normal(size=300), 't': rng.normal(size=300), 'z': rng.normal(size=300)}))
    
    assert not at.exception
    assert any("No column looks like an arm assignment" in e.value for e in at.error)
    assert any(s.label == "Select Causal Method" for s in at.selectbox)
    assert any("Sample Size Calculator" in h.value for h in at.subheader)
    assert any("Report Export" in h.value for h in at.header)


@pytest.mark.integration
def test_failed_load_shows_an_error_not_a_traceback():
    at = load_with(OSError("disk on fire"))
    
    assert not at.exception
    assert any("disk on fire" in e.value for e in at.sidebar.error)
    assert not any(s.label == "Primary Metric" for s in at.selectbox)


@pytest.mark.integration
def test_sequential_test_asks_for_arrival_order():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox[0].select("A/B test (revenue, conversion)").run()
    assert not any(s.label == "Arrival Order" for s in at.selectbox)
    
    next(s for s in at.selectbox if s.label == "Statistical Test").select("Sequential (always-valid)").run()
    assert any(s.label == "Arrival Order" for s in at.selectbox)
