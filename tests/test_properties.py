"""
Property-based tests

Instead of hand-picked inputs, hypothesis generates hundreds of awkward ones
(tiny samples, constants, huge and tiny scales, negative baselines, skew) and
checks that invariants hold for all of them.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from modules.ab_testing import ABTestingEngine
from modules.experiment_design import describe_metric, to_unit_level
from utils.decision import ship_decision
from utils.interpreters import StatisticalInterpreter

engine = ABTestingEngine()
KNOWN_DECISIONS = {"SHIP", "DO NOT SHIP", "KEEP RUNNING", "STOP - NO MEANINGFUL EFFECT", "INVALID - FIX THE EXPERIMENT"}

finite = st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False, width=64)
samples = arrays(np.float64, st.integers(min_value=2, max_value=60), elements=finite)
COMMON = dict(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def assert_well_formed(result):
    json.dumps(result, allow_nan=False)                    # API-safe: no NaN, inf, or numpy types
    assert 0.0 <= result['p_value'] <= 1.0
    assert isinstance(result['significant'], bool)
    if 'ci_lower' in result:
        assert result['ci_lower'] <= result['ci_upper']
    lift = result['relative_lift']
    if lift is not None:
        assert math.isfinite(lift)
        assert result['control_mean'] > 0                  # only ever reported against a positive baseline
        assert (lift > 0) == (result['mean_difference'] > 0) or lift == 0 or result['mean_difference'] == 0


@pytest.mark.slow
@settings(**COMMON)
@given(control=samples, treatment=samples)
def test_two_sample_tests_are_well_formed_for_any_finite_input(control, treatment):
    for method in (engine.t_test, engine.z_test, engine.mann_whitney_test):
        assert_well_formed(method(control, treatment))


@pytest.mark.slow
@settings(**COMMON)
@given(control=samples, treatment=samples)
def test_swapping_arms_flips_the_effect_and_keeps_the_p_value(control, treatment):
    forward, backward = engine.t_test(control, treatment), engine.t_test(treatment, control)
    assert forward['p_value'] == pytest.approx(backward['p_value'], abs=1e-9)
    assert forward['mean_difference'] == pytest.approx(-backward['mean_difference'], rel=1e-9, abs=1e-9)


@pytest.mark.slow
@settings(**COMMON)
@given(control=samples, treatment=samples, scale=st.floats(min_value=1e-3, max_value=1e3))
def test_changing_units_does_not_change_the_conclusion(control, treatment, scale):
    """Dollars or cents: the p-value must not care"""
    assume(np.std(control) > 1e-6 * (1 + abs(np.mean(control))))
    assume(np.std(treatment) > 1e-6 * (1 + abs(np.mean(treatment))))
    base, scaled = engine.t_test(control, treatment), engine.t_test(control * scale, treatment * scale)
    assert scaled['p_value'] == pytest.approx(base['p_value'], rel=1e-6, abs=1e-9)


@pytest.mark.slow
@settings(**COMMON)
@given(
    n_c=st.integers(1, 5000), n_t=st.integers(1, 5000),
    rate_c=st.floats(0, 1), rate_t=st.floats(0, 1)
)
def test_proportion_tests_are_well_formed_for_any_counts(n_c, n_t, rate_c, rate_t):
    counts = (int(round(rate_c * n_c)), n_c, int(round(rate_t * n_t)), n_t)
    assert_well_formed(engine.proportions_test(*counts))
    assert_well_formed(engine.chi_squared_test(*counts))
    bayes = engine.bayesian_ab_test(*counts)
    json.dumps(bayes, allow_nan=False)
    assert 0.0 <= bayes['prob_treatment_better'] <= 1.0


@pytest.mark.slow
@settings(**COMMON)
@given(control=samples, treatment=samples, guardrail=samples, up_is_good=st.booleans(), mde=st.floats(0.1, 50))
def test_decision_and_interpretation_never_crash_on_engine_output(control, treatment, guardrail, up_is_good, mde):
    primary = engine.t_test(control, treatment)
    guard = engine.t_test(control, guardrail)
    decision = ship_decision(
        primary, guardrails={'g': guard}, higher_is_better=up_is_good,
        guardrail_higher_is_better={'g': not up_is_good}, mde_pct=mde
    )
    assert decision['decision'] in KNOWN_DECISIONS
    assert decision['reasons'] and all("nan" not in r.lower() and "inf%" not in r for r in decision['reasons'])
    
    text = StatisticalInterpreter.interpret_ab_test_results(primary, higher_is_better=up_is_good)
    assert "nan" not in text.lower()
    # Never recommend shipping a significant move in the wrong direction
    if primary['significant'] and (primary['mean_difference'] > 0) != up_is_good:
        assert decision['decision'] == "DO NOT SHIP"
        assert "✅ RECOMMEND" not in text


@pytest.mark.slow
@settings(**COMMON)
@given(values=arrays(np.float64, st.integers(1, 80), elements=finite))
def test_metric_description_always_names_a_test_the_app_has(values):
    info = describe_metric(pd.Series(values))
    assert info['recommended_test'] in {'T-Test', 'Proportions Z-Test', 'Bootstrap'}
    assert info['kind'] in {'binary', 'count', 'continuous'}


@pytest.mark.slow
@settings(**COMMON)
@given(
    rows=st.lists(
        st.tuples(st.integers(0, 15), st.sampled_from(['control', 'treatment']), finite),
        min_size=1, max_size=80
    ),
    agg=st.sampled_from(['mean', 'sum', 'max'])
)
def test_unit_aggregation_invariants(rows, agg):
    df = pd.DataFrame(rows, columns=['user_id', 'variant', 'y'])
    unit_df, info = to_unit_level(df, 'user_id', 'variant', ['y'], agg=agg)
    
    arms_per_user = df.groupby('user_id').variant.nunique()
    clean_users = set(arms_per_user[arms_per_user == 1].index)
    assert set(unit_df.user_id) == clean_users                     # contaminated users gone, nobody else lost
    assert unit_df.user_id.is_unique                               # exactly one row per unit
    assert info['n_contaminated_units'] == len(arms_per_user) - len(clean_users)
    assert info['n_units'] == len(unit_df)
