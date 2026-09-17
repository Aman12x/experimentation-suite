"""
Tests for health checks and plain-English interpretation
"""

import numpy as np
import pandas as pd
import pytest

from modules.ab_testing import ABTestingEngine
from modules.health_checks import HealthChecker
from utils.interpreters import StatisticalInterpreter


def two_group_frame(rng, n_control=1000, n_treatment=1000):
    return pd.DataFrame({
        'group': ['control'] * n_control + ['treatment'] * n_treatment,
        'metric': rng.normal(100, 10, n_control + n_treatment),
    })


# =============== HEALTH CHECKS ===============

@pytest.mark.unit
def test_srm_clean_split_passes():
    result = HealthChecker().check_sample_ratio_mismatch(
        two_group_frame(np.random.default_rng(0)), 'group'
    )
    assert result['has_srm'] is False


@pytest.mark.unit
def test_srm_detects_skewed_split():
    df = two_group_frame(np.random.default_rng(0), 1000, 800)
    result = HealthChecker().check_sample_ratio_mismatch(df, 'group')
    assert result['has_srm'] is True
    assert result['severity'] == 'CRITICAL'


@pytest.mark.unit
def test_srm_respects_expected_ratio_by_label():
    df = two_group_frame(np.random.default_rng(0), 900, 100)
    checker = HealthChecker()
    assert checker.check_sample_ratio_mismatch(df, 'group')['has_srm'] is True
    assert checker.check_sample_ratio_mismatch(
        df, 'group', expected_ratio={'treatment': 0.1, 'control': 0.9}
    )['has_srm'] is False


@pytest.mark.unit
def test_srm_supports_more_than_two_groups():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({'group': rng.choice(list('ABC'), 6000), 'metric': rng.normal(size=6000)})
    result = HealthChecker().check_sample_ratio_mismatch(df, 'group')
    assert 'error' not in result
    assert result['has_srm'] is False
    
    skewed = pd.concat([df, df[df.group == 'A']])
    assert HealthChecker().check_sample_ratio_mismatch(skewed, 'group')['has_srm'] is True


@pytest.mark.unit
def test_check_that_cannot_run_is_not_reported_as_healthy():
    df = pd.DataFrame({'group': ['only'] * 50, 'metric': np.arange(50.0)})
    results = HealthChecker().run_all_checks(df, 'group', 'metric')
    assert results['overall_health'] is False
    assert any('could not run' in w for w in results['warnings'])


@pytest.mark.unit
def test_variance_warning_fires_on_large_ratio():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        'group': ['control'] * 500 + ['treatment'] * 500,
        'metric': np.concatenate([rng.normal(0, 1, 500), rng.normal(0, 5, 500)]),
    })
    result = HealthChecker().check_variance_ratio(df, 'group', 'metric')
    assert result['severity'] == 'WARNING'
    assert result['variance_ratio'] > 4


# =============== INTERPRETATION ===============

def recommendation(results):
    text = StatisticalInterpreter.interpret_ab_test_results(results)
    return text.split("### 💡 Recommendation")[1]


@pytest.mark.unit
def test_significantly_worse_treatment_is_never_recommended():
    rng = np.random.default_rng(3)
    results = ABTestingEngine().t_test(rng.normal(100, 10, 2000), rng.normal(90, 10, 2000))
    assert results['significant'] and results['relative_lift'] < 0
    
    verdict = recommendation(results)
    assert "DO NOT IMPLEMENT" in verdict
    assert "RECOMMEND:" not in verdict


@pytest.mark.unit
def test_significant_improvement_is_recommended():
    rng = np.random.default_rng(4)
    results = ABTestingEngine().t_test(rng.normal(100, 10, 2000), rng.normal(110, 10, 2000))
    assert "✅ RECOMMEND" in recommendation(results)


@pytest.mark.unit
def test_small_significant_gain_gets_caution():
    rng = np.random.default_rng(5)
    results = ABTestingEngine().t_test(rng.normal(100, 5, 50000), rng.normal(100.5, 5, 50000))
    assert results['significant'] and 0 < results['relative_lift'] <= 2
    assert "PROCEED WITH CAUTION" in recommendation(results)


@pytest.mark.unit
def test_null_result_is_not_recommended():
    rng = np.random.default_rng(6)
    results = ABTestingEngine().t_test(rng.normal(100, 10, 200), rng.normal(100, 10, 200))
    assert not results['significant']
    assert "DO NOT IMPLEMENT" in recommendation(results)


@pytest.mark.unit
def test_did_interpretation_does_not_claim_untestable_assumption():
    text = StatisticalInterpreter.interpret_causal_effect({
        'did_estimate': 5.0, 'p_value': 0.01, 'ci_lower': 3.0, 'ci_upper': 7.0,
        'parallel_trends_assumption': None,
    }, "DiD")
    assert "Not testable" in text
    assert "trends appear similar" not in text


@pytest.mark.unit
def test_iv_interpretation_warns_on_weak_instrument():
    base = {'iv_estimate': 2.0, 'ci_lower': 1.0, 'ci_upper': 3.0, 'p_value': 0.01, 'ols_estimate': 2.8}
    weak = StatisticalInterpreter.interpret_causal_effect(
        {**base, 'first_stage_f_stat': 3.4, 'weak_instrument': True}, "IV")
    strong = StatisticalInterpreter.interpret_causal_effect(
        {**base, 'first_stage_f_stat': 600.0, 'weak_instrument': False}, "IV")
    assert "Weak Instrument" in weak
    assert "Weak Instrument" not in strong


@pytest.mark.unit
def test_unknown_causal_method_raises():
    with pytest.raises(ValueError):
        StatisticalInterpreter.interpret_causal_effect({}, "nope")
