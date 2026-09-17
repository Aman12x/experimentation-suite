"""
Tests for the Causal Inference Lab

Each method is checked on simulated data where the true effect is known.
"""

import numpy as np
import pandas as pd
import pytest

from modules.causal_inference import CausalInferenceLab


@pytest.fixture
def lab():
    return CausalInferenceLab()


def confounded_data(rng, n=4000, true_effect=5.0, labels=False):
    """Treatment uptake and outcome both depend on x, so the naive gap is biased"""
    x = rng.normal(0, 1, n)
    treated = rng.random(n) < 1 / (1 + np.exp(-1.2 * x))
    y = 10 + true_effect * treated + 4 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({'treated': treated.astype(int), 'x': x, 'y': y})
    if labels:
        df['treated'] = np.where(treated, 'exposed', 'holdout')
    return df


# =============== PROPENSITY SCORE MATCHING ===============

@pytest.mark.statistical
def test_psm_recovers_effect_that_naive_comparison_misses(lab):
    df = confounded_data(np.random.default_rng(0))
    naive = df[df.treated == 1].y.mean() - df[df.treated == 0].y.mean()
    
    results = lab.propensity_score_matching(df, 'treated', 'y', ['x'], caliper=0.05)
    
    assert abs(naive - 5.0) > 2            # confounding is real in this data
    assert results['att'] == pytest.approx(5.0, abs=0.5)
    assert results['ci_lower'] < 5.0 < results['ci_upper']
    assert results['max_abs_smd'] < 0.1    # matching balanced the covariate
    before = results['balance_stats']['std_mean_diff_before'].abs().max()
    assert before > 0.5


@pytest.mark.unit
def test_psm_accepts_string_labels_with_treated_value(lab):
    df = confounded_data(np.random.default_rng(1), n=1500, labels=True)
    results = lab.propensity_score_matching(
        df, 'treated', 'y', ['x'], treated_value='exposed'
    )
    assert 'error' not in results
    assert results['n_treated_total'] == int((df.treated == 'exposed').sum())
    assert results['n_matched'] > 0


@pytest.mark.unit
def test_psm_rejects_ambiguous_labels(lab):
    df = confounded_data(np.random.default_rng(2), n=500, labels=True)
    with pytest.raises(ValueError, match="treated_value"):
        lab.propensity_score_matching(df, 'treated', 'y', ['x'])


@pytest.mark.unit
def test_psm_matches_without_replacement_and_within_caliper(lab):
    df = confounded_data(np.random.default_rng(3), n=1500)
    results = lab.propensity_score_matching(df, 'treated', 'y', ['x'], caliper=0.02)
    
    assert results['matched_control'].index.is_unique
    gap = np.abs(
        results['matched_treated']['propensity_score'].to_numpy()
        - results['matched_control']['propensity_score'].to_numpy()
    )
    assert gap.max() <= 0.02
    assert results['n_matched'] <= results['n_control_total']


@pytest.mark.unit
def test_psm_does_not_mutate_input(lab):
    df = confounded_data(np.random.default_rng(4), n=600)
    before = df.copy()
    lab.propensity_score_matching(df, 'treated', 'y', ['x'])
    pd.testing.assert_frame_equal(df, before)


# =============== DIFFERENCE-IN-DIFFERENCES ===============

def did_panel(rng, n_units=300, level_gap=100.0, effect=5.0, unit_sd=8.0):
    rows = []
    for unit in range(n_units):
        treated = unit < n_units // 2
        unit_effect = rng.normal(0, unit_sd)
        for period, trend in (('pre', 0.0), ('post', 10.0)):
            for _ in range(4):  # repeated observations per unit and period
                rows.append({
                    'unit': unit,
                    'group': 'treatment' if treated else 'control',
                    'period': period,
                    'y': 50 + level_gap * treated + trend + unit_effect
                         + effect * (treated and period == 'post') + rng.normal(0, 1),
                })
    return pd.DataFrame(rows)


@pytest.mark.statistical
def test_did_recovers_effect_despite_level_gap(lab):
    df = did_panel(np.random.default_rng(0))
    results = lab.difference_in_differences(df, 'group', 'period', 'y', 'treatment', 'post')
    
    assert results['did_estimate'] == pytest.approx(5.0, abs=0.5)
    assert results['pre_treatment_diff'] == pytest.approx(100.0, abs=3)
    # A level gap is not a parallel-trends violation and must not be flagged as one
    assert results['parallel_trends_assumption'] is None
    assert results['parallel_trends_testable'] is False


@pytest.mark.unit
def test_did_matches_hand_computed_cell_means(lab, sample_did_data):
    results = lab.difference_in_differences(
        sample_did_data, 'group', 'period', 'outcome', 'treatment', 'post'
    )
    m = sample_did_data.groupby(['group', 'period']).outcome.mean()
    by_hand = (m['treatment', 'post'] - m['treatment', 'pre']) - (m['control', 'post'] - m['control', 'pre'])
    assert results['did_estimate'] == pytest.approx(by_hand)


@pytest.mark.unit
def test_did_cluster_option_changes_standard_error_not_estimate(lab):
    df = did_panel(np.random.default_rng(1))
    plain = lab.difference_in_differences(df, 'group', 'period', 'y', 'treatment', 'post')
    clustered = lab.difference_in_differences(
        df, 'group', 'period', 'y', 'treatment', 'post', cluster_col='unit'
    )
    assert clustered['did_estimate'] == pytest.approx(plain['did_estimate'])
    assert clustered['se'] != pytest.approx(plain['se'], rel=0.05)
    assert 'cluster' in clustered['se_type']


@pytest.mark.unit
def test_did_rejects_missing_cell(lab, sample_did_data):
    with pytest.raises(ValueError, match="four cells"):
        lab.difference_in_differences(
            sample_did_data, 'group', 'period', 'outcome', 'treatment', 'not-a-period'
        )


@pytest.mark.unit
def test_did_handles_column_names_with_spaces(lab, sample_did_data):
    df = sample_did_data.rename(columns={'outcome': 'weekly sales'})
    results = lab.difference_in_differences(df, 'group', 'period', 'weekly sales', 'treatment', 'post')
    assert np.isfinite(results['did_estimate'])


# =============== INSTRUMENTAL VARIABLES ===============

def iv_data(rng, n=3000, instrument_strength=0.5, true_effect=2.0):
    z = rng.normal(size=n)
    u = rng.normal(size=n)          # unobserved confounder
    x1 = rng.normal(size=n)
    t = instrument_strength * z + 2 * x1 + u + rng.normal(size=n)
    y = true_effect * t + x1 + 3 * u + rng.normal(size=n)
    return pd.DataFrame({'y': y, 't': t, 'z': z, 'x1': x1})


@pytest.mark.statistical
def test_iv_removes_confounding_bias(lab):
    results = lab.instrumental_variables(iv_data(np.random.default_rng(0)), 'y', 't', 'z', ['x1'])
    
    assert results['iv_estimate'] == pytest.approx(2.0, abs=0.3)
    assert results['ols_estimate'] > 2.5          # naive OLS is biased upward here
    assert results['weak_instrument'] is False


@pytest.mark.unit
def test_iv_without_covariates_equals_wald_ratio(lab):
    df = iv_data(np.random.default_rng(1))
    results = lab.instrumental_variables(df, 'y', 't', 'z')
    wald = np.cov(df.z, df.y)[0, 1] / np.cov(df.z, df.t)[0, 1]
    assert results['iv_estimate'] == pytest.approx(wald)


@pytest.mark.unit
def test_iv_flags_weak_instrument_even_with_strong_covariate(lab):
    """The covariate explains most of the treatment; the instrument explains almost none.
    An overall first-stage F would look huge here, so the check has to be the partial F."""
    df = iv_data(np.random.default_rng(2), instrument_strength=0.01)
    results = lab.instrumental_variables(df, 'y', 't', 'z', ['x1'])
    
    assert results['first_stage_r2'] > 0.5
    assert results['first_stage_f_stat'] < 10
    assert results['weak_instrument'] is True


@pytest.mark.slow
@pytest.mark.statistical
def test_iv_confidence_interval_coverage(lab):
    """2SLS standard errors must come from structural residuals; a hand-rolled
    second-stage OLS gets them wrong, which shows up as broken coverage."""
    rng = np.random.default_rng(3)
    n_sims, covered = 400, 0
    for _ in range(n_sims):
        r = lab.instrumental_variables(iv_data(rng, n=800), 'y', 't', 'z', ['x1'])
        covered += r['ci_lower'] <= 2.0 <= r['ci_upper']
    assert covered / n_sims == pytest.approx(0.95, abs=0.035)


@pytest.mark.unit
def test_iv_rejects_reused_columns(lab):
    with pytest.raises(ValueError):
        lab.instrumental_variables(iv_data(np.random.default_rng(4), n=100), 'y', 't', 't')
