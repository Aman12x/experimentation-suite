"""
Tests for the advanced A/B methods: proportions, lift intervals, CUPED,
non-parametric and bootstrap tests, multiplicity correction, sequential testing
"""

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from modules.ab_testing import ABTestingEngine
from modules.ab_advanced import is_binary_metric


@pytest.fixture
def engine():
    return ABTestingEngine()


# =============== PROPORTIONS ===============

@pytest.mark.unit
@pytest.mark.statistical
def test_proportions_test_matches_statsmodels(engine):
    results = engine.proportions_test(50, 1000, 65, 1000)
    z, p = proportions_ztest([65, 50], [1000, 1000])
    
    assert results['z_statistic'] == pytest.approx(z)
    assert results['p_value'] == pytest.approx(p)
    assert results['relative_lift'] == pytest.approx(30.0)
    assert results['ci_lower'] < results['rate_difference'] < results['ci_upper']


@pytest.mark.unit
def test_binary_metric_detection():
    assert is_binary_metric([0, 1, 1, 0])
    assert is_binary_metric([1.0, 1.0])
    assert not is_binary_metric([0, 1, 2])
    assert not is_binary_metric([0.5, 1])


# =============== RELATIVE LIFT INTERVAL ===============

@pytest.mark.slow
@pytest.mark.statistical
def test_relative_lift_interval_coverage(engine):
    """Delta-method interval should cover the true lift ~95% of the time.
    The naive interval (absolute CI / control mean) is included to show it under-covers
    when the control mean is noisy."""
    rng = np.random.default_rng(0)
    true_lift = 10.0
    n_sims, covered, naive_covered = 2000, 0, 0
    for _ in range(n_sims):
        control = rng.exponential(10, 150)
        treatment = rng.exponential(11, 150)
        r = engine.t_test(control, treatment)
        covered += r['lift_ci_lower'] <= true_lift <= r['lift_ci_upper']
        naive_lo = r['ci_lower'] / r['control_mean'] * 100
        naive_hi = r['ci_upper'] / r['control_mean'] * 100
        naive_covered += naive_lo <= true_lift <= naive_hi
    
    assert covered / n_sims == pytest.approx(0.95, abs=0.02)
    assert covered >= naive_covered


# =============== CUPED ===============

def cuped_data(rng, n=2000, effect=1.0, rho_scale=0.9):
    pre_c, pre_t = rng.normal(100, 20, n), rng.normal(100, 20, n)
    control = rho_scale * pre_c + rng.normal(0, 8, n)
    treatment = rho_scale * pre_t + rng.normal(0, 8, n) + effect
    return control, treatment, pre_c, pre_t


@pytest.mark.statistical
def test_cuped_shrinks_interval_by_about_rho_squared(engine):
    control, treatment, pre_c, pre_t = cuped_data(np.random.default_rng(1))
    results = engine.cuped_test(control, treatment, pre_c, pre_t)
    
    rho = results['covariate_correlation']
    assert rho > 0.85
    assert results['variance_reduction_pct'] == pytest.approx(rho**2 * 100, abs=5)
    width_before = results['unadjusted_ci_upper'] - results['unadjusted_ci_lower']
    width_after = results['ci_upper'] - results['ci_lower']
    assert width_after < 0.6 * width_before


@pytest.mark.slow
@pytest.mark.statistical
def test_cuped_is_unbiased_and_more_powerful(engine):
    rng = np.random.default_rng(2)
    n_sims = 500
    estimates, plain_hits, cuped_hits = [], 0, 0
    for _ in range(n_sims):
        control, treatment, pre_c, pre_t = cuped_data(rng, n=400, effect=1.5)
        r = engine.cuped_test(control, treatment, pre_c, pre_t)
        estimates.append(r['mean_difference'])
        cuped_hits += r['significant']
        plain_hits += r['unadjusted_p_value'] < 0.05
    
    assert np.mean(estimates) == pytest.approx(1.5, abs=0.1)
    assert cuped_hits / n_sims > plain_hits / n_sims + 0.3


@pytest.mark.slow
@pytest.mark.statistical
def test_cuped_keeps_false_positive_rate(engine):
    rng = np.random.default_rng(3)
    n_sims = 1500
    hits = sum(
        engine.cuped_test(*cuped_data(rng, n=300, effect=0.0))['significant'] for _ in range(n_sims)
    )
    assert hits / n_sims == pytest.approx(0.05, abs=0.02)


@pytest.mark.unit
def test_cuped_rejects_misaligned_covariate(engine):
    with pytest.raises(ValueError):
        engine.cuped_test(np.ones(10), np.ones(10), np.ones(9), np.ones(10))


# =============== NON-PARAMETRIC AND BOOTSTRAP ===============

@pytest.mark.unit
def test_mann_whitney_matches_scipy(engine):
    rng = np.random.default_rng(4)
    control, treatment = rng.lognormal(3, 1, 300), rng.lognormal(3.2, 1, 300)
    results = engine.mann_whitney_test(control, treatment)
    reference = stats.mannwhitneyu(treatment, control, alternative='two-sided')
    
    assert results['p_value'] == pytest.approx(reference.pvalue)
    assert 0.5 < results['prob_superiority'] < 1


@pytest.mark.unit
def test_bootstrap_agrees_with_welch_on_well_behaved_data(engine):
    rng = np.random.default_rng(5)
    control, treatment = rng.normal(100, 10, 800), rng.normal(102, 10, 800)
    boot = engine.bootstrap_test(control, treatment)
    welch = engine.t_test(control, treatment)
    
    assert boot['ci_lower'] == pytest.approx(welch['ci_lower'], abs=0.15)
    assert boot['ci_upper'] == pytest.approx(welch['ci_upper'], abs=0.15)
    assert boot['significant'] == welch['significant']
    assert boot['lift_ci_lower'] < boot['relative_lift'] < boot['lift_ci_upper']


@pytest.mark.unit
def test_bootstrap_is_reproducible(engine):
    rng = np.random.default_rng(6)
    control, treatment = rng.normal(0, 1, 100), rng.normal(0.2, 1, 100)
    assert engine.bootstrap_test(control, treatment) == engine.bootstrap_test(control, treatment)


# =============== MULTIPLE VARIANTS ===============

@pytest.mark.unit
def test_multi_variant_adjusted_p_values_match_statsmodels(engine):
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(7)
    groups = {
        'control': rng.normal(100, 10, 500),
        'a': rng.normal(101.5, 10, 500),
        'b': rng.normal(103, 10, 500),
        'c': rng.normal(100, 10, 500),
    }
    results = engine.multi_variant_test(groups, 'control', correction='holm')
    raw = [c['p_value_raw'] for c in results['comparisons']]
    expected = multipletests(raw, method='holm')[1]
    
    assert results['n_comparisons'] == 3
    assert [c['p_value_adjusted'] for c in results['comparisons']] == pytest.approx(expected.tolist())
    assert all(c['p_value_adjusted'] >= c['p_value_raw'] for c in results['comparisons'])
    assert results['best_variant'] == 'b'


@pytest.mark.slow
@pytest.mark.statistical
def test_correction_controls_family_wise_error(engine):
    """Five identical variants: uncorrected testing finds a 'winner' far too often"""
    rng = np.random.default_rng(8)
    n_sims, raw_false, holm_false = 600, 0, 0
    for _ in range(n_sims):
        groups = {f'v{i}': rng.normal(0, 1, 200) for i in range(6)}
        r = engine.multi_variant_test(groups, 'v0', correction='holm')
        raw_false += any(c['significant_raw'] for c in r['comparisons'])
        holm_false += any(c['significant'] for c in r['comparisons'])
    
    assert raw_false / n_sims > 0.15
    assert holm_false / n_sims <= 0.075


@pytest.mark.unit
def test_multi_variant_uses_proportions_for_binary_metrics(engine):
    rng = np.random.default_rng(9)
    groups = {
        'control': rng.binomial(1, 0.10, 3000),
        'a': rng.binomial(1, 0.13, 3000),
    }
    results = engine.multi_variant_test(groups, 'control')
    assert results['metric_type'] == 'binary'
    assert results['comparisons'][0]['test_type'] == 'proportions z-test'


@pytest.mark.unit
def test_multi_variant_rejects_unknown_control(engine):
    with pytest.raises(ValueError):
        engine.multi_variant_test({'a': np.ones(5), 'b': np.ones(5)}, 'control')


# =============== SEQUENTIAL ===============

@pytest.mark.slow
@pytest.mark.statistical
def test_sequential_test_survives_peeking_where_fixed_horizon_does_not(engine):
    """No true effect, 20 looks, stop at the first significant look"""
    rng = np.random.default_rng(10)
    n_sims, naive_stops, valid_stops = 1000, 0, 0
    for _ in range(n_sims):
        r = engine.sequential_test(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000), n_looks=20)
        naive_stops += r['naive_peeking_would_stop_at_look'] is not None
        valid_stops += r['could_stop_at_look'] is not None
    
    assert naive_stops / n_sims > 0.15     # peeking at a fixed-horizon p-value is badly inflated
    assert valid_stops / n_sims <= 0.05    # the always-valid p-value holds its level


@pytest.mark.statistical
def test_sequential_test_detects_real_effect_early(engine):
    rng = np.random.default_rng(11)
    r = engine.sequential_test(rng.normal(0, 1, 5000), rng.normal(0.2, 1, 5000), n_looks=20)
    
    assert r['significant']
    assert r['could_stop_at_look'] < 20
    assert r['ci_lower'] < 0.2 < r['ci_upper']


@pytest.mark.unit
def test_always_valid_p_value_never_increases(engine):
    rng = np.random.default_rng(12)
    r = engine.sequential_test(rng.normal(0, 1, 1500), rng.normal(0.05, 1, 1500))
    p = [look['always_valid_p_value'] for look in r['looks']]
    assert all(later <= earlier for earlier, later in zip(p, p[1:]))


@pytest.mark.unit
def test_sequential_needs_minimum_sample(engine):
    with pytest.raises(ValueError):
        engine.sequential_test(np.ones(10), np.ones(10))


# =============== BAYESIAN EXPECTED LOSS ===============

@pytest.mark.unit
def test_expected_loss_favours_the_better_arm(engine):
    r = engine.bayesian_ab_test(50, 1000, 80, 1000)
    assert r['expected_loss_treatment'] < r['expected_loss_control']
    assert r['expected_loss_treatment'] >= 0
