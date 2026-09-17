"""
Statistical validity tests

Checks the engine against reference implementations and against its own
long-run guarantees (false-positive rate, interval coverage) by simulation.
"""

import numpy as np
import pytest
from scipy import stats

from modules.ab_testing import ABTestingEngine


@pytest.fixture
def engine():
    return ABTestingEngine()


@pytest.mark.unit
@pytest.mark.statistical
def test_default_t_test_matches_scipy_welch(engine):
    rng = np.random.default_rng(1)
    control = rng.normal(10, 1, 400)
    treatment = rng.normal(10.2, 3, 60)
    
    results = engine.t_test(control, treatment)
    reference = stats.ttest_ind(treatment, control, equal_var=False)
    
    assert results['variant'] == 'welch'
    assert results['t_statistic'] == pytest.approx(reference.statistic)
    assert results['p_value'] == pytest.approx(reference.pvalue)
    assert results['degrees_of_freedom'] == pytest.approx(reference.df)
    
    ci = reference.confidence_interval(0.95)
    assert results['ci_lower'] == pytest.approx(ci.low)
    assert results['ci_upper'] == pytest.approx(ci.high)


@pytest.mark.unit
@pytest.mark.statistical
def test_student_option_matches_scipy_pooled(engine):
    rng = np.random.default_rng(2)
    control = rng.normal(50, 5, 80)
    treatment = rng.normal(52, 5, 90)
    
    results = engine.t_test(control, treatment, equal_var=True)
    reference = stats.ttest_ind(treatment, control, equal_var=True)
    
    assert results['variant'] == 'student'
    assert results['p_value'] == pytest.approx(reference.pvalue)
    ci = reference.confidence_interval(0.95)
    assert results['ci_lower'] == pytest.approx(ci.low)
    assert results['ci_upper'] == pytest.approx(ci.high)


@pytest.mark.slow
@pytest.mark.statistical
def test_false_positive_rate_holds_under_unequal_variance_and_size(engine):
    """With no true effect, ~5% of tests should be significant at alpha=0.05.

    Unequal group sizes with unequal variances is the case where the pooled
    Student's test breaks down, so it is the case worth guarding.
    """
    rng = np.random.default_rng(3)
    n_sims = 2000
    welch_hits = student_hits = 0
    for _ in range(n_sims):
        control = rng.normal(0, 1, 1000)
        treatment = rng.normal(0, 4, 100)
        welch_hits += engine.t_test(control, treatment)['significant']
        student_hits += engine.t_test(control, treatment, equal_var=True)['significant']
    
    assert welch_hits / n_sims == pytest.approx(0.05, abs=0.015)
    # Documents why Welch is the default: the pooled test is badly miscalibrated here
    assert student_hits / n_sims > 0.20


@pytest.mark.slow
@pytest.mark.statistical
def test_confidence_interval_coverage(engine):
    """The 95% CI should contain the true difference about 95% of the time"""
    rng = np.random.default_rng(4)
    true_diff = 2.0
    n_sims = 2000
    covered = 0
    for _ in range(n_sims):
        control = rng.normal(100, 10, 200)
        treatment = rng.normal(100 + true_diff, 20, 80)
        r = engine.t_test(control, treatment)
        covered += r['ci_lower'] <= true_diff <= r['ci_upper']
    
    assert covered / n_sims == pytest.approx(0.95, abs=0.015)


@pytest.mark.unit
@pytest.mark.statistical
def test_bayesian_probability_matches_closed_form_integral(engine):
    from scipy import integrate
    
    results = engine.bayesian_ab_test(50, 1000, 65, 1000)
    a_c, b_c = results['control_posterior_alpha'], results['control_posterior_beta']
    a_t, b_t = results['treatment_posterior_alpha'], results['treatment_posterior_beta']
    
    # P(treatment > control) = ∫ pdf_t(x) * cdf_c(x) dx
    exact, _ = integrate.quad(
        lambda x: stats.beta.pdf(x, a_t, b_t) * stats.beta.cdf(x, a_c, b_c), 0, 1, points=[0.05, 0.065]
    )
    assert results['prob_treatment_better'] == pytest.approx(exact, abs=0.005)


@pytest.mark.unit
def test_bayesian_test_leaves_global_rng_alone(engine):
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    engine.bayesian_ab_test(50, 1000, 65, 1000)
    assert np.random.random() == expected


@pytest.mark.unit
def test_power_analysis_matches_textbook_formula(engine):
    results = engine.calculate_sample_size(baseline_mean=100, mde=5, baseline_std=20)
    # n per group ≈ 2 * (z_{1-α/2} + z_{power})² / d²
    d = 5 / 20
    approx_n = 2 * (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) ** 2 / d ** 2
    assert results['n_control'] == pytest.approx(approx_n, rel=0.02)
    assert results['total_sample_size'] == results['n_control'] + results['n_treatment']


@pytest.mark.unit
def test_invalid_inputs_raise(engine):
    with pytest.raises(ValueError):
        engine.t_test(np.array([1.0, 2.0]), np.array([1.0, 2.0]), alternative='bogus')
    with pytest.raises(ValueError):
        engine.chi_squared_test(5, 0, 5, 10)
    with pytest.raises(ValueError):
        engine.bayesian_ab_test(11, 10, 5, 10)
    with pytest.raises(ValueError):
        engine.calculate_sample_size(baseline_mean=0, mde=5, baseline_std=20)


@pytest.mark.unit
def test_results_are_json_native(engine):
    import json
    rng = np.random.default_rng(5)
    a, b = rng.normal(0, 1, 50), rng.normal(0.5, 1, 50)
    for results in (
        engine.t_test(a, b),
        engine.z_test(a, b),
        engine.chi_squared_test(50, 1000, 65, 1000),
        engine.bayesian_ab_test(50, 1000, 65, 1000),
        engine.calculate_sample_size(100, 5, 20),
    ):
        json.dumps(results, allow_nan=False)
