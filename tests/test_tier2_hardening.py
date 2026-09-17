"""
Tests for ratio metrics, the bounded bootstrap, time-based health checks and export limits
"""

import io
import os
import resource
import time

import numpy as np
import pandas as pd
import pytest

from modules.ab_testing import ABTestingEngine
from modules.ab_advanced import BOOTSTRAP_MAX_UNITS
from modules.experiment_design import describe_metric, guess_time_column, to_unit_level
from modules.health_checks import HealthChecker
from utils.decision import ship_decision, INVALID
from utils.report_generator import ReportGenerator, RAW_DATA_ROW_LIMIT

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
MULTI = "Multi-variant checkout test (CUPED, guardrails)"


@pytest.fixture
def engine():
    return ABTestingEngine()


# =============== RATIO METRICS ===============

def ratio_arm(rng, n, per_session=5.0):
    """Users with more sessions also spend more per session, so mean-of-ratios != ratio-of-totals"""
    sessions = rng.integers(1, 30, n)
    revenue = np.array([rng.exponential(per_session * (1 + s / 30), s).sum() for s in sessions])
    return revenue, sessions.astype(float)


@pytest.mark.unit
def test_ratio_test_targets_the_ratio_of_totals(engine):
    rng = np.random.default_rng(0)
    rev_c, ses_c = ratio_arm(rng, 3000)
    rev_t, ses_t = ratio_arm(rng, 3000, per_session=5.5)
    r = engine.ratio_metric_test(rev_c, ses_c, rev_t, ses_t)
    
    assert r['control_mean'] == pytest.approx(rev_c.sum() / ses_c.sum())
    assert r['treatment_mean'] == pytest.approx(rev_t.sum() / ses_t.sum())
    assert r['control_mean'] != pytest.approx((rev_c / ses_c).mean(), rel=0.02)   # not the mean of ratios
    assert r['significant']
    assert r['relative_lift'] == pytest.approx(10.0, abs=4)
    assert r['lift_ci_lower'] < r['relative_lift'] < r['lift_ci_upper']


@pytest.mark.slow
@pytest.mark.statistical
def test_ratio_test_holds_its_false_positive_rate_and_coverage(engine):
    rng = np.random.default_rng(1)
    n_sims, hits = 600, 0
    for _ in range(n_sims):
        hits += engine.ratio_metric_test(*ratio_arm(rng, 400), *ratio_arm(rng, 400))['significant']
    assert hits / n_sims == pytest.approx(0.05, abs=0.025)


@pytest.mark.unit
def test_ratio_test_validates_input(engine):
    with pytest.raises(ValueError, match="one value per unit"):
        engine.ratio_metric_test([1, 2, 3], [1, 2], [1, 2], [1, 2])
    with pytest.raises(ValueError, match="positive"):
        engine.ratio_metric_test([1, 2], [0, 0], [1, 2], [1, 2])


@pytest.mark.unit
def test_per_column_aggregation_sums_numerator_and_denominator():
    df = pd.DataFrame({
        'user_id': [1, 1, 2], 'variant': ['a', 'a', 'b'],
        'revenue': [10.0, 30.0, 5.0], 'sessions': [1, 1, 1], 'latency': [100.0, 300.0, 50.0],
    })
    unit_df, _ = to_unit_level(df, 'user_id', 'variant', ['revenue', 'sessions', 'latency'],
                               agg={'revenue': 'sum', 'sessions': 'sum'})
    row = unit_df.set_index('user_id').loc[1]
    assert (row.revenue, row.sessions, row.latency) == (40.0, 2, 200.0)   # latency falls back to mean


# =============== BOUNDED BOOTSTRAP ===============

@pytest.mark.slow
def test_bootstrap_stays_fast_and_flat_in_memory_at_100k_per_arm(engine):
    """Used to take 173 s and ~935 MiB at this size"""
    rng = np.random.default_rng(2)
    control, treatment = rng.exponential(10, 100_000), rng.exponential(10.2, 100_000)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.time()
    r = engine.bootstrap_test(control, treatment)
    elapsed = time.time() - started
    grown = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
    grown_mib = grown / 1024 / (1024 if os.uname().sysname == 'Darwin' else 1)
    
    assert elapsed < 30
    assert grown_mib < 200
    assert r['n_resamples'] >= 1000
    welch = engine.t_test(control, treatment)
    assert r['ci_lower'] == pytest.approx(welch['ci_lower'], abs=0.02)
    assert r['ci_upper'] == pytest.approx(welch['ci_upper'], abs=0.02)


@pytest.mark.unit
def test_bootstrap_refuses_sizes_it_cannot_serve(engine):
    big = np.zeros(BOOTSTRAP_MAX_UNITS // 2 + 1)
    with pytest.raises(ValueError, match="Welch"):
        engine.bootstrap_test(big, big)


@pytest.mark.unit
def test_auto_does_not_recommend_bootstrap_for_large_heavy_tailed_metrics():
    rng = np.random.default_rng(3)
    small = pd.Series(np.where(rng.random(5_000) < 0.9, 0, rng.lognormal(4, 1.2, 5_000)))
    large = pd.Series(np.where(rng.random(150_000) < 0.9, 0, rng.lognormal(4, 1.2, 150_000)))
    assert describe_metric(small)['recommended_test'] == 'Bootstrap'
    assert describe_metric(large)['recommended_test'] == 'T-Test'
    assert "heavy-tailed" in describe_metric(large)['reason']


# =============== TIME-BASED HEALTH ===============

def daily_frame(rng, days=14, per_day=600, broken_day=None):
    frames = []
    for d in range(days):
        share = 0.80 if d == broken_day else 0.50
        frames.append(pd.DataFrame({
            'variant': np.where(rng.random(per_day) < share, 'control', 'treatment'),
            'exposed_at': pd.Timestamp('2026-03-02') + pd.Timedelta(days=d) + pd.to_timedelta(rng.integers(0, 1440, per_day), unit='m'),
            'y': rng.normal(size=per_day),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.mark.unit
def test_clean_two_week_experiment_passes_time_checks():
    result = HealthChecker().check_over_time(daily_frame(np.random.default_rng(4)), 'variant', 'exposed_at')
    assert result['n_days'] == 14 and result['covers_whole_weeks']
    assert result['days_with_srm'] == []


@pytest.mark.unit
def test_one_broken_day_is_caught_and_invalidates_the_decision(engine):
    """600 of 8,400 units on a bad day: the overall split can look passable while that day is badly off"""
    rng = np.random.default_rng(5)
    df = daily_frame(rng, broken_day=6)
    checker = HealthChecker()
    health = checker.run_all_checks(df, 'variant', 'y', time_col='exposed_at')
    
    assert [d['day'] for d in health['over_time']['days_with_srm']] == ['2026-03-08']
    assert health['overall_health'] is False
    
    primary = engine.t_test(rng.normal(100, 10, 3000), rng.normal(105, 10, 3000))
    decision = ship_decision(primary, health={**health, 'sample_ratio_mismatch': {'has_srm': False}})
    assert decision['decision'] == INVALID
    assert "2026-03-08" in decision['reasons'][0]


@pytest.mark.unit
@pytest.mark.parametrize("days,fragment", [(3, "only 3 day"), (10, "not a whole number of weeks")])
def test_short_or_partial_week_experiments_are_flagged(days, fragment):
    checker = HealthChecker()
    checker.check_over_time(daily_frame(np.random.default_rng(6), days=days), 'variant', 'exposed_at')
    assert any(fragment in w for w in checker.warnings)


@pytest.mark.unit
def test_unreadable_timestamps_are_reported_not_ignored():
    df = pd.DataFrame({'variant': ['a', 'b'] * 50, 'when': ['not a date'] * 100, 'y': np.arange(100.0)})
    health = HealthChecker().run_all_checks(df, 'variant', 'y', time_col='when')
    assert health['overall_health'] is False
    assert any("Over-time check could not run" in w for w in health['warnings'])


@pytest.mark.unit
def test_time_column_guess_and_first_exposure_carried_through_aggregation():
    df = pd.DataFrame({
        'user_id': [1, 1, 2], 'variant': ['a', 'a', 'b'], 'y': [1.0, 3.0, 5.0],
        'exposed_at': ['2026-03-05 10:00', '2026-03-02 09:00', '2026-03-03 12:00'],
    })
    assert guess_time_column(df, exclude=('user_id', 'variant')) == 'exposed_at'
    unit_df, _ = to_unit_level(df, 'user_id', 'variant', ['y'], order_col='exposed_at')
    assert unit_df.set_index('user_id').loc[1, 'exposed_at'] == '2026-03-02 09:00'


@pytest.mark.unit
def test_numeric_duration_columns_are_never_guessed_as_timestamps():
    df = pd.DataFrame({
        'time_on_site_seconds': [10, 20, 30], 'date_bucket': [1, 2, 3],
        'variant': ['control', 'treatment', 'control'], 'country': ['US', 'IN', 'US'],
    })
    assert guess_time_column(df) is None


# =============== EXPORT LIMIT ===============

@pytest.mark.unit
def test_excel_export_omits_oversized_raw_data():
    big = pd.DataFrame({'x': np.zeros(RAW_DATA_ROW_LIMIT + 1)})
    workbook = ReportGenerator().create_excel_report({'p_value': 0.5}, big, 't-test')
    sheets = pd.read_excel(io.BytesIO(workbook.getvalue()), sheet_name=None)
    assert "Raw data omitted" in sheets['Raw Data'].iloc[0, 0]


# =============== APP ===============

def load_app(sample):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.sidebar.selectbox[0].select(sample).run()
    return at


def pick(at, label, value):
    next(s for s in at.selectbox if s.label == label).select(value)
    return at


def run(at):
    next(b for b in at.button if "Run A/B Test" in b.label).click()
    return at.run()


@pytest.mark.integration
def test_app_guesses_the_timestamp_and_reports_time_checks():
    at = load_app(MULTI)
    assert next(s for s in at.selectbox if s.label.startswith("Exposure Timestamp")).value == "exposed_at"
    assert "exposed_at" not in next(s for s in at.selectbox if s.label == "Primary Metric").options
    
    pick(at, "Primary Metric", "revenue").run()
    at = run(at)
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "Traffic split is stable across all 14 days" in body
    assert "Experiment covers 2 full week(s)" in body


@pytest.mark.integration
def test_app_ratio_metric_flow():
    at = load_app(MULTI)
    pick(at, "Primary Metric", "revenue").run()
    next(c for c in at.checkbox if c.label.startswith("This metric is a ratio")).check().run()
    pick(at, "Denominator Column", "sessions").run()
    
    summary = next(i.value for i in at.info if "as it will be analyzed" in i.value)
    assert "ratio of totals of **revenue / sessions**" in summary
    assert "Ratio Metric (delta method)" in summary
    
    at = run(at)
    assert not at.exception
    df = pd.read_csv(os.path.join(os.path.dirname(APP), "data", "multivariant_checkout_test.csv"))
    control = df[df.variant == 'control']
    shown = {m.label: m.value for m in at.metric}
    assert shown["Control Mean"] == f"{control.revenue.sum() / control.sessions.sum():.4f}"


@pytest.mark.integration
def test_sequential_without_a_timestamp_warns_about_row_order():
    at = load_app("A/B test (revenue, conversion)")
    pick(at, "Primary Metric", "revenue")
    pick(at, "Statistical Test", "Sequential (always-valid)").run()
    assert any("row order is assumed" in w.value for w in at.warning)
    
    at = load_app(MULTI)
    pick(at, "Primary Metric", "revenue")
    pick(at, "Statistical Test", "Sequential (always-valid)").run()
    assert not any("row order is assumed" in w.value for w in at.warning)
