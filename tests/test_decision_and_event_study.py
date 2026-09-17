"""
Tests for the ship decision and the event-study pre-trend test
"""

import numpy as np
import pandas as pd
import pytest

from modules.ab_testing import ABTestingEngine
from modules.causal_inference import CausalInferenceLab
from modules.health_checks import HealthChecker
from utils.decision import (
    ship_decision, SHIP, DO_NOT_SHIP, KEEP_RUNNING, STOP_NO_EFFECT, INVALID
)


@pytest.fixture
def engine():
    return ABTestingEngine()


def result(engine, rng, control_mean, treatment_mean, n=3000, sd=10):
    return engine.t_test(rng.normal(control_mean, sd, n), rng.normal(treatment_mean, sd, n))


# =============== SHIP DECISION ===============

@pytest.mark.unit
def test_ship_when_primary_wins_and_guardrails_hold(engine):
    rng = np.random.default_rng(0)
    decision = ship_decision(
        result(engine, rng, 100, 104),
        guardrails={'latency': result(engine, rng, 800, 800)},
        guardrail_higher_is_better={'latency': False},
    )
    assert decision['decision'] == SHIP


@pytest.mark.unit
def test_guardrail_regression_blocks_a_winning_primary(engine):
    rng = np.random.default_rng(1)
    decision = ship_decision(
        result(engine, rng, 100, 104),
        guardrails={'latency': result(engine, rng, 800, 830)},
        guardrail_higher_is_better={'latency': False},
    )
    assert decision['decision'] == DO_NOT_SHIP
    assert decision['harmed_guardrails'] == ['latency']


@pytest.mark.unit
def test_lower_is_better_guardrail_improving_is_not_harm(engine):
    rng = np.random.default_rng(2)
    decision = ship_decision(
        result(engine, rng, 100, 104),
        guardrails={'latency': result(engine, rng, 800, 770)},
        guardrail_higher_is_better={'latency': False},
    )
    assert decision['decision'] == SHIP


@pytest.mark.unit
def test_significantly_worse_primary_is_not_shipped(engine):
    decision = ship_decision(result(engine, np.random.default_rng(3), 100, 96))
    assert decision['decision'] == DO_NOT_SHIP


@pytest.mark.unit
def test_lower_is_better_primary(engine):
    decision = ship_decision(result(engine, np.random.default_rng(4), 100, 96), higher_is_better=False)
    assert decision['decision'] == SHIP


@pytest.mark.unit
def test_underpowered_null_keeps_running(engine):
    r = result(engine, np.random.default_rng(5), 100, 100, n=60, sd=30)
    assert not r['significant']
    assert ship_decision(r, mde_pct=2.0)['decision'] == KEEP_RUNNING


@pytest.mark.unit
def test_precise_null_stops_for_futility(engine):
    r = result(engine, np.random.default_rng(6), 100, 100, n=200000, sd=10)
    assert not r['significant']
    assert ship_decision(r, mde_pct=2.0)['decision'] == STOP_NO_EFFECT


@pytest.mark.unit
def test_srm_overrides_everything(engine):
    rng = np.random.default_rng(7)
    df = pd.DataFrame({'group': ['control'] * 1000 + ['treatment'] * 700,
                       'metric': rng.normal(size=1700)})
    health = HealthChecker().run_all_checks(df, 'group', 'metric')
    decision = ship_decision(result(engine, rng, 100, 110), health=health)
    assert decision['decision'] == INVALID


# =============== EVENT STUDY ===============

def quarterly_panel(rng, diverging_pre_trend=0.0, effect=20.0, n_stores=200):
    rows = []
    periods = [(y, q) for y in (2023, 2024) for q in (1, 2, 3, 4)]
    for store in range(n_stores):
        treated = store < n_stores // 2
        store_effect = rng.normal(0, 10)
        for i, (year, quarter) in enumerate(periods):
            post = year == 2024
            rows.append({
                'store': store,
                'region': 'treatment' if treated else 'control',
                'year': year,
                'quarter': quarter,
                'sales': 500 + 50 * treated + 5 * i + store_effect
                         + diverging_pre_trend * i * treated
                         + effect * (treated and post) + rng.normal(0, 5),
            })
    return pd.DataFrame(rows)


@pytest.mark.statistical
def test_event_study_passes_parallel_trends_and_recovers_effect():
    df = quarterly_panel(np.random.default_rng(0))
    r = CausalInferenceLab().event_study(
        df, 'region', ['year', 'quarter'], 'sales', 'treatment', (2024, 1), cluster_col='store'
    )
    assert r['parallel_trends_assumption'] is True
    assert r['n_pre_periods_tested'] == 3
    assert r['average_post_effect'] == pytest.approx(20.0, abs=1.5)
    assert r['reference_period'] == '2023 / 4'
    pre = r['coefficients'][r['coefficients'].event_time < 0]
    assert ((pre.ci_lower < 0) & (pre.ci_upper > 0)).all()


@pytest.mark.statistical
def test_event_study_catches_diverging_pre_trends():
    df = quarterly_panel(np.random.default_rng(1), diverging_pre_trend=4.0)
    r = CausalInferenceLab().event_study(
        df, 'region', ['year', 'quarter'], 'sales', 'treatment', (2024, 1), cluster_col='store'
    )
    assert r['parallel_trends_assumption'] is False
    assert r['pre_trend_p_value'] < 0.001


@pytest.mark.unit
def test_event_study_needs_two_pre_periods():
    df = quarterly_panel(np.random.default_rng(2), n_stores=20)
    with pytest.raises(ValueError, match="two pre-treatment periods"):
        CausalInferenceLab().event_study(df, 'region', ['year', 'quarter'], 'sales', 'treatment', (2023, 2))
