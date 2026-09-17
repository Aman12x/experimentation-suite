"""
Tests for the experiment design helpers
"""

import numpy as np
import pandas as pd
import pytest

from modules.ab_testing import ABTestingEngine
from modules.experiment_design import (
    guess_unit_column, guess_assignment_column, guess_control_label,
    metric_candidates, describe_metric, to_unit_level, experiment_summary
)


@pytest.fixture
def session_level():
    """Users are randomized, but the table has one row per session.
    Heavy users have more sessions AND higher spend per session, so rows are correlated."""
    rng = np.random.default_rng(0)
    rows = []
    for user in range(600):
        arm = 'treatment' if user % 2 else 'control'
        user_level = rng.normal(50, 15)
        for _ in range(rng.integers(1, 12)):
            rows.append({'user_id': user, 'variant': arm, 'spend': user_level + rng.normal(0, 3)})
    return pd.DataFrame(rows)


# =============== GUESSING ===============

@pytest.mark.unit
def test_guesses_on_a_typical_table():
    df = pd.DataFrame({
        'order_id': range(6), 'user_id': [1, 1, 2, 2, 3, 3],
        'variant': ['control', 'treatment'] * 3, 'revenue': np.arange(6.0), 'converted': [0, 1] * 3,
    })
    assert guess_unit_column(df) == 'user_id'            # user-level id beats order_id
    assert guess_assignment_column(df, exclude='user_id') == 'variant'
    assert metric_candidates(df, 'user_id', 'variant') == ['revenue', 'converted']


@pytest.mark.unit
def test_no_id_column_means_no_unit_guess():
    assert guess_unit_column(pd.DataFrame({'variant': ['a', 'b'], 'y': [1.0, 2.0]})) is None


@pytest.mark.unit
@pytest.mark.parametrize("labels,expected", [
    (['treatment', 'control'], 1), (['B', 'A'], 1), (['new', 'old'], 0), ([1, 0], 1),
])
def test_control_label_guess(labels, expected):
    assert guess_control_label(labels) == expected


# =============== METRIC DESCRIPTION ===============

@pytest.mark.unit
def test_binary_metric_recommends_proportions():
    info = describe_metric(pd.Series([0, 1, 0, 0, 1, np.nan]))
    assert info['kind'] == 'binary'
    assert info['recommended_test'] == 'Proportions Z-Test'
    assert "rate" in info['summary']


@pytest.mark.unit
def test_well_behaved_metric_recommends_welch():
    info = describe_metric(pd.Series(np.random.default_rng(1).normal(100, 10, 2000)))
    assert info['kind'] == 'continuous'
    assert info['recommended_test'] == 'T-Test'


@pytest.mark.unit
def test_heavy_tailed_metric_recommends_bootstrap():
    rng = np.random.default_rng(2)
    revenue = np.where(rng.random(5000) < 0.9, 0, rng.lognormal(4, 1.2, 5000)).round()
    info = describe_metric(pd.Series(revenue))
    assert info['kind'] == 'count'
    assert info['recommended_test'] == 'Bootstrap'
    assert info['zero_share'] > 0.5


@pytest.mark.unit
def test_recommended_tests_exist_in_the_app():
    import ast, pathlib
    source = pathlib.Path(__file__).parent.parent / 'app.py'
    tree = ast.parse(source.read_text())
    test_types = next(
        ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'TEST_TYPES'
    )
    for series in (pd.Series([0, 1]), pd.Series([1.5, 2.5, 3.5]), pd.Series([0] * 50 + [1000])):
        assert describe_metric(series)['recommended_test'] in test_types


# =============== UNIT-LEVEL AGGREGATION ===============

@pytest.mark.unit
def test_one_row_per_unit_passes_through(sample_ab_data):
    unit_df, info = to_unit_level(sample_ab_data, 'user_id', 'group', ['metric'])
    assert info['aggregated'] is False
    assert info['n_units'] == len(sample_ab_data) == len(unit_df)


@pytest.mark.unit
def test_repeated_units_collapse_to_one_row(session_level):
    unit_df, info = to_unit_level(session_level, 'user_id', 'variant', ['spend'], agg='mean')
    assert info['aggregated'] is True
    assert info['n_units'] == 600 == unit_df['user_id'].nunique() == len(unit_df)
    assert info['max_rows_per_unit'] > 1
    
    user0 = session_level[session_level.user_id == 0].spend
    assert unit_df.set_index('user_id').loc[0, 'spend'] == pytest.approx(user0.mean())
    
    totals, _ = to_unit_level(session_level, 'user_id', 'variant', ['spend'], agg='sum')
    assert totals.set_index('user_id').loc[0, 'spend'] == pytest.approx(user0.sum())


@pytest.mark.unit
def test_units_seen_in_both_arms_are_dropped():
    df = pd.DataFrame({
        'user_id': [1, 1, 2, 3, 3], 'variant': ['control', 'treatment', 'control', 'treatment', 'treatment'],
        'y': [1.0, 2.0, 3.0, 4.0, 6.0],
    })
    unit_df, info = to_unit_level(df, 'user_id', 'variant', ['y'])
    assert info['n_contaminated_units'] == 1
    assert sorted(unit_df.user_id) == [2, 3]
    assert unit_df.set_index('user_id').loc[3, 'y'] == 5.0


@pytest.mark.unit
def test_no_unit_column_keeps_rows(session_level):
    unit_df, info = to_unit_level(session_level, None, 'variant', ['spend'])
    assert len(unit_df) == len(session_level)
    assert info['aggregated'] is False


@pytest.mark.slow
@pytest.mark.statistical
def test_row_level_analysis_inflates_false_positives_and_unit_level_does_not():
    """Why the unit of randomization matters: no true effect, users randomized,
    sessions analysed as if independent."""
    engine = ABTestingEngine()
    rng = np.random.default_rng(3)
    n_sims, row_hits, unit_hits = 400, 0, 0
    for _ in range(n_sims):
        users = np.arange(300)
        arm = rng.permutation(np.repeat(['control', 'treatment'], 150))
        level = rng.normal(50, 15, 300)
        sessions = rng.integers(1, 12, 300)
        df = pd.DataFrame({
            'user_id': np.repeat(users, sessions),
            'variant': np.repeat(arm, sessions),
            'spend': np.repeat(level, sessions) + rng.normal(0, 3, sessions.sum()),
        })
        row_hits += engine.t_test(
            df[df.variant == 'control'].spend.values, df[df.variant == 'treatment'].spend.values
        )['significant']
        unit_df, _ = to_unit_level(df, 'user_id', 'variant', ['spend'])
        unit_hits += engine.t_test(
            unit_df[unit_df.variant == 'control'].spend.values,
            unit_df[unit_df.variant == 'treatment'].spend.values
        )['significant']
    
    assert row_hits / n_sims > 0.25
    assert unit_hits / n_sims == pytest.approx(0.05, abs=0.03)


# =============== SUMMARY ===============

@pytest.mark.unit
def test_summary_reads_as_a_design_statement():
    text = experiment_summary(
        unit_label="user_id units", n_units=2000, arms={'control': 1000, 'treatment': 1000},
        control='control', metric='churned', metric_kind='binary', higher_is_better=False,
        mde_pct=5, test_name='Proportions Z-Test', alpha=0.05,
        guardrails={'latency_ms': False, 'retention': True},
        expected_split={'control': 0.5, 'treatment': 0.5},
    )
    assert "2,000 user_id units" in text
    assert "rate of **churned**" in text
    assert "**falls** by at least **5%**" in text
    assert "**latency_ms** must not rise" in text
    assert "**retention** must not fall" in text
    assert "50% / 50%" in text
