"""
Tests for the warehouse path: tests from summary statistics, and the SQL that produces them

The end-to-end tests load the sample data into DuckDB, run the generated SQL
there, feed the handful of aggregate rows to the from-stats methods, and
require the same answer as the row-level analysis of the same file.
"""

import os

import numpy as np
import pandas as pd
import pytest

from modules.ab_testing import ABTestingEngine
from modules.experiment_design import to_unit_level
from modules.from_stats import srm_from_counts
from modules.health_checks import HealthChecker
from modules.sql_templates import aggregate_sql, rows_to_arms, DIALECTS

duckdb = pytest.importorskip("duckdb")

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SAME = ('p_value', 'mean_difference', 'ci_lower', 'ci_upper', 'relative_lift',
        'lift_ci_lower', 'lift_ci_upper', 'control_mean', 'treatment_mean')


@pytest.fixture
def engine():
    return ABTestingEngine()


def moments(y, x=None):
    out = {'n': len(y), 'mean': float(np.mean(y)), 'var': float(np.var(y, ddof=1))}
    if x is not None:
        out.update(mean_x=float(np.mean(x)), var_x=float(np.var(x, ddof=1)), cov=float(np.cov(y, x, ddof=1)[0, 1]))
    return out


def assert_same(a, b, keys=SAME):
    for key in keys:
        if key in a or key in b:
            assert a[key] == pytest.approx(b[key], rel=1e-9, abs=1e-12), key
    assert a['significant'] == b['significant']


# =============== FROM STATS == FROM ROWS ===============

@pytest.mark.unit
@pytest.mark.parametrize("equal_var", [False, True])
def test_t_test_from_stats_matches_rows(engine, equal_var):
    rng = np.random.default_rng(0)
    c, t = rng.normal(100, 10, 5000), rng.normal(100.6, 14, 3000)
    assert_same(engine.t_test_from_stats(moments(c), moments(t), equal_var=equal_var),
                engine.t_test(c, t, equal_var=equal_var),
                keys=SAME + ('t_statistic', 'degrees_of_freedom', 'cohens_d'))


@pytest.mark.unit
def test_cuped_from_stats_matches_rows(engine):
    rng = np.random.default_rng(1)
    xc, xt = rng.normal(100, 20, 4000), rng.normal(100, 20, 4200)
    yc, yt = 0.9 * xc + rng.normal(0, 8, 4000), 0.9 * xt + rng.normal(1.0, 8, 4200)
    assert_same(engine.cuped_from_stats(moments(yc, xc), moments(yt, xt)), engine.cuped_test(yc, yt, xc, xt),
                keys=SAME + ('theta', 'covariate_correlation', 'variance_reduction_pct', 'unadjusted_p_value'))


@pytest.mark.unit
def test_ratio_from_stats_matches_rows(engine):
    rng = np.random.default_rng(2)
    dc, dt = rng.integers(1, 30, 2000).astype(float), rng.integers(1, 30, 2000).astype(float)
    nc, nt = dc * rng.exponential(5, 2000), dt * rng.exponential(5.4, 2000)
    assert_same(engine.ratio_metric_from_stats(moments(nc, dc), moments(nt, dt)),
                engine.ratio_metric_test(nc, dc, nt, dt), keys=SAME + ('z_statistic',))


@pytest.mark.unit
def test_multi_variant_from_stats_matches_rows(engine):
    rng = np.random.default_rng(3)
    groups = {'control': rng.normal(100, 10, 800), 'a': rng.normal(101, 10, 800), 'b': rng.normal(103, 10, 800)}
    from_stats = engine.multi_variant_from_stats({k: moments(v) for k, v in groups.items()}, 'control')
    from_rows = engine.multi_variant_test(groups, 'control')
    
    assert from_stats['best_variant'] == from_rows['best_variant'] == 'b'
    for s, r in zip(from_stats['comparisons'], from_rows['comparisons']):
        assert s['group'] == r['group']
        assert s['p_value_adjusted'] == pytest.approx(r['p_value_adjusted'], rel=1e-9)


@pytest.mark.unit
def test_srm_from_counts_matches_the_row_level_check():
    df = pd.DataFrame({'g': ['control'] * 1000 + ['treatment'] * 800})
    row_level = HealthChecker().check_sample_ratio_mismatch(df, 'g')
    from_counts = srm_from_counts({'control': 1000, 'treatment': 800})
    assert from_counts['p_value'] == pytest.approx(row_level['p_value'])
    assert from_counts['has_srm'] == row_level['has_srm'] is True
    assert srm_from_counts({'control': 900, 'treatment': 100}, {'control': 0.9, 'treatment': 0.1})['has_srm'] is False


@pytest.mark.unit
def test_bad_statistics_are_rejected(engine):
    good = {'n': 100, 'mean': 1.0, 'var': 2.0}
    for bad in ({'n': 100, 'mean': 1.0}, {'n': 1, 'mean': 1.0, 'var': 2.0}, {'n': 100, 'mean': 1.0, 'var': -1.0},
                {'n': 100, 'mean': float('nan'), 'var': 2.0}, {'n': 10.5, 'mean': 1.0, 'var': 2.0}):
        with pytest.raises(ValueError):
            engine.t_test_from_stats(good, bad)
    with pytest.raises(ValueError, match="mean_x"):
        engine.cuped_from_stats(good, good)


@pytest.mark.unit
def test_constant_arms_from_stats(engine):
    r = engine.t_test_from_stats({'n': 5, 'mean': 0.0, 'var': 0.0}, {'n': 5, 'mean': 1.0, 'var': 0.0})
    assert r['p_value'] == 0.0 and r['t_statistic'] is None and r['relative_lift'] is None


# =============== SQL: VALIDATION ===============

@pytest.mark.unit
@pytest.mark.parametrize("kwargs", [
    {'table': 'events; DROP TABLE users'}, {'table': 'a.b.c.d'}, {'table': 'events e'},
    {'metric_col': 'revenue) FROM x --'}, {'assignment_col': 'variant"'}, {'unit_col': 'user id'},
    {'dialect': 'oracle'}, {'analysis': 'median'}, {'unit_agg': 'median'},
    {'analysis': 'cuped'}, {'start_date': '2026-03-02'},
    {'time_col': 'ts', 'start_date': "2026-03-02' OR '1'='1"},
])
def test_sql_builder_rejects_anything_that_is_not_a_plain_identifier(kwargs):
    base = dict(dialect='postgres', table='events', assignment_col='variant', metric_col='revenue')
    with pytest.raises(ValueError):
        aggregate_sql(**{**base, **kwargs})


@pytest.mark.unit
def test_dialects_differ_only_where_they_must():
    built = {d: aggregate_sql(d, 'proj-1.ds.events', 'variant', 'revenue', unit_col='user_id')['sql'] for d in DIALECTS}
    assert "`proj-1.ds.events`" in built['bigquery'] and "FLOAT64" in built['bigquery']
    assert '"proj-1"."ds"."events"' in built['snowflake'] and "DOUBLE)" in built['snowflake']
    assert "DOUBLE PRECISION" in built['postgres']
    for sql in built.values():
        assert "COUNT(DISTINCT" in sql and "VAR_SAMP" in sql and sql.count(";") == 0


# =============== SQL: END TO END ON DUCKDB ===============

@pytest.fixture
def warehouse():
    con = duckdb.connect()
    con.execute(f"CREATE TABLE checkout AS SELECT * FROM read_csv_auto('{DATA}/multivariant_checkout_test.csv')")
    
    # A session-level table: users randomized, several rows each, a few users leaking across arms
    rng = np.random.default_rng(4)
    users = 1500
    sessions = rng.integers(1, 8, users)
    frame = pd.DataFrame({
        'user_id': np.repeat(np.arange(users), sessions),
        'variant': np.repeat(rng.choice(['control', 'treatment'], users), sessions),
    })
    frame['spend'] = np.repeat(rng.normal(50, 15, users), sessions) + rng.normal(0, 3, len(frame))
    frame.loc[frame.user_id < 20, 'variant'] = np.where(np.arange((frame.user_id < 20).sum()) % 2, 'control', 'treatment')
    con.register('sessions_frame', frame)
    con.execute("CREATE TABLE sessions AS SELECT * FROM sessions_frame")
    yield con, frame
    con.close()


def query(con, **kwargs):
    built = aggregate_sql('duckdb', **kwargs)
    return rows_to_arms(con.execute(built['sql']).df().to_dict('records'))


@pytest.mark.integration
def test_mean_analysis_in_the_warehouse_matches_the_row_level_result(engine, warehouse):
    con, _ = warehouse
    arms = query(con, table='checkout', assignment_col='variant', metric_col='revenue')
    df = pd.read_csv(f"{DATA}/multivariant_checkout_test.csv")
    rows = engine.t_test(df[df.variant == 'control'].revenue.values, df[df.variant == 'express_pay'].revenue.values)
    assert_same(engine.t_test_from_stats(arms['control'], arms['express_pay']), rows)
    
    mv = engine.multi_variant_from_stats(arms, 'control')
    assert mv['best_variant'] == 'express_pay'


@pytest.mark.integration
def test_cuped_and_ratio_in_the_warehouse_match_rows(engine, warehouse):
    con, _ = warehouse
    df = pd.read_csv(f"{DATA}/multivariant_checkout_test.csv")
    c, t = df[df.variant == 'control'], df[df.variant == 'one_page_checkout']
    
    arms = query(con, table='checkout', assignment_col='variant', metric_col='revenue',
                 analysis='cuped', second_col='pre_revenue')
    assert_same(engine.cuped_from_stats(arms['control'], arms['one_page_checkout']),
                engine.cuped_test(c.revenue.values, t.revenue.values, c.pre_revenue.values, t.pre_revenue.values))
    
    arms = query(con, table='checkout', assignment_col='variant', metric_col='revenue',
                 analysis='ratio', second_col='sessions', unit_col='user_id')
    assert_same(engine.ratio_metric_from_stats(arms['control'], arms['one_page_checkout']),
                engine.ratio_metric_test(c.revenue.values, c.sessions.values, t.revenue.values, t.sessions.values))


@pytest.mark.integration
def test_proportion_analysis_in_the_warehouse(engine, warehouse):
    con, _ = warehouse
    arms = query(con, table='checkout', assignment_col='variant', metric_col='converted', analysis='proportion')
    df = pd.read_csv(f"{DATA}/multivariant_checkout_test.csv")
    control = df[df.variant == 'control']
    assert arms['control'] == {'n': len(control), 'successes': control.converted.sum()}


@pytest.mark.integration
def test_unit_collapse_and_contamination_drop_happen_in_sql(engine, warehouse):
    """The SQL must do what to_unit_level does: one row per user, leaky users removed"""
    con, frame = warehouse
    arms = query(con, table='sessions', assignment_col='variant', metric_col='spend', unit_col='user_id')
    
    unit_df, info = to_unit_level(frame, 'user_id', 'variant', ['spend'])
    assert info['n_contaminated_units'] > 0
    assert arms['control']['n'] + arms['treatment']['n'] == info['n_units']
    rows = engine.t_test(unit_df[unit_df.variant == 'control'].spend.values,
                         unit_df[unit_df.variant == 'treatment'].spend.values)
    assert_same(engine.t_test_from_stats(arms['control'], arms['treatment']), rows)


@pytest.mark.integration
def test_date_window_filters_in_sql(warehouse):
    con, _ = warehouse
    full = query(con, table='checkout', assignment_col='variant', metric_col='revenue')
    week = query(con, table='checkout', assignment_col='variant', metric_col='revenue',
                 time_col='exposed_at', start_date='2026-03-02', end_date='2026-03-08')
    df = pd.read_csv(f"{DATA}/multivariant_checkout_test.csv", parse_dates=['exposed_at'])
    expected = ((df.variant == 'control') & (df.exposed_at.dt.date <= pd.Timestamp('2026-03-08').date())).sum()
    assert week['control']['n'] == expected < full['control']['n']
