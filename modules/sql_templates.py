"""
Warehouse SQL Templates
Generate the one aggregate query whose result feeds the from-stats tests

The query returns one row per arm. Row-level data stays in the warehouse; only
counts, means, variances and covariances come back.

Identifiers are validated rather than escaped, and no free-text SQL is
accepted, so a caller (including an LLM agent) cannot smuggle a statement in.
"""

import re
from datetime import date
from typing import Dict, Optional

DIALECTS = ('bigquery', 'snowflake', 'postgres', 'duckdb')
ANALYSES = ('mean', 'proportion', 'cuped', 'ratio')

_COLUMN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_TABLE = re.compile(r'^[A-Za-z_][A-Za-z0-9_\-]*(\.[A-Za-z_][A-Za-z0-9_\-]*){0,2}$')

_FLOAT = {'bigquery': 'FLOAT64', 'snowflake': 'DOUBLE', 'postgres': 'DOUBLE PRECISION', 'duckdb': 'DOUBLE'}


def _column(name: str, role: str) -> str:
    if not isinstance(name, str) or not _COLUMN.match(name):
        raise ValueError(f"{role} '{name}' is not a plain column name (letters, digits, underscore)")
    return name


def _quote(identifier: str, dialect: str) -> str:
    mark = '`' if dialect == 'bigquery' else '"'
    if dialect == 'bigquery':
        return f"{mark}{identifier}{mark}"             # BigQuery quotes the whole dotted path
    return ".".join(f"{mark}{part}{mark}" for part in identifier.split('.'))


def _iso(value: str, role: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError):
        raise ValueError(f"{role} must be an ISO date like 2026-03-02")


def aggregate_sql(
    dialect: str,
    table: str,
    assignment_col: str,
    metric_col: str,
    analysis: str = 'mean',
    unit_col: Optional[str] = None,
    second_col: Optional[str] = None,
    unit_agg: str = 'mean',
    time_col: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict:
    """
    Build the aggregate query for one metric
    
    Args:
        dialect: 'bigquery', 'snowflake', 'postgres', or 'duckdb'
        table: Table name, optionally qualified (project.dataset.table)
        assignment_col: Arm assignment column
        metric_col: Metric column (the numerator for a ratio)
        analysis: 'mean', 'proportion', 'cuped', or 'ratio'
        unit_col: Randomization unit id. When given, rows are first collapsed to one
            per unit, and units that appear in more than one arm are dropped.
        second_col: The pre-experiment covariate for 'cuped', the denominator for 'ratio'
        unit_agg: How to combine a unit's rows for the metric: 'mean', 'sum', or 'max'.
            Ratio analyses always sum both columns.
        time_col: Optional exposure timestamp, used only with start_date / end_date
        start_date: Inclusive ISO date filter on time_col
        end_date: Inclusive ISO date filter on time_col
        
    Returns:
        {'sql': ..., 'columns': what each output column means, 'feeds': which engine method takes it}
    """
    if dialect not in DIALECTS:
        raise ValueError(f"dialect must be one of {DIALECTS}")
    if analysis not in ANALYSES:
        raise ValueError(f"analysis must be one of {ANALYSES}")
    if unit_agg not in ('mean', 'sum', 'max'):
        raise ValueError("unit_agg must be 'mean', 'sum', or 'max'")
    if not isinstance(table, str) or not _TABLE.match(table):
        raise ValueError(f"table '{table}' is not a plain (optionally dotted) table name")
    needs_second = analysis in ('cuped', 'ratio')
    if needs_second and not second_col:
        raise ValueError(f"analysis '{analysis}' needs second_col "
                         f"({'the covariate' if analysis == 'cuped' else 'the denominator'})")
    if (start_date or end_date) and not time_col:
        raise ValueError("start_date / end_date need time_col")
    
    q = lambda name: _quote(name, dialect)
    arm, y = q(_column(assignment_col, 'assignment_col')), q(_column(metric_col, 'metric_col'))
    x = q(_column(second_col, 'second_col')) if needs_second else None
    unit = q(_column(unit_col, 'unit_col')) if unit_col else None
    flt = _FLOAT[dialect]
    
    filters = [f"{arm} IS NOT NULL", f"{y} IS NOT NULL"] + ([f"{x} IS NOT NULL"] if x else [])
    if time_col:
        t = q(_column(time_col, 'time_col'))
        if start_date:
            filters.append(f"CAST({t} AS DATE) >= DATE '{_iso(start_date, 'start_date')}'")
        if end_date:
            filters.append(f"CAST({t} AS DATE) <= DATE '{_iso(end_date, 'end_date')}'")
    where = "\n    AND ".join(filters)
    
    agg = {'mean': 'AVG', 'sum': 'SUM', 'max': 'MAX'}['sum' if analysis == 'ratio' else unit_agg]
    x_agg = 'SUM' if analysis == 'ratio' else 'AVG'     # a covariate is a unit attribute; average its rows
    
    if unit:
        second = f",\n    {x_agg}(CAST({x} AS {flt})) AS x" if x else ""
        source = (
            f"units AS (\n"
            f"  SELECT\n    {unit} AS unit_id,\n    MIN({arm}) AS arm,\n"
            f"    {agg}(CAST({y} AS {flt})) AS y{second}\n"
            f"  FROM {q(table)}\n  WHERE {where}\n"
            f"  GROUP BY {unit}\n"
            f"  HAVING COUNT(DISTINCT {arm}) = 1   -- drop units that saw more than one arm\n)"
        )
    else:
        second = f",\n    CAST({x} AS {flt}) AS x" if x else ""
        source = (
            f"units AS (\n  SELECT\n    {arm} AS arm,\n    CAST({y} AS {flt}) AS y{second}\n"
            f"  FROM {q(table)}\n  WHERE {where}\n)"
        )
    
    if analysis == 'proportion':
        select = "  COUNT(*) AS n,\n  SUM(CASE WHEN y > 0 THEN 1 ELSE 0 END) AS successes"
        columns = {'arm': 'arm label', 'n': 'units in the arm', 'successes': 'units with a positive metric'}
        feeds = 'proportions_test / multi_variant_from_stats(metric_type="proportion")'
    else:
        select = "  COUNT(*) AS n,\n  AVG(y) AS mean,\n  VAR_SAMP(y) AS var"
        columns = {'arm': 'arm label', 'n': 'units in the arm', 'mean': 'mean of the metric',
                   'var': 'sample variance of the metric'}
        feeds = 't_test_from_stats / multi_variant_from_stats'
        if x:
            select += ",\n  AVG(x) AS mean_x,\n  VAR_SAMP(x) AS var_x,\n  COVAR_SAMP(y, x) AS cov"
            other = 'covariate' if analysis == 'cuped' else 'denominator'
            columns.update({'mean_x': f'mean of the {other}', 'var_x': f'sample variance of the {other}',
                            'cov': f'sample covariance of metric and {other}'})
            feeds = 'cuped_from_stats' if analysis == 'cuped' else 'ratio_metric_from_stats'
    
    sql = f"WITH {source}\nSELECT\n  arm,\n{select}\nFROM units\nGROUP BY arm\nORDER BY arm"
    return {'sql': sql, 'columns': columns, 'feeds': feeds, 'dialect': dialect}


def rows_to_arms(rows) -> Dict[str, Dict]:
    """Turn the query result (a list of dict rows, one per arm) into {arm label: stats}"""
    arms = {}
    for row in rows:
        row = {str(k).lower(): v for k, v in dict(row).items()}
        if 'arm' not in row:
            raise ValueError("Each row needs an 'arm' column")
        label = str(row.pop('arm'))
        arms[label] = {k: (float(v) if v is not None else None) for k, v in row.items()}
    if len(arms) < 2:
        raise ValueError("Need at least two arms")
    return arms
