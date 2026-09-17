"""
Experiment Design Helpers
Turn a raw table into an explicit experiment definition: who was randomized,
what is being measured, and which test fits the metric
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from .ab_advanced import is_binary_metric

ROW_IS_UNIT = "(each row is one unit)"

CONTROL_LABELS = ('control', 'ctrl', 'baseline', 'holdout', 'a', '0', 'false')


def guess_unit_column(df: pd.DataFrame) -> Optional[str]:
    """Best guess at the randomization unit: an id-like column, preferring user-level ids"""
    candidates = [
        c for c in df.columns
        if c.lower() == 'id' or c.lower().endswith(('_id', 'id')) or 'uuid' in c.lower()
    ]
    if not candidates:
        return None
    preferred = ('user', 'customer', 'visitor', 'account', 'member', 'device', 'session', 'store', 'unit')
    candidates.sort(key=lambda c: next((i for i, p in enumerate(preferred) if p in c.lower()), len(preferred)))
    return candidates[0]


def guess_assignment_column(df: pd.DataFrame, exclude: Optional[str] = None) -> Optional[str]:
    """Best guess at the assignment column: few distinct values, named like an experiment arm"""
    names = ('variant', 'group', 'arm', 'treatment', 'bucket', 'cohort', 'experiment', 'assignment', 'test')
    low_cardinality = [
        c for c in df.columns
        if c != exclude and 2 <= df[c].nunique(dropna=True) <= 10
    ]
    named = [c for c in low_cardinality if any(n in c.lower() for n in names)]
    non_numeric = [c for c in low_cardinality if not pd.api.types.is_numeric_dtype(df[c])]
    for pool in (named, non_numeric, low_cardinality):
        if pool:
            return pool[0]
    return None


def guess_control_label(labels: List[Any]) -> int:
    """Index of the label that looks like a control arm (0 if none does)"""
    return next((i for i, g in enumerate(labels) if str(g).strip().lower() in CONTROL_LABELS), 0)


def metric_candidates(df: pd.DataFrame, unit_col: Optional[str], assignment_col: Optional[str]) -> List[str]:
    """Numeric columns that can be a metric: not the unit id, not the assignment, not another id"""
    numeric = df.select_dtypes(include=[np.number]).columns
    return [
        c for c in numeric
        if c not in (unit_col, assignment_col)
        and not (c.lower() == 'id' or c.lower().endswith('_id'))
    ]


def describe_metric(values: pd.Series) -> Dict:
    """
    Classify a metric and recommend the test that fits it

    Returns:
        kind: 'binary', 'count', or 'continuous'
        recommended_test: name used by the app's test selector
        summary: what the metric is, in words
        reason: why that test
    """
    clean = pd.Series(values).dropna().astype(float)
    if clean.empty:
        raise ValueError("Metric has no values")

    if is_binary_metric(clean):
        return {
            'kind': 'binary',
            'recommended_test': 'Proportions Z-Test',
            'summary': f"a yes/no outcome, so the quantity compared is a **rate** ({clean.mean():.2%} overall)",
            'reason': "A two-proportion z-test compares rates directly and gives an interval on the rate difference.",
            'skew': float(clean.skew()) if len(clean) > 2 else 0.0,
            'zero_share': float((clean == 0).mean())
        }

    skew = float(clean.skew()) if len(clean) > 2 else 0.0
    zero_share = float((clean == 0).mean())
    is_count = bool((clean >= 0).all() and (clean == clean.round()).all())
    heavy_tail = abs(skew) > 2 or zero_share > 0.5

    if heavy_tail:
        detail = []
        if zero_share > 0.5:
            detail.append(f"{zero_share:.0%} of units are zero")
        if abs(skew) > 2:
            detail.append(f"skew {skew:.1f}")
        return {
            'kind': 'count' if is_count else 'continuous',
            'recommended_test': 'Bootstrap',
            'summary': f"a {'count' if is_count else 'continuous'} value per unit with a heavy tail "
                       f"({', '.join(detail)}), so the quantity compared is the **mean**",
            'reason': "A few large values dominate the mean. The bootstrap makes no distributional "
                      "assumption; Welch's t-test is still reasonable with thousands of units per arm.",
            'skew': skew,
            'zero_share': zero_share
        }

    return {
        'kind': 'count' if is_count else 'continuous',
        'recommended_test': 'T-Test',
        'summary': f"a {'count' if is_count else 'continuous'} value per unit, so the quantity compared is the **mean** "
                   f"({clean.mean():,.2f} overall)",
        'reason': "Welch's t-test compares means and stays valid when the arms differ in size or variance.",
        'skew': skew,
        'zero_share': zero_share
    }


def to_unit_level(
    df: pd.DataFrame,
    unit_col: Optional[str],
    assignment_col: str,
    value_cols: List[str],
    agg: str = 'mean',
    order_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Collapse the data to one row per randomization unit

    Tests assume independent observations. If users were randomized but the
    table has one row per session or order, rows from the same user are
    correlated and the standard errors come out too small. Aggregating to the
    unit of randomization fixes that.

    Units that appear in more than one arm are contaminated (they saw both
    experiences) and are dropped.

    Args:
        df: Raw data
        unit_col: Randomization unit id column, or None if each row already is one unit
        assignment_col: Arm assignment column
        value_cols: Metric, guardrail, and covariate columns to carry along
        agg: How to combine a unit's rows: 'mean', 'sum', or 'max'
        order_col: Optional timestamp (or sequence) column. Units come back ordered by
            their first appearance, which sequential tests rely on. Without it, the
            existing row order is taken as arrival order.

    Returns:
        (unit-level frame, diagnostics)
    """
    if agg not in ('mean', 'sum', 'max'):
        raise ValueError("agg must be 'mean', 'sum', or 'max'")

    value_cols = list(dict.fromkeys(value_cols))
    data = df.dropna(subset=[assignment_col])
    if order_col is not None:
        data = data.sort_values(order_col, kind='stable')

    if unit_col is None:
        return data[[assignment_col] + value_cols].copy(), {
            'n_rows': int(len(data)), 'n_units': int(len(data)),
            'aggregated': False, 'max_rows_per_unit': 1, 'n_contaminated_units': 0
        }

    data = data.dropna(subset=[unit_col])
    arms_per_unit = data.groupby(unit_col)[assignment_col].nunique()
    contaminated = arms_per_unit[arms_per_unit > 1].index
    data = data[~data[unit_col].isin(contaminated)]

    rows_per_unit = data.groupby(unit_col).size()
    needs_aggregation = bool((rows_per_unit > 1).any())

    if needs_aggregation:
        unit_df = (
            data.groupby(unit_col, sort=False)   # keep units in order of first appearance
            .agg({assignment_col: 'first', **{c: agg for c in value_cols}})
            .reset_index()
        )
    else:
        unit_df = data[[unit_col, assignment_col] + value_cols].copy()

    return unit_df, {
        'n_rows': int(len(df)),
        'n_units': int(len(unit_df)),
        'aggregated': needs_aggregation,
        'max_rows_per_unit': int(rows_per_unit.max()) if len(rows_per_unit) else 0,
        'n_contaminated_units': int(len(contaminated))
    }


def experiment_summary(
    unit_label: str,
    n_units: int,
    arms: Dict[str, int],
    control: str,
    metric: str,
    metric_kind: str,
    higher_is_better: bool,
    mde_pct: Optional[float],
    test_name: str,
    alpha: float,
    guardrails: Optional[Dict[str, bool]] = None,
    expected_split: Optional[Dict[str, float]] = None
) -> str:
    """One paragraph stating the experiment as the analysis will treat it"""
    arm_text = ", ".join(f"**{a}** ({n:,})" for a, n in arms.items())
    split_text = (
        "planned split " + " / ".join(f"{expected_split[a]:.0%}" for a in arms)
        if expected_split else "planned as an equal split"
    )
    quantity = "rate" if metric_kind == 'binary' else "mean"
    direction = "rises" if higher_is_better else "falls"

    text = (
        f"**{n_units:,} {unit_label}** were randomized into {arm_text}, {split_text}; "
        f"**{control}** is the baseline.\n\n"
        f"The decision metric is the {quantity} of **{metric}** per {unit_label.rstrip('s')}. "
        f"A win means it **{direction}**"
    )
    text += f" by at least **{mde_pct:g}%** relative to control.\n\n" if mde_pct else ".\n\n"

    if guardrails:
        parts = [f"**{g}** must not {'fall' if up_is_good else 'rise'}" for g, up_is_good in guardrails.items()]
        text += "Guardrails: " + "; ".join(parts) + ".\n\n"

    text += f"Analysis: **{test_name}** at α = {alpha:g}."
    return text
