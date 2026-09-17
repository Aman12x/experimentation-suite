"""
Warehouse Analysis
One call from per-arm aggregate rows to tests, sample-ratio check, decisions and interpretation.
Shared by the MCP server and the REST API.
"""

from typing import Any, Dict, List, Optional

from statsmodels.stats.multitest import multipletests

from modules.ab_testing import ABTestingEngine
from modules.from_stats import srm_from_counts
from modules.sql_templates import rows_to_arms
from utils.decision import ship_decision
from utils.interpreters import StatisticalInterpreter

engine = ABTestingEngine()


def analyze_aggregates(
    rows: List[Dict[str, Any]],
    control_label: str,
    analysis: str = 'mean',
    higher_is_better: bool = True,
    alpha: float = 0.05,
    correction: str = 'holm',
    mde_pct: Optional[float] = None,
    expected_split: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Analyze an experiment from the rows returned by sql_templates.aggregate_sql
    
    Args:
        rows: One dict per arm: arm, n, and mean/var (or successes; plus mean_x, var_x, cov
            for 'cuped' and 'ratio')
        control_label: Which arm is the baseline
        analysis: 'mean', 'proportion', 'cuped', or 'ratio'
        higher_is_better: False for churn, latency, error rate
        alpha: Significance level
        correction: Multiplicity correction when there are several variants
        mde_pct: Smallest relative lift worth shipping
        expected_split: Planned traffic share per arm; defaults to equal
    """
    if analysis not in ('mean', 'proportion', 'cuped', 'ratio'):
        raise ValueError("analysis must be 'mean', 'proportion', 'cuped', or 'ratio'")
    if correction not in ('holm', 'bonferroni', 'fdr_bh', 'none'):
        raise ValueError("correction must be 'holm', 'bonferroni', 'fdr_bh', or 'none'")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    arms = rows_to_arms(rows)
    if control_label not in arms:
        raise ValueError(f"control_label '{control_label}' is not among the arms {sorted(arms)}")

    counts = {label: int(stats['n']) for label, stats in arms.items()}
    srm = srm_from_counts(counts, expected_split)
    health = {'sample_ratio_mismatch': srm}

    def compare(treatment_label: str) -> Dict:
        control, treatment = arms[control_label], arms[treatment_label]
        if analysis == 'proportion':
            return engine.proportions_test(
                int(control['successes']), int(control['n']),
                int(treatment['successes']), int(treatment['n']), alpha=alpha
            )
        if analysis == 'cuped':
            return engine.cuped_from_stats(control, treatment, alpha=alpha)
        if analysis == 'ratio':
            return engine.ratio_metric_from_stats(control, treatment, alpha=alpha)
        return engine.t_test_from_stats(control, treatment, alpha=alpha)

    variants = [label for label in arms if label != control_label]
    results = {label: compare(label) for label in variants}

    multiple = len(variants) > 1
    if multiple and correction != 'none':
        adjusted = multipletests([results[v]['p_value'] for v in variants], alpha=alpha, method=correction)[1]
        for label, p_adj in zip(variants, adjusted):
            results[label]['p_value_raw'] = results[label]['p_value']
            results[label]['p_value'] = float(p_adj)
            results[label]['significant'] = bool(p_adj < alpha)

    comparisons = []
    for label in variants:
        result = results[label]
        decision = ship_decision(result, health=health, higher_is_better=higher_is_better, mde_pct=mde_pct)
        comparisons.append({
            'variant': label,
            'decision': decision['decision'],
            'reasons': decision['reasons'],
            'result': result,
            'interpretation': StatisticalInterpreter.interpret_ab_test_results(
                result, higher_is_better=higher_is_better
            ),
        })

    return {
        'control': control_label,
        'analysis': analysis,
        'units_per_arm': counts,
        'sample_ratio_check': srm,
        'correction': correction if multiple else None,
        'comparisons': comparisons,
        'note': (
            "p-values are corrected for the number of variants." if multiple and correction != 'none' else None
        ),
    }
