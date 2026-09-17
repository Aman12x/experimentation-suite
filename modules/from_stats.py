"""
Tests From Summary Statistics
Run the same tests from per-arm aggregates instead of raw rows

A warehouse can compute counts, means, variances and covariances over any
number of rows in one GROUP BY. These entry points take those few numbers and
return exactly what the row-level methods return, so row-level data never has
to leave the warehouse.

Bootstrap and Mann-Whitney are not here: they need the rows themselves.
"""

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, Optional

from .ab_advanced import relative_lift_fields


def _moments(arm: Dict, name: str, need_covariate: bool = False) -> Dict:
    """Validate one arm's aggregates. Keys: n, mean, var, and for two-column methods mean_x, var_x, cov"""
    required = ['n', 'mean', 'var'] + (['mean_x', 'var_x', 'cov'] if need_covariate else [])
    missing = [k for k in required if arm.get(k) is None]
    if missing:
        raise ValueError(f"{name} is missing {missing}. Expected keys: {required}")

    values = {k: float(arm[k]) for k in required}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError(f"{name} contains non-finite statistics")
    if values['n'] < 2 or values['n'] != int(values['n']):
        raise ValueError(f"{name}: n must be a whole number of at least 2")
    if values['var'] < 0 or values.get('var_x', 0) < 0:
        raise ValueError(f"{name}: variance cannot be negative")
    values['n'] = int(values['n'])
    return values


def _welch(mean_c, var_c, n_c, mean_t, var_t, n_t, alpha, equal_var=False) -> Dict:
    """Two-sample t-test from moments; shared by every method in this module"""
    diff = mean_t - mean_c

    if var_c == 0 and var_t == 0:
        same = diff == 0
        t_stat, p_value, df = (0.0 if same else None), (1.0 if same else 0.0), float(n_c + n_t - 2)
        ci_lower = ci_upper = diff
    else:
        result = stats.ttest_ind_from_stats(
            mean_t, np.sqrt(var_t), n_t, mean_c, np.sqrt(var_c), n_c, equal_var=equal_var
        )
        t_stat, p_value = float(result.statistic), float(result.pvalue)
        if equal_var:
            df = float(n_c + n_t - 2)
            pooled = ((n_c - 1) * var_c + (n_t - 1) * var_t) / df
            se = np.sqrt(pooled * (1/n_c + 1/n_t))
        else:
            se = np.sqrt(var_c / n_c + var_t / n_t)
            df = float(
                (var_c / n_c + var_t / n_t)**2
                / ((var_c / n_c)**2 / (n_c - 1) + (var_t / n_t)**2 / (n_t - 1))
            )
        half_width = stats.t.ppf(1 - alpha/2, df) * se
        ci_lower, ci_upper = diff - half_width, diff + half_width

    pooled_std = np.sqrt(((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2))
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'zero_variance': bool(var_c == 0 and var_t == 0),
        'degrees_of_freedom': df,
        'mean_difference': float(diff),
        'cohens_d': float(diff / pooled_std) if pooled_std > 0 else 0.0,
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'significant': bool(p_value < alpha),
        'alpha': alpha
    }


class FromStatsMethods:
    """Mixed into ABTestingEngine"""

    def t_test_from_stats(
        self,
        control: Dict,
        treatment: Dict,
        alpha: float = 0.05,
        equal_var: bool = False
    ) -> Dict:
        """
        Welch's (or Student's) t-test from per-arm n, mean and sample variance

        Args:
            control: {'n': ..., 'mean': ..., 'var': ...} with var the sample variance (n - 1 denominator)
            treatment: Same keys for the treatment arm
            alpha: Significance level
            equal_var: Assume equal variances (Student's t-test)
        """
        c, t = _moments(control, 'control'), _moments(treatment, 'treatment')
        return {
            'test_type': 't-test',
            'variant': 'student' if equal_var else 'welch',
            'source': 'summary statistics',
            **_welch(c['mean'], c['var'], c['n'], t['mean'], t['var'], t['n'], alpha, equal_var),
            'control_mean': c['mean'],
            'treatment_mean': t['mean'],
            'control_std': float(np.sqrt(c['var'])),
            'treatment_std': float(np.sqrt(t['var'])),
            'control_n': c['n'],
            'treatment_n': t['n'],
            **relative_lift_fields(c['mean'], t['mean'], c['var'] / c['n'], t['var'] / t['n'], alpha)
        }

    def cuped_from_stats(self, control: Dict, treatment: Dict, alpha: float = 0.05) -> Dict:
        """
        CUPED from per-arm moments of the metric (y) and the pre-experiment covariate (x)

        Args:
            control: {'n', 'mean', 'var'} for the metric plus {'mean_x', 'var_x', 'cov'} where
                cov is the sample covariance between metric and covariate within the arm
            treatment: Same keys for the treatment arm
            alpha: Significance level
        """
        c = _moments(control, 'control', need_covariate=True)
        t = _moments(treatment, 'treatment', need_covariate=True)
        n = c['n'] + t['n']

        # Pooled covariance and variance across both arms = within-arm part + between-arm part
        grand_y = (c['n'] * c['mean'] + t['n'] * t['mean']) / n
        grand_x = (c['n'] * c['mean_x'] + t['n'] * t['mean_x']) / n

        def pooled(within_c, within_t, dev_c_a, dev_c_b, dev_t_a, dev_t_b):
            return (
                (c['n'] - 1) * within_c + (t['n'] - 1) * within_t
                + c['n'] * dev_c_a * dev_c_b + t['n'] * dev_t_a * dev_t_b
            ) / (n - 1)

        dx_c, dx_t = c['mean_x'] - grand_x, t['mean_x'] - grand_x
        dy_c, dy_t = c['mean'] - grand_y, t['mean'] - grand_y
        var_x = pooled(c['var_x'], t['var_x'], dx_c, dx_c, dx_t, dx_t)
        if var_x <= 0:
            raise ValueError("Covariate has zero variance, nothing to adjust for")
        cov_xy = pooled(c['cov'], t['cov'], dy_c, dx_c, dy_t, dx_t)
        var_y = pooled(c['var'], t['var'], dy_c, dy_c, dy_t, dy_t)
        theta = cov_xy / var_x

        def adjusted(arm):
            mean = arm['mean'] - theta * (arm['mean_x'] - grand_x)
            var = arm['var'] - 2 * theta * arm['cov'] + theta**2 * arm['var_x']
            return mean, max(var, 0.0)

        (adj_mean_c, adj_var_c), (adj_mean_t, adj_var_t) = adjusted(c), adjusted(t)
        result = _welch(adj_mean_c, adj_var_c, c['n'], adj_mean_t, adj_var_t, t['n'], alpha)
        plain = _welch(c['mean'], c['var'], c['n'], t['mean'], t['var'], t['n'], alpha)

        width_before = (plain['ci_upper'] - plain['ci_lower'])**2
        width_after = (result['ci_upper'] - result['ci_lower'])**2
        result.update({
            'test_type': 'cuped t-test',
            'variant': 'welch',
            'source': 'summary statistics',
            'theta': float(theta),
            'covariate_correlation': float(cov_xy / np.sqrt(var_x * var_y)) if var_y > 0 else 0.0,
            'variance_reduction_pct': float((1 - width_after / width_before) * 100) if width_before > 0 else 0.0,
            'unadjusted_p_value': plain['p_value'],
            'unadjusted_ci_lower': plain['ci_lower'],
            'unadjusted_ci_upper': plain['ci_upper'],
            'unadjusted_mean_difference': plain['mean_difference'],
            'control_mean': c['mean'],
            'treatment_mean': c['mean'] + result['mean_difference'],
            'control_n': c['n'],
            'treatment_n': t['n'],
            **relative_lift_fields(c['mean'], c['mean'] + result['mean_difference'], c['var'] / c['n'])
        })
        return result

    def ratio_metric_from_stats(self, control: Dict, treatment: Dict, alpha: float = 0.05) -> Dict:
        """
        Delta-method ratio test from per-arm moments of per-unit numerator (y) and denominator (x)

        Args:
            control: {'n', 'mean', 'var'} for the numerator plus {'mean_x', 'var_x', 'cov'} for
                the denominator and their covariance, all computed over units
            treatment: Same keys for the treatment arm
            alpha: Significance level
        """
        def arm(values, name):
            m = _moments(values, name, need_covariate=True)
            if m['mean_x'] <= 0:
                raise ValueError(f"{name}: denominator must be positive on average")
            ratio = m['mean'] / m['mean_x']
            variance = (m['var'] - 2 * ratio * m['cov'] + ratio**2 * m['var_x']) / (m['n'] * m['mean_x']**2)
            return ratio, max(float(variance), 0.0), m['n']

        ratio_c, var_c, n_c = arm(control, 'control')
        ratio_t, var_t, n_t = arm(treatment, 'treatment')

        diff = ratio_t - ratio_c
        se = np.sqrt(var_c + var_t)
        z_stat = diff / se if se > 0 else 0.0
        p_value = 2 * stats.norm.sf(abs(z_stat)) if se > 0 else 1.0
        half_width = stats.norm.ppf(1 - alpha/2) * se

        return {
            'test_type': 'ratio metric (delta method)',
            'source': 'summary statistics',
            'z_statistic': float(z_stat),
            'p_value': float(p_value),
            'control_mean': float(ratio_c),
            'treatment_mean': float(ratio_t),
            'control_n': n_c,
            'treatment_n': n_t,
            'mean_difference': float(diff),
            'ci_lower': float(diff - half_width),
            'ci_upper': float(diff + half_width),
            **relative_lift_fields(ratio_c, ratio_t, var_c, var_t, alpha),
            'significant': bool(p_value < alpha),
            'alpha': alpha
        }

    def multi_variant_from_stats(
        self,
        arms: Dict[str, Dict],
        control_label: str,
        alpha: float = 0.05,
        correction: str = 'holm',
        metric_type: str = 'mean'
    ) -> Dict:
        """
        Every variant against control from per-arm aggregates, with multiplicity correction

        Args:
            arms: {label: stats}. For metric_type 'mean': {'n', 'mean', 'var'}.
                For 'proportion': {'n', 'successes'}.
            control_label: Which label is the control
            alpha: Family-wise or false-discovery level
            correction: 'holm', 'bonferroni', 'fdr_bh', or 'none'
            metric_type: 'mean' or 'proportion'
        """
        if control_label not in arms:
            raise ValueError(f"Control label '{control_label}' not found")
        if len(arms) < 2:
            raise ValueError("Need a control and at least one variant")
        if correction not in ('holm', 'bonferroni', 'fdr_bh', 'none'):
            raise ValueError("correction must be 'holm', 'bonferroni', 'fdr_bh', or 'none'")
        if metric_type not in ('mean', 'proportion'):
            raise ValueError("metric_type must be 'mean' or 'proportion'")

        control = arms[control_label]
        comparisons = []
        for label, arm in arms.items():
            if label == control_label:
                continue
            if metric_type == 'proportion':
                r = self.proportions_test(
                    int(control['successes']), int(control['n']), int(arm['successes']), int(arm['n']), alpha=alpha
                )
            else:
                r = self.t_test_from_stats(control, arm, alpha=alpha)
            comparisons.append({**r, 'group': str(label)})

        raw = [c['p_value'] for c in comparisons]
        adjusted = raw if correction == 'none' else multipletests(raw, alpha=alpha, method=correction)[1].tolist()
        for c, p_adj in zip(comparisons, adjusted):
            c['p_value_raw'] = c['p_value']
            c['p_value_adjusted'] = float(p_adj)
            c['significant_raw'] = c['significant']
            c['significant'] = bool(p_adj < alpha)

        winners = [c for c in comparisons if c['significant'] and c['mean_difference'] > 0]
        return {
            'test_type': 'multi-variant',
            'source': 'summary statistics',
            'control': str(control_label),
            'correction': correction,
            'n_comparisons': len(comparisons),
            'metric_type': 'binary' if metric_type == 'proportion' else 'continuous',
            'comparisons': comparisons,
            'best_variant': max(winners, key=lambda c: c['mean_difference'])['group'] if winners else None,
            'alpha': alpha
        }


def srm_from_counts(counts: Dict[str, int], expected_ratio: Optional[Dict[str, float]] = None,
                    alpha: float = 0.001) -> Dict:
    """
    Sample ratio mismatch check from per-arm unit counts

    Args:
        counts: {arm label: number of units}
        expected_ratio: {arm label: planned share}. Defaults to an equal split.
        alpha: Significance level (0.001 is the convention for SRM)
    """
    if len(counts) < 2:
        raise ValueError("SRM check requires at least 2 arms")
    labels = list(counts)
    observed = np.array([counts[a] for a in labels], dtype=float)
    if (observed < 0).any() or observed.sum() == 0:
        raise ValueError("Counts must be non-negative and not all zero")

    if expected_ratio is None:
        ratio = np.full(len(labels), 1 / len(labels))
    else:
        missing = [a for a in labels if a not in expected_ratio]
        if missing:
            raise ValueError(f"No expected share given for arms: {missing}")
        ratio = np.array([expected_ratio[a] for a in labels], dtype=float)
        if (ratio <= 0).any():
            raise ValueError("Expected shares must be positive")
        ratio = ratio / ratio.sum()

    chi2, p_value = stats.chisquare(observed, ratio * observed.sum())
    return {
        'has_srm': bool(p_value < alpha),
        'chi2_statistic': float(chi2),
        'p_value': float(p_value),
        'groups': labels,
        'observed_counts': observed.astype(int).tolist(),
        'expected_proportions': ratio.tolist(),
        'actual_proportions': (observed / observed.sum()).tolist(),
        'severity': 'CRITICAL' if p_value < alpha else 'OK'
    }
