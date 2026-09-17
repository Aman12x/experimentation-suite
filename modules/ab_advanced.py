"""
Advanced A/B Testing Methods
Proportion tests, relative-lift intervals, CUPED, non-parametric and bootstrap
tests, multi-variant comparisons with multiplicity correction, and
always-valid sequential testing
"""

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, Optional, Sequence


def is_binary_metric(values: Sequence) -> bool:
    """True when a metric only takes the values 0 and 1"""
    unique = set(np.unique(np.asarray(values, dtype=float)).tolist())
    return len(unique) > 0 and unique <= {0.0, 1.0}


def relative_lift_interval(
    control_mean: float,
    treatment_mean: float,
    control_var_of_mean: float,
    treatment_var_of_mean: float,
    alpha: float = 0.05
) -> Dict:
    """
    Delta-method confidence interval for relative lift, (treatment / control - 1) * 100

    Dividing the absolute CI by the control mean ignores the noise in the
    control mean itself; the delta method accounts for it.

    Args:
        control_mean: Mean of the control group
        treatment_mean: Mean of the treatment group
        control_var_of_mean: Variance of the control mean (s² / n)
        treatment_var_of_mean: Variance of the treatment mean (s² / n)
        alpha: Significance level
    """
    if control_mean == 0:
        return {'relative_lift': 0.0, 'lift_ci_lower': float('nan'), 'lift_ci_upper': float('nan')}

    ratio = treatment_mean / control_mean
    var_ratio = (
        treatment_var_of_mean / control_mean**2
        + treatment_mean**2 * control_var_of_mean / control_mean**4
    )
    half_width = stats.norm.ppf(1 - alpha/2) * np.sqrt(var_ratio)

    return {
        'relative_lift': float((ratio - 1) * 100),
        'lift_ci_lower': float((ratio - 1 - half_width) * 100),
        'lift_ci_upper': float((ratio - 1 + half_width) * 100)
    }


class AdvancedABMethods:
    """Mixed into ABTestingEngine"""

    def proportions_test(
        self,
        control_success: int,
        control_total: int,
        treatment_success: int,
        treatment_total: int,
        alternative: str = 'two-sided',
        alpha: float = 0.05
    ) -> Dict:
        """
        Two-proportion z-test for binary metrics (conversion, click-through)

        Args:
            control_success: Number of successes in control
            control_total: Total observations in control
            treatment_success: Number of successes in treatment
            treatment_total: Total observations in treatment
            alternative: 'two-sided', 'greater', or 'less'
            alpha: Significance level
        """
        if control_total <= 0 or treatment_total <= 0:
            raise ValueError("Group totals must be positive")
        if not (0 <= control_success <= control_total and 0 <= treatment_success <= treatment_total):
            raise ValueError("Successes must be between 0 and the group total")
        if alternative not in ('two-sided', 'greater', 'less'):
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

        p_c = control_success / control_total
        p_t = treatment_success / treatment_total
        diff = p_t - p_c

        # Pooled standard error under the null for the test statistic
        p_pool = (control_success + treatment_success) / (control_total + treatment_total)
        se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1/control_total + 1/treatment_total))
        z_stat = diff / se_pooled if se_pooled > 0 else 0.0

        if alternative == 'two-sided':
            p_value = 2 * stats.norm.sf(abs(z_stat))
        elif alternative == 'greater':
            p_value = stats.norm.sf(z_stat)
        else:
            p_value = stats.norm.cdf(z_stat)

        # Unpooled standard error for the confidence interval
        var_c = p_c * (1 - p_c) / control_total
        var_t = p_t * (1 - p_t) / treatment_total
        z_critical = stats.norm.ppf(1 - alpha/2)
        half_width = z_critical * np.sqrt(var_c + var_t)

        return {
            'test_type': 'proportions z-test',
            'z_statistic': float(z_stat),
            'p_value': float(p_value),
            'control_rate': float(p_c),
            'treatment_rate': float(p_t),
            'control_mean': float(p_c),
            'treatment_mean': float(p_t),
            'control_n': int(control_total),
            'treatment_n': int(treatment_total),
            'rate_difference': float(diff),
            'mean_difference': float(diff),
            'ci_lower': float(diff - half_width),
            'ci_upper': float(diff + half_width),
            **relative_lift_interval(p_c, p_t, var_c, var_t, alpha),
            'significant': bool(p_value < alpha),
            'alpha': alpha
        }

    def cuped_test(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        control_covariate: np.ndarray,
        treatment_covariate: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """
        CUPED variance reduction (Deng et al., 2013)

        Removes the part of the metric that a pre-experiment covariate already
        explains, then runs Welch's t-test on what is left. The covariate must
        be measured before assignment so the treatment cannot affect it.

        Args:
            control: Control group metric
            treatment: Treatment group metric
            control_covariate: Pre-experiment covariate for control users
            treatment_covariate: Pre-experiment covariate for treatment users
            alpha: Significance level
        """
        control = np.asarray(control, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        x_c = np.asarray(control_covariate, dtype=float)
        x_t = np.asarray(treatment_covariate, dtype=float)
        if len(control) != len(x_c) or len(treatment) != len(x_t):
            raise ValueError("Each covariate must line up one-to-one with its group's metric")

        y = np.concatenate([control, treatment])
        x = np.concatenate([x_c, x_t])
        var_x = np.var(x, ddof=1)
        if var_x == 0:
            raise ValueError("Covariate has zero variance, nothing to adjust for")

        # theta from pooled data; centering on the pooled mean keeps the estimate unbiased
        theta = np.cov(y, x, ddof=1)[0, 1] / var_x
        control_adj = control - theta * (x_c - x.mean())
        treatment_adj = treatment - theta * (x_t - x.mean())

        unadjusted = self.t_test(control, treatment, alpha=alpha)
        results = self.t_test(control_adj, treatment_adj, alpha=alpha)

        var_before = (unadjusted['ci_upper'] - unadjusted['ci_lower'])**2
        var_after = (results['ci_upper'] - results['ci_lower'])**2

        results.update({
            'test_type': 'cuped t-test',
            'theta': float(theta),
            'covariate_correlation': float(np.corrcoef(y, x)[0, 1]),
            'variance_reduction_pct': float((1 - var_after / var_before) * 100) if var_before > 0 else 0.0,
            'unadjusted_p_value': unadjusted['p_value'],
            'unadjusted_ci_lower': unadjusted['ci_lower'],
            'unadjusted_ci_upper': unadjusted['ci_upper'],
            'unadjusted_mean_difference': unadjusted['mean_difference'],
            # Report lift against the raw control mean; adjusted means are shifted
            'control_mean': unadjusted['control_mean'],
            'treatment_mean': unadjusted['control_mean'] + results['mean_difference'],
        })
        results['relative_lift'] = (
            results['mean_difference'] / unadjusted['control_mean'] * 100
            if unadjusted['control_mean'] != 0 else 0.0
        )
        results.pop('lift_ci_lower', None)
        results.pop('lift_ci_upper', None)
        return results

    def mann_whitney_test(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        alternative: str = 'two-sided',
        alpha: float = 0.05
    ) -> Dict:
        """
        Mann-Whitney U test for skewed or heavy-tailed metrics

        Tests whether a random treatment user tends to have a higher value than
        a random control user. It does not test the difference in means.
        """
        control = np.asarray(control, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        if len(control) == 0 or len(treatment) == 0:
            raise ValueError("Both groups need at least one observation")

        u_stat, p_value = stats.mannwhitneyu(treatment, control, alternative=alternative)
        control_mean, treatment_mean = np.mean(control), np.mean(treatment)

        return {
            'test_type': 'mann-whitney u',
            'u_statistic': float(u_stat),
            'p_value': float(p_value),
            # P(treatment > control) + 0.5 * P(tie); 0.5 means no tendency either way
            'prob_superiority': float(u_stat / (len(control) * len(treatment))),
            'control_mean': float(control_mean),
            'treatment_mean': float(treatment_mean),
            'control_median': float(np.median(control)),
            'treatment_median': float(np.median(treatment)),
            'control_n': int(len(control)),
            'treatment_n': int(len(treatment)),
            'mean_difference': float(treatment_mean - control_mean),
            'relative_lift': float((treatment_mean - control_mean) / control_mean * 100) if control_mean != 0 else 0.0,
            'significant': bool(p_value < alpha),
            'alpha': alpha
        }

    def bootstrap_test(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        alpha: float = 0.05,
        n_resamples: int = 5000,
        seed: int = 42
    ) -> Dict:
        """
        Percentile bootstrap for the difference in means and the relative lift

        Makes no distributional assumption, which suits revenue-style metrics
        where a few large values dominate.
        """
        control = np.asarray(control, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        if len(control) < 2 or len(treatment) < 2:
            raise ValueError("Bootstrap needs at least two observations per group")

        rng = np.random.default_rng(seed)
        boot_c = rng.choice(control, (n_resamples, len(control))).mean(axis=1)
        boot_t = rng.choice(treatment, (n_resamples, len(treatment))).mean(axis=1)

        diffs = boot_t - boot_c
        ci_lower, ci_upper = np.percentile(diffs, [100 * alpha/2, 100 * (1 - alpha/2)])

        control_mean, treatment_mean = np.mean(control), np.mean(treatment)
        results = {
            'test_type': 'bootstrap',
            # Two-sided p-value from how often the resampled difference crosses zero
            'p_value': float(min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))),
            'control_mean': float(control_mean),
            'treatment_mean': float(treatment_mean),
            'control_n': int(len(control)),
            'treatment_n': int(len(treatment)),
            'mean_difference': float(treatment_mean - control_mean),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'relative_lift': float((treatment_mean - control_mean) / control_mean * 100) if control_mean != 0 else 0.0,
            'n_resamples': n_resamples,
            'alpha': alpha
        }
        if (boot_c > 0).all():
            lifts = (boot_t / boot_c - 1) * 100
            lo, hi = np.percentile(lifts, [100 * alpha/2, 100 * (1 - alpha/2)])
            results.update({'lift_ci_lower': float(lo), 'lift_ci_upper': float(hi)})

        results['significant'] = bool(not (ci_lower <= 0 <= ci_upper))
        return results

    def multi_variant_test(
        self,
        groups: Dict[str, np.ndarray],
        control_label: str,
        alpha: float = 0.05,
        correction: str = 'holm'
    ) -> Dict:
        """
        Compare every variant against control with a multiplicity correction

        Testing k variants at alpha each inflates the chance of at least one
        false win to roughly 1 - (1 - alpha)^k. Adjusted p-values keep it at alpha.

        Args:
            groups: {label: metric values} including the control
            control_label: Which label is the control
            alpha: Family-wise (holm, bonferroni) or false-discovery (fdr_bh) level
            correction: 'holm', 'bonferroni', 'fdr_bh', or 'none'
        """
        if control_label not in groups:
            raise ValueError(f"Control label '{control_label}' not found")
        if len(groups) < 2:
            raise ValueError("Need a control and at least one variant")
        if correction not in ('holm', 'bonferroni', 'fdr_bh', 'none'):
            raise ValueError("correction must be 'holm', 'bonferroni', 'fdr_bh', or 'none'")

        control = np.asarray(groups[control_label], dtype=float)
        binary = all(is_binary_metric(v) for v in groups.values())

        comparisons = []
        for label, values in groups.items():
            if label == control_label:
                continue
            values = np.asarray(values, dtype=float)
            if binary:
                r = self.proportions_test(
                    int(control.sum()), len(control), int(values.sum()), len(values), alpha=alpha
                )
            else:
                r = self.t_test(control, values, alpha=alpha)
            comparisons.append({**r, 'group': str(label)})

        raw = [c['p_value'] for c in comparisons]
        if correction == 'none':
            adjusted = raw
        else:
            adjusted = multipletests(raw, alpha=alpha, method=correction)[1].tolist()

        for c, p_adj in zip(comparisons, adjusted):
            c['p_value_raw'] = c['p_value']
            c['p_value_adjusted'] = float(p_adj)
            c['significant_raw'] = c['significant']
            c['significant'] = bool(p_adj < alpha)

        winners = [c for c in comparisons if c['significant'] and c['mean_difference'] > 0]
        best = max(winners, key=lambda c: c['mean_difference'])['group'] if winners else None

        return {
            'test_type': 'multi-variant',
            'control': str(control_label),
            'correction': correction,
            'n_comparisons': len(comparisons),
            'metric_type': 'binary' if binary else 'continuous',
            'comparisons': comparisons,
            'best_variant': best,
            'alpha': alpha
        }

    def sequential_test(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        alpha: float = 0.05,
        tau: Optional[float] = None,
        n_looks: int = 20,
        min_per_group: int = 50
    ) -> Dict:
        """
        Always-valid sequential test (mixture SPRT, Johari et al., 2017)

        A fixed-horizon p-value is only valid if you look once. Checking it
        daily and stopping at the first p < alpha inflates false positives
        several-fold. The always-valid p-value can be checked at every look and
        stopped on at any time while keeping the false-positive rate at alpha.

        Data is taken in arrival order and evaluated at n_looks checkpoints.

        Args:
            control: Control observations in arrival order
            treatment: Treatment observations in arrival order
            alpha: Significance level
            tau: Std dev of the normal mixing prior over the true difference.
                Defaults to 10% of the pooled standard deviation.
            n_looks: Number of interim looks
            min_per_group: Observations per group before the first look
        """
        control = np.asarray(control, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        if min(len(control), len(treatment)) < max(min_per_group, 2):
            raise ValueError(f"Need at least {max(min_per_group, 2)} observations per group")

        if tau is None:
            tau = 0.1 * np.std(np.concatenate([control, treatment]), ddof=1)
        if tau <= 0:
            raise ValueError("tau must be positive")
        tau2 = tau**2

        fractions = np.linspace(0, 1, n_looks + 1)[1:]
        looks = []
        running_p = 1.0
        stopped_at = None

        for frac in fractions:
            n_c = max(min_per_group, int(round(frac * len(control))))
            n_t = max(min_per_group, int(round(frac * len(treatment))))
            c, t = control[:n_c], treatment[:n_t]

            diff = t.mean() - c.mean()
            v = c.var(ddof=1) / n_c + t.var(ddof=1) / n_t
            if v <= 0:
                continue

            # Mixture likelihood ratio against H0: difference = 0
            log_lr = 0.5 * np.log(v / (v + tau2)) + tau2 * diff**2 / (2 * v * (v + tau2))
            running_p = min(running_p, float(np.exp(-log_lr)), 1.0)

            half_width = np.sqrt(
                (2 * v * (v + tau2) / tau2) * (np.log(1 / alpha) + 0.5 * np.log((v + tau2) / v))
            )
            fixed_p = 2 * stats.norm.sf(abs(diff) / np.sqrt(v))

            looks.append({
                'n_control': n_c,
                'n_treatment': n_t,
                'mean_difference': float(diff),
                'always_valid_p_value': running_p,
                'fixed_horizon_p_value': float(fixed_p),
                'ci_lower': float(diff - half_width),
                'ci_upper': float(diff + half_width)
            })
            if stopped_at is None and running_p < alpha:
                stopped_at = len(looks)

        if not looks:
            raise ValueError("Metric has zero variance, nothing to test")

        final = looks[-1]
        control_mean = float(control.mean())
        return {
            'test_type': 'sequential (mSPRT)',
            'p_value': final['always_valid_p_value'],
            'control_mean': control_mean,
            'treatment_mean': float(treatment.mean()),
            'control_n': int(len(control)),
            'treatment_n': int(len(treatment)),
            'mean_difference': final['mean_difference'],
            'ci_lower': final['ci_lower'],
            'ci_upper': final['ci_upper'],
            'relative_lift': final['mean_difference'] / control_mean * 100 if control_mean != 0 else 0.0,
            'significant': bool(final['always_valid_p_value'] < alpha),
            'could_stop_at_look': stopped_at,
            'could_stop_at_n': (
                looks[stopped_at - 1]['n_control'] + looks[stopped_at - 1]['n_treatment']
                if stopped_at else None
            ),
            'naive_peeking_would_stop_at_look': next(
                (i + 1 for i, l in enumerate(looks) if l['fixed_horizon_p_value'] < alpha), None
            ),
            'tau': float(tau),
            'looks': looks,
            'alpha': alpha
        }
