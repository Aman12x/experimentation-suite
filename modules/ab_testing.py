"""
A/B Testing Engine
Implements statistical tests, power analysis, and Bayesian A/B testing
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import tt_ind_solve_power, zt_ind_solve_power
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from typing import Dict, Tuple, Optional, List

from .ab_advanced import AdvancedABMethods, relative_lift_interval


class ABTestingEngine(AdvancedABMethods):
    """Comprehensive A/B testing engine with multiple statistical methods"""
    
    def __init__(self):
        self.results: Dict = {}
    
    def t_test(
        self, 
        control: np.ndarray, 
        treatment: np.ndarray,
        alternative: str = 'two-sided',
        alpha: float = 0.05,
        equal_var: bool = False
    ) -> Dict:
        """
        Perform independent samples t-test
        
        Defaults to Welch's t-test, which stays valid when group variances
        or sample sizes differ. Pass equal_var=True for Student's pooled test.
        
        Args:
            control: Control group data
            treatment: Treatment group data
            alternative: 'two-sided', 'greater', or 'less'
            alpha: Significance level
            equal_var: Assume equal variances (Student's t-test)
            
        Returns:
            Dictionary with test results
        """
        control = np.asarray(control, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        if len(control) == 0 or len(treatment) == 0:
            raise ValueError("Both groups need at least one observation")
        if alternative not in ('two-sided', 'greater', 'less'):
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
        
        n_c, n_t = len(control), len(treatment)
        var_c = np.var(control, ddof=1) if n_c > 1 else np.nan
        var_t = np.var(treatment, ddof=1) if n_t > 1 else np.nan
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(
            treatment, control, alternative=alternative, equal_var=equal_var
        )
        if np.isnan(p_value):  # zero variance in both groups (e.g. identical arrays)
            t_stat, p_value = 0.0, 1.0
        
        # Calculate statistics
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        pooled_std = np.sqrt(
            ((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2)
        ) if n_c + n_t > 2 else np.nan
        
        # Effect size (Cohen's d)
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0.0
        
        # Standard error and degrees of freedom for the chosen test
        if equal_var:
            se = pooled_std * np.sqrt(1/n_c + 1/n_t)
            df = n_c + n_t - 2
        else:
            se = np.sqrt(var_c / n_c + var_t / n_t)
            # Welch-Satterthwaite degrees of freedom
            denom = (var_c / n_c)**2 / (n_c - 1) + (var_t / n_t)**2 / (n_t - 1) if min(n_c, n_t) > 1 else np.nan
            df = (var_c / n_c + var_t / n_t)**2 / denom if denom and denom > 0 else np.nan
        
        # Confidence interval for difference in means
        mean_difference = treatment_mean - control_mean
        if np.isnan(se) or np.isnan(df):
            # Zero variance or a single observation: no spread to build an interval from
            ci_lower = ci_upper = mean_difference
            df = float(max(n_c + n_t - 2, 0))
        else:
            t_critical = stats.t.ppf(1 - alpha/2, df)
            ci_lower = mean_difference - t_critical * se
            ci_upper = mean_difference + t_critical * se
        
        # Relative lift
        relative_lift = (mean_difference / control_mean) * 100 if control_mean != 0 else 0.0
        
        return {
            'test_type': 't-test',
            'variant': 'student' if equal_var else 'welch',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'degrees_of_freedom': float(df),
            'control_mean': float(control_mean),
            'treatment_mean': float(treatment_mean),
            'control_std': float(np.nan_to_num(np.sqrt(var_c))),
            'treatment_std': float(np.nan_to_num(np.sqrt(var_t))),
            'control_n': int(n_c),
            'treatment_n': int(n_t),
            'mean_difference': float(mean_difference),
            'cohens_d': float(cohens_d),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'relative_lift': float(relative_lift),
            **{k: v for k, v in relative_lift_interval(
                control_mean, treatment_mean, var_c / n_c, var_t / n_t, alpha
            ).items() if k != 'relative_lift' and not np.isnan(v)},
            'significant': bool(p_value < alpha),
            'alpha': alpha
        }
    
    def z_test(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        alternative: str = 'two-sided',
        alpha: float = 0.05
    ) -> Dict:
        """
        Perform Z-test for large samples
        
        Args:
            control: Control group data
            treatment: Treatment group data
            alternative: 'two-sided', 'greater', or 'less'
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        control = np.asarray(control, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        if len(control) < 2 or len(treatment) < 2:
            raise ValueError("Z-test needs at least two observations per group")
        if alternative not in ('two-sided', 'greater', 'less'):
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
        
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        control_std = np.std(control, ddof=1)
        treatment_std = np.std(treatment, ddof=1)
        
        # Z-statistic
        se = np.sqrt((control_std**2 / len(control)) + (treatment_std**2 / len(treatment)))
        z_stat = (treatment_mean - control_mean) / se if se > 0 else 0.0
        
        # P-value based on alternative hypothesis
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(z_stat)
        else:  # 'less'
            p_value = stats.norm.cdf(z_stat)
        
        # Confidence interval
        z_critical = stats.norm.ppf(1 - alpha/2)
        ci_lower = (treatment_mean - control_mean) - z_critical * se
        ci_upper = (treatment_mean - control_mean) + z_critical * se
        
        # Effect size
        pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0.0
        
        relative_lift = ((treatment_mean - control_mean) / control_mean) * 100 if control_mean != 0 else 0.0
        
        return {
            'test_type': 'z-test',
            'z_statistic': float(z_stat),
            'p_value': float(p_value),
            'control_mean': float(control_mean),
            'treatment_mean': float(treatment_mean),
            'control_std': float(control_std),
            'treatment_std': float(treatment_std),
            'control_n': int(len(control)),
            'treatment_n': int(len(treatment)),
            'mean_difference': float(treatment_mean - control_mean),
            'cohens_d': float(cohens_d),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'relative_lift': float(relative_lift),
            **{k: v for k, v in relative_lift_interval(
                control_mean, treatment_mean,
                control_std**2 / len(control), treatment_std**2 / len(treatment), alpha
            ).items() if k != 'relative_lift' and not np.isnan(v)},
            'significant': bool(p_value < alpha),
            'alpha': alpha
        }
    
    def chi_squared_test(
        self,
        control_success: int,
        control_total: int,
        treatment_success: int,
        treatment_total: int,
        alpha: float = 0.05
    ) -> Dict:
        """
        Perform Chi-squared test for proportions
        
        Args:
            control_success: Number of successes in control
            control_total: Total observations in control
            treatment_success: Number of successes in treatment
            treatment_total: Total observations in treatment
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        if control_total <= 0 or treatment_total <= 0:
            raise ValueError("Group totals must be positive")
        if not (0 <= control_success <= control_total and 0 <= treatment_success <= treatment_total):
            raise ValueError("Successes must be between 0 and the group total")
        
        # Create contingency table
        observed = np.array([
            [treatment_success, treatment_total - treatment_success],
            [control_success, control_total - control_success]
        ])
        
        # Chi-squared test
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        
        # Proportions
        control_rate = control_success / control_total
        treatment_rate = treatment_success / treatment_total
        
        # Confidence interval for difference in proportions
        se = np.sqrt(
            (control_rate * (1 - control_rate) / control_total) +
            (treatment_rate * (1 - treatment_rate) / treatment_total)
        )
        z_critical = stats.norm.ppf(1 - alpha/2)
        diff = treatment_rate - control_rate
        ci_lower = diff - z_critical * se
        ci_upper = diff + z_critical * se
        
        # Relative lift
        relative_lift = ((treatment_rate - control_rate) / control_rate) * 100 if control_rate > 0 else 0
        
        return {
            'test_type': 'chi-squared',
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'control_rate': float(control_rate),
            'treatment_rate': float(treatment_rate),
            'control_mean': float(control_rate),
            'treatment_mean': float(treatment_rate),
            'control_n': int(control_total),
            'treatment_n': int(treatment_total),
            'rate_difference': float(diff),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'relative_lift': float(relative_lift),
            'significant': bool(p_value < alpha),
            'alpha': alpha
        }
    
    def calculate_sample_size(
        self,
        baseline_mean: float,
        mde: float,  # Minimum Detectable Effect (as percentage)
        baseline_std: float,
        alpha: float = 0.05,
        power: float = 0.80,
        ratio: float = 1.0
    ) -> Dict:
        """
        Calculate required sample size for A/B test
        
        Args:
            baseline_mean: Mean of the control group
            mde: Minimum detectable effect as percentage (e.g., 5 for 5%)
            baseline_std: Standard deviation of the metric
            alpha: Significance level
            power: Statistical power (1 - beta)
            ratio: Treatment to control ratio
            
        Returns:
            Dictionary with sample size calculations
        """
        # Convert MDE percentage to absolute difference
        effect_size = (mde / 100) * baseline_mean
        
        # Cohen's d
        cohens_d = effect_size / baseline_std
        
        if cohens_d == 0:
            raise ValueError("Baseline mean and MDE must produce a non-zero effect")
        
        # Calculate sample size per group
        n_control = tt_ind_solve_power(
            effect_size=abs(cohens_d),
            alpha=alpha,
            power=power,
            ratio=ratio,
            alternative='two-sided'
        )
        
        n_control = int(np.ceil(n_control))
        n_treatment = int(np.ceil(n_control * ratio))
        
        return {
            'n_control': n_control,
            'n_treatment': n_treatment,
            'total_sample_size': n_control + n_treatment,
            'cohens_d': float(cohens_d),
            'mde_absolute': float(effect_size),
            'mde_percentage': float(mde),
            'alpha': alpha,
            'power': power,
            'ratio': ratio
        }
    
    def bayesian_ab_test(
        self,
        control_success: int,
        control_total: int,
        treatment_success: int,
        treatment_total: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ) -> Dict:
        """
        Bayesian A/B test using Beta-Binomial conjugate priors
        
        Args:
            control_success: Successes in control
            control_total: Total in control
            treatment_success: Successes in treatment
            treatment_total: Total in treatment
            prior_alpha: Beta prior alpha parameter
            prior_beta: Beta prior beta parameter
            
        Returns:
            Dictionary with Bayesian results
        """
        if control_total <= 0 or treatment_total <= 0:
            raise ValueError("Group totals must be positive")
        if not (0 <= control_success <= control_total and 0 <= treatment_success <= treatment_total):
            raise ValueError("Successes must be between 0 and the group total")
        
        # Posterior parameters
        control_alpha = prior_alpha + control_success
        control_beta = prior_beta + (control_total - control_success)
        
        treatment_alpha = prior_alpha + treatment_success
        treatment_beta = prior_beta + (treatment_total - treatment_success)
        
        # Posterior means
        control_mean = control_alpha / (control_alpha + control_beta)
        treatment_mean = treatment_alpha / (treatment_alpha + treatment_beta)
        
        # Monte Carlo simulation to calculate probability that treatment > control
        # Local generator keeps results reproducible without touching global RNG state
        rng = np.random.default_rng(42)
        n_samples = 100000
        
        control_samples = rng.beta(control_alpha, control_beta, n_samples)
        treatment_samples = rng.beta(treatment_alpha, treatment_beta, n_samples)
        
        prob_treatment_better = np.mean(treatment_samples > control_samples)
        
        # Expected loss: how much conversion rate you give up, on average, if you
        # pick one arm and the other was actually better
        loss_if_ship_treatment = np.mean(np.maximum(control_samples - treatment_samples, 0))
        loss_if_keep_control = np.mean(np.maximum(treatment_samples - control_samples, 0))
        
        # Expected lift
        lift_samples = (treatment_samples - control_samples) / control_samples
        expected_lift = np.mean(lift_samples) * 100
        lift_ci_lower = np.percentile(lift_samples, 2.5) * 100
        lift_ci_upper = np.percentile(lift_samples, 97.5) * 100
        
        return {
            'test_type': 'bayesian',
            'control_posterior_alpha': float(control_alpha),
            'control_posterior_beta': float(control_beta),
            'treatment_posterior_alpha': float(treatment_alpha),
            'treatment_posterior_beta': float(treatment_beta),
            'control_mean': float(control_mean),
            'treatment_mean': float(treatment_mean),
            'prob_treatment_better': float(prob_treatment_better),
            'expected_lift': float(expected_lift),
            'lift_ci_lower': float(lift_ci_lower),
            'lift_ci_upper': float(lift_ci_upper),
            'expected_loss_treatment': float(loss_if_ship_treatment),
            'expected_loss_control': float(loss_if_keep_control),
            'control_n': int(control_total),
            'treatment_n': int(treatment_total)
        }
