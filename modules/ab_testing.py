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
import streamlit as st


class ABTestingEngine:
    """Comprehensive A/B testing engine with multiple statistical methods"""
    
    def __init__(self):
        self.results: Dict = {}
    
    def t_test(
        self, 
        control: np.ndarray, 
        treatment: np.ndarray,
        alternative: str = 'two-sided',
        alpha: float = 0.05
    ) -> Dict:
        """
        Perform independent samples t-test
        
        Args:
            control: Control group data
            treatment: Treatment group data
            alternative: 'two-sided', 'greater', or 'less'
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(treatment, control, alternative=alternative)
        
        # Calculate statistics
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        pooled_std = np.sqrt(
            ((len(control) - 1) * np.var(control, ddof=1) + 
             (len(treatment) - 1) * np.var(treatment, ddof=1)) / 
            (len(control) + len(treatment) - 2)
        )
        
        # Effect size (Cohen's d)
        cohens_d = (treatment_mean - control_mean) / pooled_std
        
        # Confidence interval for difference in means
        se = pooled_std * np.sqrt(1/len(control) + 1/len(treatment))
        df = len(control) + len(treatment) - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)
        ci_lower = (treatment_mean - control_mean) - t_critical * se
        ci_upper = (treatment_mean - control_mean) + t_critical * se
        
        # Relative lift
        relative_lift = ((treatment_mean - control_mean) / control_mean) * 100
        
        return {
            'test_type': 't-test',
            't_statistic': t_stat,
            'p_value': p_value,
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'control_std': np.std(control, ddof=1),
            'treatment_std': np.std(treatment, ddof=1),
            'control_n': len(control),
            'treatment_n': len(treatment),
            'mean_difference': treatment_mean - control_mean,
            'cohens_d': cohens_d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'relative_lift': relative_lift,
            'significant': p_value < alpha,
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
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        control_std = np.std(control, ddof=1)
        treatment_std = np.std(treatment, ddof=1)
        
        # Z-statistic
        se = np.sqrt((control_std**2 / len(control)) + (treatment_std**2 / len(treatment)))
        z_stat = (treatment_mean - control_mean) / se
        
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
        cohens_d = (treatment_mean - control_mean) / pooled_std
        
        relative_lift = ((treatment_mean - control_mean) / control_mean) * 100
        
        return {
            'test_type': 'z-test',
            'z_statistic': z_stat,
            'p_value': p_value,
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'control_std': control_std,
            'treatment_std': treatment_std,
            'control_n': len(control),
            'treatment_n': len(treatment),
            'mean_difference': treatment_mean - control_mean,
            'cohens_d': cohens_d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'relative_lift': relative_lift,
            'significant': p_value < alpha,
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
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'control_n': control_total,
            'treatment_n': treatment_total,
            'rate_difference': diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'relative_lift': relative_lift,
            'significant': p_value < alpha,
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
        
        # Calculate sample size per group
        try:
            n_control = tt_ind_solve_power(
                effect_size=cohens_d,
                alpha=alpha,
                power=power,
                ratio=ratio,
                alternative='two-sided'
            )
            
            n_treatment = n_control * ratio
            total_n = n_control + n_treatment
            
            return {
                'n_control': int(np.ceil(n_control)),
                'n_treatment': int(np.ceil(n_treatment)),
                'total_sample_size': int(np.ceil(total_n)),
                'cohens_d': cohens_d,
                'mde_absolute': effect_size,
                'mde_percentage': mde,
                'alpha': alpha,
                'power': power,
                'ratio': ratio
            }
        except Exception as e:
            st.error(f"Error calculating sample size: {str(e)}")
            return {}
    
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
        # Posterior parameters
        control_alpha = prior_alpha + control_success
        control_beta = prior_beta + (control_total - control_success)
        
        treatment_alpha = prior_alpha + treatment_success
        treatment_beta = prior_beta + (treatment_total - treatment_success)
        
        # Posterior means
        control_mean = control_alpha / (control_alpha + control_beta)
        treatment_mean = treatment_alpha / (treatment_alpha + treatment_beta)
        
        # Monte Carlo simulation to calculate probability that treatment > control
        np.random.seed(42)
        n_samples = 100000
        
        control_samples = np.random.beta(control_alpha, control_beta, n_samples)
        treatment_samples = np.random.beta(treatment_alpha, treatment_beta, n_samples)
        
        prob_treatment_better = np.mean(treatment_samples > control_samples)
        
        # Expected lift
        lift_samples = (treatment_samples - control_samples) / control_samples
        expected_lift = np.mean(lift_samples) * 100
        lift_ci_lower = np.percentile(lift_samples, 2.5) * 100
        lift_ci_upper = np.percentile(lift_samples, 97.5) * 100
        
        return {
            'test_type': 'bayesian',
            'control_posterior_alpha': control_alpha,
            'control_posterior_beta': control_beta,
            'treatment_posterior_alpha': treatment_alpha,
            'treatment_posterior_beta': treatment_beta,
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'prob_treatment_better': prob_treatment_better,
            'expected_lift': expected_lift,
            'lift_ci_lower': lift_ci_lower,
            'lift_ci_upper': lift_ci_upper,
            'control_n': control_total,
            'treatment_n': treatment_total
        }
