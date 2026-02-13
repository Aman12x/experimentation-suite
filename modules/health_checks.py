"""
Health Checks Module
Automated data quality checks and Sample Ratio Mismatch (SRM) detection
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


class HealthChecker:
    """Automated health checks for A/B tests"""
    
    def __init__(self):
        self.warnings: List[str] = []
        self.checks_passed: List[str] = []
    
    def run_all_checks(
        self,
        df: pd.DataFrame,
        group_col: str,
        metric_col: str,
        expected_ratio: Tuple[float, float] = (0.5, 0.5)
    ) -> Dict:
        """
        Run all health checks on the experiment data
        
        Args:
            df: DataFrame with experiment data
            group_col: Group assignment column
            metric_col: Metric column
            expected_ratio: Expected (control, treatment) ratio
            
        Returns:
            Dictionary with all check results
        """
        self.warnings = []
        self.checks_passed = []
        
        # Run checks
        srm_result = self.check_sample_ratio_mismatch(df, group_col, expected_ratio)
        outliers_result = self.check_outliers(df, metric_col)
        missing_result = self.check_missing_data(df, group_col, metric_col)
        variance_result = self.check_variance_ratio(df, group_col, metric_col)
        normality_result = self.check_normality(df, group_col, metric_col)
        
        return {
            'sample_ratio_mismatch': srm_result,
            'outliers': outliers_result,
            'missing_data': missing_result,
            'variance_ratio': variance_result,
            'normality': normality_result,
            'warnings': self.warnings,
            'checks_passed': self.checks_passed,
            'overall_health': len(self.warnings) == 0
        }
    
    def check_sample_ratio_mismatch(
        self,
        df: pd.DataFrame,
        group_col: str,
        expected_ratio: Tuple[float, float] = (0.5, 0.5),
        alpha: float = 0.001  # More stringent for SRM
    ) -> Dict:
        """
        Check for Sample Ratio Mismatch (SRM)
        
        SRM occurs when the observed traffic split differs significantly
        from the expected split, indicating data quality issues.
        
        Args:
            df: DataFrame
            group_col: Column with group assignments
            expected_ratio: Expected (control, treatment) proportions
            alpha: Significance level (typically 0.001 for SRM)
            
        Returns:
            Dictionary with SRM check results
        """
        # Get group counts
        groups = df[group_col].value_counts().sort_index()
        
        if len(groups) != 2:
            return {'error': 'SRM check requires exactly 2 groups'}
        
        observed = groups.values
        total = observed.sum()
        
        # Expected counts based on ratio
        expected = np.array([
            expected_ratio[0] * total,
            expected_ratio[1] * total
        ])
        
        # Chi-squared test
        chi2_stat = np.sum((observed - expected)**2 / expected)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        
        # Actual proportions
        actual_proportions = observed / total
        
        has_srm = p_value < alpha
        
        if has_srm:
            self.warnings.append(
                f"⚠️ SAMPLE RATIO MISMATCH DETECTED (p={p_value:.6f}). "
                f"Expected {expected_ratio}, got ({actual_proportions[0]:.3f}, {actual_proportions[1]:.3f}). "
                "This suggests data quality issues - investigate before interpreting results!"
            )
        else:
            self.checks_passed.append("✓ No Sample Ratio Mismatch detected")
        
        return {
            'has_srm': has_srm,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'expected_counts': expected,
            'observed_counts': observed,
            'expected_proportions': expected_ratio,
            'actual_proportions': actual_proportions.tolist(),
            'severity': 'CRITICAL' if has_srm else 'OK'
        }
    
    def check_outliers(
        self,
        df: pd.DataFrame,
        metric_col: str,
        iqr_multiplier: float = 3.0
    ) -> Dict:
        """
        Check for outliers using IQR method
        
        Args:
            df: DataFrame
            metric_col: Metric column to check
            iqr_multiplier: IQR multiplier for outlier threshold
            
        Returns:
            Dictionary with outlier information
        """
        data = df[metric_col].dropna()
        
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        n_outliers = len(outliers)
        outlier_pct = (n_outliers / len(data)) * 100
        
        if outlier_pct > 5:
            self.warnings.append(
                f"⚠️ {outlier_pct:.1f}% of data are outliers. "
                "Consider robust statistical methods or data cleaning."
            )
        else:
            self.checks_passed.append(
                f"✓ Outliers within acceptable range ({outlier_pct:.1f}%)"
            )
        
        return {
            'n_outliers': n_outliers,
            'outlier_percentage': outlier_pct,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'iqr': iqr,
            'severity': 'WARNING' if outlier_pct > 5 else 'OK'
        }
    
    def check_missing_data(
        self,
        df: pd.DataFrame,
        group_col: str,
        metric_col: str
    ) -> Dict:
        """
        Check for missing data patterns
        
        Args:
            df: DataFrame
            group_col: Group column
            metric_col: Metric column
            
        Returns:
            Dictionary with missing data information
        """
        total_rows = len(df)
        
        # Overall missing
        missing_group = df[group_col].isnull().sum()
        missing_metric = df[metric_col].isnull().sum()
        
        missing_group_pct = (missing_group / total_rows) * 100
        missing_metric_pct = (missing_metric / total_rows) * 100
        
        # Check if missing is different across groups
        missing_by_group = df.groupby(group_col)[metric_col].apply(
            lambda x: x.isnull().sum() / len(x) * 100
        )
        
        if missing_metric_pct > 10:
            self.warnings.append(
                f"⚠️ {missing_metric_pct:.1f}% missing data in metric. "
                "This may bias results."
            )
        else:
            self.checks_passed.append(
                f"✓ Missing data acceptable ({missing_metric_pct:.1f}%)"
            )
        
        return {
            'total_rows': total_rows,
            'missing_group': missing_group,
            'missing_metric': missing_metric,
            'missing_group_pct': missing_group_pct,
            'missing_metric_pct': missing_metric_pct,
            'missing_by_group': missing_by_group.to_dict(),
            'severity': 'WARNING' if missing_metric_pct > 10 else 'OK'
        }
    
    def check_variance_ratio(
        self,
        df: pd.DataFrame,
        group_col: str,
        metric_col: str,
        threshold: float = 4.0
    ) -> Dict:
        """
        Check variance ratio between groups (Levene's test)
        
        Large variance differences can affect test validity
        
        Args:
            df: DataFrame
            group_col: Group column
            metric_col: Metric column
            threshold: Variance ratio threshold for warning
            
        Returns:
            Dictionary with variance check results
        """
        groups = df.groupby(group_col)[metric_col].apply(list)
        
        if len(groups) != 2:
            return {'error': 'Variance check requires exactly 2 groups'}
        
        group_data = [np.array(g) for g in groups.values]
        
        # Levene's test for equal variances
        statistic, p_value = stats.levene(*group_data)
        
        # Variance ratio
        variances = [np.var(g, ddof=1) for g in group_data]
        variance_ratio = max(variances) / min(variances)
        
        if variance_ratio > threshold:
            self.warnings.append(
                f"⚠️ Large variance ratio ({variance_ratio:.2f}). "
                "Consider using Welch's t-test or transformation."
            )
        else:
            self.checks_passed.append(
                f"✓ Variance ratio acceptable ({variance_ratio:.2f})"
            )
        
        return {
            'variance_ratio': variance_ratio,
            'levene_statistic': statistic,
            'levene_p_value': p_value,
            'equal_variances': p_value > 0.05,
            'group_variances': {str(k): v for k, v in zip(groups.index, variances)},
            'severity': 'WARNING' if variance_ratio > threshold else 'OK'
        }
    
    def check_normality(
        self,
        df: pd.DataFrame,
        group_col: str,
        metric_col: str,
        sample_size_threshold: int = 30
    ) -> Dict:
        """
        Check normality assumption using Shapiro-Wilk test
        
        Args:
            df: DataFrame
            group_col: Group column
            metric_col: Metric column
            sample_size_threshold: Sample size below which normality is important
            
        Returns:
            Dictionary with normality check results
        """
        groups = df.groupby(group_col)[metric_col].apply(list)
        
        normality_results = {}
        all_normal = True
        
        for group_name, group_data in groups.items():
            group_array = np.array(group_data)
            
            if len(group_array) < 3:
                normality_results[str(group_name)] = {
                    'error': 'Sample too small for normality test'
                }
                continue
            
            # Shapiro-Wilk test (for samples < 5000)
            if len(group_array) <= 5000:
                statistic, p_value = stats.shapiro(group_array)
                is_normal = p_value > 0.05
            else:
                # Use Kolmogorov-Smirnov for larger samples
                statistic, p_value = stats.kstest(
                    group_array,
                    lambda x: stats.norm.cdf(x, np.mean(group_array), np.std(group_array, ddof=1))
                )
                is_normal = p_value > 0.05
            
            normality_results[str(group_name)] = {
                'is_normal': is_normal,
                'statistic': statistic,
                'p_value': p_value,
                'sample_size': len(group_array)
            }
            
            if not is_normal:
                all_normal = False
        
        # Check if normality matters
        min_sample_size = min(len(g) for g in groups.values)
        normality_matters = min_sample_size < sample_size_threshold
        
        if not all_normal and normality_matters:
            self.warnings.append(
                "⚠️ Data may not be normally distributed with small sample size. "
                "Consider non-parametric tests or larger sample."
            )
        else:
            self.checks_passed.append(
                "✓ Normality assumption met or sample size sufficient"
            )
        
        return {
            'all_normal': all_normal,
            'normality_matters': normality_matters,
            'min_sample_size': min_sample_size,
            'by_group': normality_results,
            'severity': 'WARNING' if (not all_normal and normality_matters) else 'OK'
        }
