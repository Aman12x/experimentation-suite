"""
Causal Inference Lab
Implements Propensity Score Matching, Difference-in-Differences, and Instrumental Variables
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Any, Dict, List, Optional


class CausalInferenceLab:
    """Causal inference methods for observational data"""

    def __init__(self):
        self.results: Dict = {}

    @staticmethod
    def _binary_indicator(series: pd.Series, positive_value: Any = None) -> pd.Series:
        """Turn a two-level column into a 0/1 indicator"""
        if positive_value is not None:
            if not (series == positive_value).any():
                raise ValueError(f"Value '{positive_value}' not found in column '{series.name}'")
            return (series == positive_value).astype(int)

        values = set(series.dropna().unique().tolist())
        if values <= {0, 1, True, False}:
            return series.astype(int)

        raise ValueError(
            f"Column '{series.name}' is not 0/1. Pass treated_value to say which "
            f"label marks the treated group (found: {sorted(map(str, values))[:5]})"
        )

    def propensity_score_matching(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: List[str],
        caliper: float = 0.1,
        matching_method: str = 'nearest',
        treated_value: Any = None
    ) -> Dict:
        """
        Propensity Score Matching for observational studies

        Args:
            df: DataFrame with treatment, outcome, and covariates
            treatment_col: Treatment indicator column (0/1, bool, or two labels)
            outcome_col: Outcome variable column
            covariate_cols: List of covariate columns for matching
            caliper: Maximum allowed propensity score distance for matching
            matching_method: 'nearest' (greedy 1:1 without replacement)
            treated_value: Label marking the treated group when the column is not 0/1

        Returns:
            Dictionary with ATT and matched sample info
        """
        if matching_method != 'nearest':
            raise ValueError("Only 'nearest' matching currently implemented")
        if not covariate_cols:
            raise ValueError("At least one covariate is required for matching")

        # Prepare data
        data = df[[treatment_col, outcome_col] + covariate_cols].dropna().copy()
        data['_treated'] = self._binary_indicator(data[treatment_col], treated_value)

        n_treated = int(data['_treated'].sum())
        n_control = int(len(data) - n_treated)
        if n_treated == 0 or n_control == 0:
            return {
                'error': 'Insufficient treated or control observations',
                'n_treated': n_treated,
                'n_control': n_control
            }

        # Fit propensity score model
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(data[covariate_cols], data['_treated'])
        data['propensity_score'] = ps_model.predict_proba(data[covariate_cols])[:, 1]

        treated = data[data['_treated'] == 1]
        control = data[data['_treated'] == 0]

        # Greedy 1:1 nearest-neighbour matching without replacement
        control_ps = control['propensity_score'].to_numpy()
        control_index = control.index.to_numpy()
        available = np.ones(len(control), dtype=bool)

        matched_treated_idx = []
        matched_control_idx = []

        for idx, ps_treated in treated['propensity_score'].items():
            if not available.any():
                break
            distance = np.where(available, np.abs(control_ps - ps_treated), np.inf)
            nearest = int(np.argmin(distance))

            # Apply caliper
            if distance[nearest] <= caliper:
                matched_treated_idx.append(idx)
                matched_control_idx.append(control_index[nearest])
                available[nearest] = False

        if len(matched_treated_idx) < 2:
            return {
                'error': 'Fewer than two matches found within the caliper',
                'n_treated': n_treated,
                'n_control': n_control
            }

        matched_treated = data.loc[matched_treated_idx]
        matched_control = data.loc[matched_control_idx]

        # ATT from matched-pair differences
        pair_diff = (
            matched_treated[outcome_col].to_numpy() - matched_control[outcome_col].to_numpy()
        )
        n_pairs = len(pair_diff)
        att = pair_diff.mean()
        se_att = pair_diff.std(ddof=1) / np.sqrt(n_pairs)

        if se_att > 0:
            t_stat = att / se_att
            p_value = 2 * stats.t.sf(abs(t_stat), n_pairs - 1)
        else:
            t_stat, p_value = 0.0, 1.0

        t_critical = stats.t.ppf(0.975, n_pairs - 1)
        ci_lower = att - t_critical * se_att
        ci_upper = att + t_critical * se_att

        # Balance diagnostics, before and after matching
        balance_stats = self._calculate_balance(
            matched_treated[covariate_cols],
            matched_control[covariate_cols],
            covariate_cols
        )
        balance_before = self._calculate_balance(
            treated[covariate_cols], control[covariate_cols], covariate_cols
        )
        balance_stats['std_mean_diff_before'] = balance_before['std_mean_diff']

        return {
            'method': 'Propensity Score Matching',
            'att': float(att),
            'se': float(se_att),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'n_treated_total': n_treated,
            'n_control_total': n_control,
            'n_matched': n_pairs,
            'match_rate': n_pairs / n_treated * 100,
            'treated_outcome_mean': float(matched_treated[outcome_col].mean()),
            'control_outcome_mean': float(matched_control[outcome_col].mean()),
            'max_abs_smd': float(balance_stats['std_mean_diff'].abs().max()),
            'balance_stats': balance_stats,
            'matched_treated': matched_treated,
            'matched_control': matched_control
        }

    def _calculate_balance(
        self,
        treated_covariates: pd.DataFrame,
        control_covariates: pd.DataFrame,
        covariate_cols: List[str]
    ) -> pd.DataFrame:
        """Calculate standardized mean differences for balance assessment"""
        balance = []

        for col in covariate_cols:
            treated_mean = treated_covariates[col].mean()
            control_mean = control_covariates[col].mean()

            pooled_std = np.sqrt(
                (treated_covariates[col].var() + control_covariates[col].var()) / 2
            )

            smd = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0

            balance.append({
                'covariate': col,
                'treated_mean': treated_mean,
                'control_mean': control_mean,
                'std_mean_diff': smd
            })

        return pd.DataFrame(balance)

    def difference_in_differences(
        self,
        df: pd.DataFrame,
        group_col: str,
        time_col: str,
        outcome_col: str,
        treatment_group: Any,
        post_period: Any,
        cluster_col: Optional[str] = None
    ) -> Dict:
        """
        Difference-in-Differences analysis

        Args:
            df: Panel data with group, time, and outcome
            group_col: Column identifying groups (treated vs control)
            time_col: Column identifying time periods (pre vs post)
            outcome_col: Outcome variable
            treatment_group: Value identifying the treatment group
            post_period: Value identifying the post-treatment period
            cluster_col: Optional unit column for cluster-robust standard errors

        Returns:
            Dictionary with DiD estimate and results
        """
        cols = [group_col, time_col, outcome_col] + ([cluster_col] if cluster_col else [])
        data = df[cols].dropna().copy()

        # Create treatment indicators
        data['treated'] = (data[group_col] == treatment_group).astype(int)
        data['post'] = (data[time_col] == post_period).astype(int)
        data['treated_post'] = data['treated'] * data['post']
        data['_outcome'] = data[outcome_col].astype(float)

        if data.groupby(['treated', 'post']).ngroups < 4:
            raise ValueError(
                "DiD needs observations in all four cells "
                "(treated/control x pre/post). Check the treatment group and post period values."
            )

        # Estimate DiD model: Y = β0 + β1*Treated + β2*Post + β3*Treated*Post + ε
        model = smf.ols("_outcome ~ treated + post + treated_post", data=data)
        if cluster_col:
            groups = pd.factorize(data[cluster_col])[0]
            fit = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
            se_type = f'cluster-robust ({cluster_col})'
        else:
            fit = model.fit(cov_type='HC1')
            se_type = 'heteroskedasticity-robust (HC1)'

        # DiD estimate is the coefficient on treated_post
        did_estimate = fit.params['treated_post']
        se = fit.bse['treated_post']
        p_value = fit.pvalues['treated_post']
        ci_lower, ci_upper = fit.conf_int().loc['treated_post']

        # Mean outcomes by cell for the trends plot
        means = data.groupby(['treated', 'post'])['_outcome'].mean().unstack()
        pre_diff = means.loc[1, 0] - means.loc[0, 0]
        post_diff = means.loc[1, 1] - means.loc[0, 1]

        return {
            'method': 'Difference-in-Differences',
            'did_estimate': float(did_estimate),
            'se': float(se),
            'se_type': se_type,
            'p_value': float(p_value),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'r_squared': float(fit.rsquared),
            'model_summary': fit.summary(),
            'pre_treatment_diff': float(pre_diff),
            'post_treatment_diff': float(post_diff),
            # A level gap between groups is allowed by DiD, and two periods carry
            # no information about trends, so this design cannot test the assumption.
            'parallel_trends_assumption': None,
            'parallel_trends_testable': False,
            'group_time_means': means,
            'n_observations': int(len(data)),
            'significant': bool(p_value < 0.05)
        }

    def event_study(
        self,
        df: pd.DataFrame,
        group_col: str,
        period_cols: List[str],
        outcome_col: str,
        treatment_group: Any,
        first_treated_period: Any,
        cluster_col: Optional[str] = None
    ) -> Dict:
        """
        Event-study DiD with a pre-trend test (needs several pre-treatment periods)
        
        Estimates a separate treated-vs-control gap for every period, relative
        to the last pre-treatment period. If trends were parallel before the
        intervention, the pre-period gaps are jointly zero; the Wald test on
        them is the usual check of the parallel-trends assumption.
        
        Args:
            df: Panel data
            group_col: Column identifying treated vs control groups
            period_cols: One or more columns that together order time (e.g. ['year', 'quarter'])
            outcome_col: Outcome variable
            treatment_group: Value identifying the treatment group
            first_treated_period: First treated period, as a value (one period
                column) or a tuple of values (several period columns)
            cluster_col: Optional unit column for cluster-robust standard errors
        """
        period_cols = [period_cols] if isinstance(period_cols, str) else list(period_cols)
        cols = [group_col, outcome_col] + period_cols + ([cluster_col] if cluster_col else [])
        data = df[cols].dropna().copy()
        
        # Collapse the period columns into one ordered index
        keys = data[period_cols].apply(tuple, axis=1)
        ordered = sorted(keys.unique())
        first = first_treated_period if isinstance(first_treated_period, tuple) else (first_treated_period,)
        if first not in ordered:
            raise ValueError(f"First treated period {first_treated_period} not found in the data")
        
        start = ordered.index(first)
        if start < 2:
            raise ValueError("A pre-trend test needs at least two pre-treatment periods")
        
        data['_t'] = keys.map({k: i for i, k in enumerate(ordered)}) - start  # 0 = first treated period
        data['treated'] = (data[group_col] == treatment_group).astype(int)
        data['_outcome'] = data[outcome_col].astype(float)
        if data['treated'].nunique() < 2:
            raise ValueError("Need both treated and control observations")
        
        # Period dummies and treated-by-period interactions, reference = last pre period (-1)
        design = pd.DataFrame({'const': 1.0, 'treated': data['treated'].astype(float)}, index=data.index)
        event_times = [t for t in sorted(data['_t'].unique()) if t != -1]
        for t in event_times:
            design[f'period_{t}'] = (data['_t'] == t).astype(float)
            design[f'gap_{t}'] = design[f'period_{t}'] * design['treated']
        
        model = sm.OLS(data['_outcome'], design)
        if cluster_col:
            fit = model.fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(data[cluster_col])[0]})
        else:
            fit = model.fit(cov_type='HC1')
        
        conf = fit.conf_int()
        labels = {i - start: ' / '.join(map(str, k)) for i, k in enumerate(ordered)}
        coefficients = pd.DataFrame([{
            'event_time': int(t),
            'period': labels[t],
            'estimate': float(fit.params[f'gap_{t}']),
            'ci_lower': float(conf.loc[f'gap_{t}', 0]),
            'ci_upper': float(conf.loc[f'gap_{t}', 1]),
            'p_value': float(fit.pvalues[f'gap_{t}'])
        } for t in event_times])
        
        pre_terms = [f'gap_{t}' for t in event_times if t < -1]
        restriction = np.zeros((len(pre_terms), len(fit.params)))
        for row, term in enumerate(pre_terms):
            restriction[row, list(fit.params.index).index(term)] = 1.0
        pre_test = fit.wald_test(restriction, use_f=False, scalar=True)
        pre_p = float(pre_test.pvalue)
        
        post = coefficients[coefficients.event_time >= 0]
        return {
            'method': 'Event Study',
            'coefficients': coefficients,
            'pre_trend_p_value': pre_p,
            'pre_trend_statistic': float(pre_test.statistic),
            'n_pre_periods_tested': len(pre_terms),
            # Failing to reject is supporting evidence, not proof
            'parallel_trends_assumption': bool(pre_p >= 0.05),
            'parallel_trends_testable': True,
            'average_post_effect': float(post['estimate'].mean()),
            'reference_period': labels[-1],
            'n_observations': int(len(data))
        }
    
    def instrumental_variables(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        instrument_col: str,
        covariate_cols: Optional[List[str]] = None
    ) -> Dict:
        """
        Two-Stage Least Squares (2SLS) estimation with instrumental variables

        Args:
            df: DataFrame
            outcome_col: Dependent variable
            treatment_col: Endogenous treatment variable
            instrument_col: Instrumental variable
            covariate_cols: Optional control variables

        Returns:
            Dictionary with IV estimates
        """
        from statsmodels.sandbox.regression.gmm import IV2SLS

        covariate_cols = list(covariate_cols or [])
        if len({outcome_col, treatment_col, instrument_col, *covariate_cols}) < 3 + len(covariate_cols):
            raise ValueError("Outcome, treatment, instrument, and covariates must be different columns")

        data = df[[outcome_col, treatment_col, instrument_col] + covariate_cols].dropna().astype(float)

        y = data[outcome_col]
        controls = sm.add_constant(data[covariate_cols], has_constant='add')
        exog = pd.concat([controls, data[[treatment_col]]], axis=1)
        instruments = pd.concat([controls, data[[instrument_col]]], axis=1)

        # First stage: Treatment ~ Instrument + Covariates
        first_stage = sm.OLS(data[treatment_col], instruments).fit()
        first_stage_restricted = sm.OLS(data[treatment_col], controls).fit()

        # Instrument strength: partial F for the excluded instrument only.
        # The overall first-stage F also credits the covariates and hides weak instruments.
        f_stat, f_pvalue, _ = first_stage.compare_f_test(first_stage_restricted)

        # 2SLS with standard errors built from the structural residuals
        iv_fit = IV2SLS(y, exog, instrument=instruments).fit()

        iv_estimate = iv_fit.params[treatment_col]
        se = iv_fit.bse[treatment_col]
        p_value = iv_fit.pvalues[treatment_col]
        ci_lower, ci_upper = iv_fit.conf_int().loc[treatment_col]

        # Naive OLS for comparison
        ols_fit = sm.OLS(y, exog).fit()

        return {
            'method': 'Instrumental Variables (2SLS)',
            'iv_estimate': float(iv_estimate),
            'se': float(se),
            'p_value': float(p_value),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'ols_estimate': float(ols_fit.params[treatment_col]),
            'first_stage_f_stat': float(f_stat),
            'first_stage_f_pvalue': float(f_pvalue),
            'weak_instrument': bool(f_stat < 10),  # Rule of thumb
            'first_stage_r2': float(first_stage.rsquared),
            'n_observations': int(len(data)),
            'significant': bool(p_value < 0.05)
        }
