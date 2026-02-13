"""
Causal Inference Lab
Implements Propensity Score Matching, Difference-in-Differences, and Causal Graphs
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import statsmodels.formula.api as smf
from typing import Dict, List, Optional, Tuple
import streamlit as st


class CausalInferenceLab:
    """Causal inference methods for observational data"""
    
    def __init__(self):
        self.results: Dict = {}
    
    def propensity_score_matching(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: List[str],
        caliper: float = 0.1,
        matching_method: str = 'nearest'
    ) -> Dict:
        """
        Propensity Score Matching for observational studies
        
        Args:
            df: DataFrame with treatment, outcome, and covariates
            treatment_col: Binary treatment indicator column
            outcome_col: Outcome variable column
            covariate_cols: List of covariate columns for matching
            caliper: Maximum allowed distance for matching
            matching_method: 'nearest' or 'radius'
            
        Returns:
            Dictionary with ATT and matched sample info
        """
        # Prepare data
        data = df[[treatment_col, outcome_col] + covariate_cols].dropna()
        
        # Fit propensity score model
        X = data[covariate_cols]
        y = data[treatment_col]
        
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X, y)
        
        # Get propensity scores
        data['propensity_score'] = ps_model.predict_proba(X)[:, 1]
        
        # Separate treated and control
        treated = data[data[treatment_col] == 1].copy()
        control = data[data[treatment_col] == 0].copy()
        
        if len(treated) == 0 or len(control) == 0:
            return {
                'error': 'Insufficient treated or control observations',
                'n_treated': len(treated),
                'n_control': len(control)
            }
        
        # Matching
        if matching_method == 'nearest':
            matched_control_idx = []
            matched_treated_idx = []
            
            for idx, treated_row in treated.iterrows():
                ps_treated = treated_row['propensity_score']
                
                # Find nearest control
                control_copy = control.copy()
                control_copy['ps_diff'] = abs(control_copy['propensity_score'] - ps_treated)
                
                # Apply caliper
                valid_matches = control_copy[control_copy['ps_diff'] <= caliper]
                
                if len(valid_matches) > 0:
                    nearest_idx = valid_matches['ps_diff'].idxmin()
                    matched_control_idx.append(nearest_idx)
                    matched_treated_idx.append(idx)
                    
                    # Remove matched control to avoid reuse
                    control = control.drop(nearest_idx)
            
            # Create matched sample
            matched_treated = treated.loc[matched_treated_idx]
            matched_control = data.loc[matched_control_idx]
            
        else:
            st.error("Only 'nearest' matching currently implemented")
            return {}
        
        # Calculate ATT (Average Treatment Effect on the Treated)
        att = matched_treated[outcome_col].mean() - matched_control[outcome_col].mean()
        
        # Standard error and confidence interval
        se_att = np.sqrt(
            matched_treated[outcome_col].var() / len(matched_treated) +
            matched_control[outcome_col].var() / len(matched_control)
        )
        
        t_critical = stats.t.ppf(0.975, len(matched_treated) + len(matched_control) - 2)
        ci_lower = att - t_critical * se_att
        ci_upper = att + t_critical * se_att
        
        # T-test
        t_stat, p_value = stats.ttest_ind(
            matched_treated[outcome_col],
            matched_control[outcome_col]
        )
        
        # Balance diagnostics
        balance_stats = self._calculate_balance(
            matched_treated[covariate_cols],
            matched_control[covariate_cols],
            covariate_cols
        )
        
        return {
            'method': 'Propensity Score Matching',
            'att': att,
            'se': se_att,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_statistic': t_stat,
            'p_value': p_value,
            'n_treated_total': len(treated),
            'n_control_total': len(control) + len(matched_control),
            'n_matched': len(matched_treated),
            'match_rate': len(matched_treated) / len(treated) * 100,
            'treated_outcome_mean': matched_treated[outcome_col].mean(),
            'control_outcome_mean': matched_control[outcome_col].mean(),
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
        treatment_group: any,
        post_period: any
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
            
        Returns:
            Dictionary with DiD estimate and results
        """
        # Create treatment indicators
        df = df.copy()
        df['treated'] = (df[group_col] == treatment_group).astype(int)
        df['post'] = (df[time_col] == post_period).astype(int)
        df['treated_post'] = df['treated'] * df['post']
        
        # Estimate DiD model: Y = β0 + β1*Treated + β2*Post + β3*Treated*Post + ε
        formula = f"{outcome_col} ~ treated + post + treated_post"
        model = smf.ols(formula, data=df).fit()
        
        # DiD estimate is the coefficient on treated_post
        did_estimate = model.params['treated_post']
        se = model.bse['treated_post']
        p_value = model.pvalues['treated_post']
        ci_lower, ci_upper = model.conf_int().loc['treated_post']
        
        # Calculate mean outcomes for parallel trends visualization
        means = df.groupby(['treated', 'post'])[outcome_col].mean().unstack()
        
        # Pre-treatment difference
        pre_diff = means.loc[1, 0] - means.loc[0, 0] if 0 in means.columns else 0
        
        # Post-treatment difference
        post_diff = means.loc[1, 1] - means.loc[0, 1] if 1 in means.columns else 0
        
        return {
            'method': 'Difference-in-Differences',
            'did_estimate': did_estimate,
            'se': se,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'r_squared': model.rsquared,
            'model_summary': model.summary(),
            'pre_treatment_diff': pre_diff,
            'post_treatment_diff': post_diff,
            'parallel_trends_assumption': abs(pre_diff) < abs(did_estimate) * 0.1,
            'group_time_means': means,
            'n_observations': len(df),
            'significant': p_value < 0.05
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
        try:
            from statsmodels.sandbox.regression.gmm import IV2SLS
            
            # Prepare data
            data = df[[outcome_col, treatment_col, instrument_col]].copy()
            if covariate_cols:
                data = df[[outcome_col, treatment_col, instrument_col] + covariate_cols].copy()
            data = data.dropna()
            
            # First stage: Treatment ~ Instrument + Covariates
            if covariate_cols:
                first_stage_formula = f"{treatment_col} ~ {instrument_col} + {' + '.join(covariate_cols)}"
            else:
                first_stage_formula = f"{treatment_col} ~ {instrument_col}"
            
            first_stage = smf.ols(first_stage_formula, data=data).fit()
            
            # Check instrument strength (F-statistic)
            f_stat = first_stage.fvalue
            
            # Second stage: Outcome ~ Predicted_Treatment + Covariates
            data['predicted_treatment'] = first_stage.fittedvalues
            
            if covariate_cols:
                second_stage_formula = f"{outcome_col} ~ predicted_treatment + {' + '.join(covariate_cols)}"
            else:
                second_stage_formula = f"{outcome_col} ~ predicted_treatment"
            
            second_stage = smf.ols(second_stage_formula, data=data).fit()
            
            # IV estimate
            iv_estimate = second_stage.params['predicted_treatment']
            se = second_stage.bse['predicted_treatment']
            p_value = second_stage.pvalues['predicted_treatment']
            ci_lower, ci_upper = second_stage.conf_int().loc['predicted_treatment']
            
            return {
                'method': 'Instrumental Variables (2SLS)',
                'iv_estimate': iv_estimate,
                'se': se,
                'p_value': p_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'first_stage_f_stat': f_stat,
                'weak_instrument': f_stat < 10,  # Rule of thumb
                'first_stage_r2': first_stage.rsquared,
                'second_stage_r2': second_stage.rsquared,
                'n_observations': len(data)
            }
            
        except ImportError:
            st.warning("IV estimation requires statsmodels. Simplified estimation used.")
            return {}
