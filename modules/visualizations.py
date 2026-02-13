"""
Visualizations Module
Creates interactive Plotly charts for experiment analysis
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class Visualizer:
    """Creates publication-quality visualizations for experiments"""
    
    def __init__(self):
        self.color_palette = {
            'control': '#3498db',
            'treatment': '#e74c3c',
            'neutral': '#95a5a6',
            'success': '#2ecc71',
            'warning': '#f39c12'
        }
    
    def plot_distribution_comparison(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        metric_name: str = "Metric"
    ) -> go.Figure:
        """
        Create overlapping histograms comparing distributions
        
        Args:
            control: Control group data
            treatment: Treatment group data
            metric_name: Name of the metric
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Control histogram
        fig.add_trace(go.Histogram(
            x=control,
            name='Control',
            opacity=0.7,
            marker_color=self.color_palette['control'],
            nbinsx=30
        ))
        
        # Treatment histogram
        fig.add_trace(go.Histogram(
            x=treatment,
            name='Treatment',
            opacity=0.7,
            marker_color=self.color_palette['treatment'],
            nbinsx=30
        ))
        
        fig.update_layout(
            title=f"Distribution Comparison: {metric_name}",
            xaxis_title=metric_name,
            yaxis_title="Frequency",
            barmode='overlay',
            template='plotly_white',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def plot_box_comparison(
        self,
        df: pd.DataFrame,
        group_col: str,
        metric_col: str
    ) -> go.Figure:
        """
        Create box plots comparing groups
        
        Args:
            df: DataFrame with data
            group_col: Group column
            metric_col: Metric column
            
        Returns:
            Plotly figure
        """
        fig = px.box(
            df,
            x=group_col,
            y=metric_col,
            color=group_col,
            color_discrete_map={
                df[group_col].unique()[0]: self.color_palette['control'],
                df[group_col].unique()[1]: self.color_palette['treatment']
            },
            points='outliers'
        )
        
        fig.update_layout(
            title=f"Box Plot Comparison: {metric_col}",
            xaxis_title="Group",
            yaxis_title=metric_col,
            template='plotly_white',
            showlegend=False,
            height=500
        )
        
        return fig
    
    def plot_confidence_interval(
        self,
        results: Dict,
        metric_name: str = "Difference"
    ) -> go.Figure:
        """
        Plot confidence interval for treatment effect
        
        Args:
            results: Results dictionary from A/B test
            metric_name: Name of the metric
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Point estimate
        point_estimate = results.get('mean_difference', results.get('att', 0))
        ci_lower = results['ci_lower']
        ci_upper = results['ci_upper']
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=[ci_lower, point_estimate, ci_upper],
            y=[metric_name, metric_name, metric_name],
            mode='markers+lines',
            marker=dict(size=[10, 15, 10], color=self.color_palette['treatment']),
            line=dict(width=3, color=self.color_palette['treatment']),
            name='95% CI'
        ))
        
        # Add zero line
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color=self.color_palette['neutral'],
            annotation_text="No Effect"
        )
        
        fig.update_layout(
            title=f"Treatment Effect: {metric_name}",
            xaxis_title="Effect Size",
            yaxis_title="",
            template='plotly_white',
            showlegend=False,
            height=300,
            yaxis=dict(showticklabels=False)
        )
        
        return fig
    
    def plot_power_analysis(
        self,
        effect_sizes: np.ndarray,
        sample_sizes: List[int],
        alpha: float = 0.05,
        power: float = 0.80
    ) -> go.Figure:
        """
        Plot power analysis curves
        
        Args:
            effect_sizes: Array of effect sizes
            sample_sizes: List of sample sizes to plot
            alpha: Significance level
            power: Target power
            
        Returns:
            Plotly figure
        """
        from statsmodels.stats.power import tt_ind_solve_power
        
        fig = go.Figure()
        
        for n in sample_sizes:
            powers = []
            for effect in effect_sizes:
                try:
                    p = tt_ind_solve_power(
                        effect_size=effect,
                        nobs1=n,
                        alpha=alpha,
                        ratio=1.0,
                        alternative='two-sided'
                    )
                    powers.append(p)
                except:
                    powers.append(np.nan)
            
            fig.add_trace(go.Scatter(
                x=effect_sizes,
                y=powers,
                mode='lines',
                name=f'n={n}',
                line=dict(width=2)
            ))
        
        # Add target power line
        fig.add_hline(
            y=power,
            line_dash="dash",
            line_color=self.color_palette['success'],
            annotation_text=f"Target Power ({power})"
        )
        
        fig.update_layout(
            title="Statistical Power Analysis",
            xaxis_title="Effect Size (Cohen's d)",
            yaxis_title="Statistical Power",
            template='plotly_white',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def plot_bayesian_posteriors(
        self,
        control_alpha: float,
        control_beta: float,
        treatment_alpha: float,
        treatment_beta: float
    ) -> go.Figure:
        """
        Plot Bayesian posterior distributions
        
        Args:
            control_alpha: Control posterior alpha
            control_beta: Control posterior beta
            treatment_alpha: Treatment posterior alpha
            treatment_beta: Treatment posterior beta
            
        Returns:
            Plotly figure
        """
        from scipy.stats import beta
        
        x = np.linspace(0, 1, 1000)
        
        control_posterior = beta.pdf(x, control_alpha, control_beta)
        treatment_posterior = beta.pdf(x, treatment_alpha, treatment_beta)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x,
            y=control_posterior,
            mode='lines',
            name='Control',
            fill='tozeroy',
            line=dict(color=self.color_palette['control'], width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=x,
            y=treatment_posterior,
            mode='lines',
            name='Treatment',
            fill='tozeroy',
            line=dict(color=self.color_palette['treatment'], width=2),
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Bayesian Posterior Distributions",
            xaxis_title="Conversion Rate",
            yaxis_title="Density",
            template='plotly_white',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def plot_propensity_scores(
        self,
        matched_treated: pd.DataFrame,
        matched_control: pd.DataFrame
    ) -> go.Figure:
        """
        Plot propensity score distributions before and after matching
        
        Args:
            matched_treated: Matched treated units with propensity scores
            matched_control: Matched control units with propensity scores
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=matched_treated['propensity_score'],
            name='Treated',
            opacity=0.7,
            marker_color=self.color_palette['treatment'],
            nbinsx=30
        ))
        
        fig.add_trace(go.Histogram(
            x=matched_control['propensity_score'],
            name='Control (Matched)',
            opacity=0.7,
            marker_color=self.color_palette['control'],
            nbinsx=30
        ))
        
        fig.update_layout(
            title="Propensity Score Distribution After Matching",
            xaxis_title="Propensity Score",
            yaxis_title="Frequency",
            barmode='overlay',
            template='plotly_white',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def plot_did_trends(
        self,
        group_time_means: pd.DataFrame
    ) -> go.Figure:
        """
        Plot parallel trends for Difference-in-Differences
        
        Args:
            group_time_means: DataFrame with means by group and time
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        groups = group_time_means.index
        times = group_time_means.columns
        
        for group in groups:
            fig.add_trace(go.Scatter(
                x=times,
                y=group_time_means.loc[group],
                mode='lines+markers',
                name='Treatment' if group == 1 else 'Control',
                line=dict(
                    width=3,
                    color=self.color_palette['treatment'] if group == 1 else self.color_palette['control']
                ),
                marker=dict(size=10)
            ))
        
        # Add vertical line at treatment time
        if len(times) >= 2:
            fig.add_vline(
                x=times[-1],
                line_dash="dash",
                line_color=self.color_palette['neutral'],
                annotation_text="Treatment"
            )
        
        fig.update_layout(
            title="Difference-in-Differences: Parallel Trends",
            xaxis_title="Time Period",
            yaxis_title="Mean Outcome",
            template='plotly_white',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def plot_balance_diagnostics(
        self,
        balance_stats: pd.DataFrame
    ) -> go.Figure:
        """
        Plot standardized mean differences for covariate balance
        
        Args:
            balance_stats: DataFrame with balance statistics
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Sort by absolute SMD
        balance_stats = balance_stats.copy()
        balance_stats['abs_smd'] = abs(balance_stats['std_mean_diff'])
        balance_stats = balance_stats.sort_values('abs_smd', ascending=True)
        
        # Color based on threshold
        colors = [
            self.color_palette['success'] if abs(smd) < 0.1 else self.color_palette['warning']
            for smd in balance_stats['std_mean_diff']
        ]
        
        fig.add_trace(go.Bar(
            y=balance_stats['covariate'],
            x=balance_stats['std_mean_diff'],
            orientation='h',
            marker_color=colors
        ))
        
        # Add threshold lines
        fig.add_vline(x=-0.1, line_dash="dash", line_color=self.color_palette['neutral'])
        fig.add_vline(x=0.1, line_dash="dash", line_color=self.color_palette['neutral'])
        fig.add_vline(x=0, line_color='black', line_width=1)
        
        fig.update_layout(
            title="Covariate Balance: Standardized Mean Differences",
            xaxis_title="Standardized Mean Difference",
            yaxis_title="Covariate",
            template='plotly_white',
            height=max(400, len(balance_stats) * 30)
        )
        
        return fig
