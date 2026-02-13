"""
Experimentation & Causal Analysis Suite
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from modules import (
    DataHandler,
    ABTestingEngine,
    CausalInferenceLab,
    HealthChecker,
    Visualizer
)
from utils import StatisticalInterpreter, ReportGenerator

# Page configuration
st.set_page_config(
    page_title="Experimentation & Causal Analysis Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_handler' not in st.session_state:
    st.session_state.data_handler = DataHandler()
if 'ab_engine' not in st.session_state:
    st.session_state.ab_engine = ABTestingEngine()
if 'causal_lab' not in st.session_state:
    st.session_state.causal_lab = CausalInferenceLab()
if 'health_checker' not in st.session_state:
    st.session_state.health_checker = HealthChecker()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()
if 'interpreter' not in st.session_state:
    st.session_state.interpreter = StatisticalInterpreter()
if 'report_gen' not in st.session_state:
    st.session_state.report_gen = ReportGenerator()

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #ecf0f1;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔬 Experimentation & Causal Analysis Suite</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Professional-grade statistical testing and causal inference platform</p>', unsafe_allow_html=True)

# Sidebar - Data Upload
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV or Parquet)",
        type=['csv', 'parquet'],
        help="Upload your experiment data to begin analysis"
    )
    
    if uploaded_file:
        try:
            data = st.session_state.data_handler.load_data(uploaded_file)
            st.success(f"✅ Loaded {len(data):,} rows × {len(data.columns)} columns")
            
            st.subheader("📊 Data Info")
            st.write(f"**Numeric columns:** {len(st.session_state.data_handler.numeric_columns)}")
            st.write(f"**Categorical columns:** {len(st.session_state.data_handler.categorical_columns)}")
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

    st.divider()
    
    # Quick Help
    with st.expander("ℹ️ Quick Help"):
        st.markdown("""
        **Getting Started:**
        1. Upload your dataset
        2. Choose your analysis type
        3. Configure parameters
        4. Review results and export
        
        **Data Requirements:**
        - CSV or Parquet format
        - For A/B tests: Group column + Metric column
        - For causal inference: Treatment, outcome, and covariates
        """)

# Main content - Tabs
if uploaded_file:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview",
        "🧪 A/B Testing",
        "🎯 Causal Inference",
        "📄 Report Export"
    ])
    
    # ==================== TAB 1: DATA OVERVIEW ====================
    with tab1:
        st.header("📊 Data Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Dataset Preview")
            st.dataframe(
                st.session_state.data_handler.data.head(100),
                use_container_width=True,
                height=400
            )
        
        with col2:
            st.subheader("Summary Statistics")
            summary = st.session_state.data_handler.get_summary_stats()
            st.dataframe(summary, use_container_width=True, height=400)
        
        # Descriptive statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(
            st.session_state.data_handler.data.describe(),
            use_container_width=True
        )
    
    # ==================== TAB 2: A/B TESTING ====================
    with tab2:
        st.header("🧪 A/B Testing Engine")
        
        # Configuration
        with st.expander("⚙️ Test Configuration", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                group_col = st.selectbox(
                    "Group Column",
                    options=st.session_state.data_handler.categorical_columns,
                    help="Column containing control/treatment assignments"
                )
            
            with col2:
                metric_col = st.selectbox(
                    "Metric Column",
                    options=st.session_state.data_handler.numeric_columns,
                    help="Numeric metric to analyze"
                )
            
            with col3:
                test_type = st.selectbox(
                    "Test Type",
                    options=['T-Test', 'Z-Test', 'Chi-Squared', 'Bayesian'],
                    help="Statistical test to perform"
                )
            
            alpha = st.slider(
                "Significance Level (α)",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                help="Probability of Type I error"
            )
        
        # Validate and run test
        if st.button("🚀 Run A/B Test", type="primary", use_container_width=True):
            
            # Validation
            is_valid, msg = st.session_state.data_handler.validate_ab_test_columns(
                group_col, metric_col
            )
            
            if not is_valid:
                st.error(f"❌ Validation Error: {msg}")
            else:
                with st.spinner("Running analysis..."):
                    # Prepare data
                    df = st.session_state.data_handler.prepare_ab_data(
                        group_col, metric_col
                    )
                    
                    # Run health checks
                    st.subheader("🏥 Health Checks")
                    health_results = st.session_state.health_checker.run_all_checks(
                        df, group_col, metric_col
                    )
                    
                    if health_results['overall_health']:
                        st.success("✅ All health checks passed!")
                    else:
                        st.warning("⚠️ Some health checks failed. Review warnings below.")
                    
                    for warning in health_results['warnings']:
                        st.markdown(f'<div class="warning-box">{warning}</div>', unsafe_allow_html=True)
                    
                    for check in health_results['checks_passed']:
                        st.markdown(f'<div class="success-box">{check}</div>', unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Get groups
                    groups = df[group_col].unique()
                    control_group = groups[0]
                    treatment_group = groups[1]
                    
                    control_data = df[df[group_col] == control_group][metric_col].values
                    treatment_data = df[df[group_col] == treatment_group][metric_col].values
                    
                    # Run test based on type
                    if test_type == 'T-Test':
                        results = st.session_state.ab_engine.t_test(
                            control_data, treatment_data, alpha=alpha
                        )
                    elif test_type == 'Z-Test':
                        results = st.session_state.ab_engine.z_test(
                            control_data, treatment_data, alpha=alpha
                        )
                    elif test_type == 'Chi-Squared':
                        # For chi-squared, need binary conversion
                        st.info("ℹ️ Converting continuous metric to binary (above median = success)")
                        control_median = np.median(control_data)
                        control_success = np.sum(control_data > control_median)
                        treatment_success = np.sum(treatment_data > control_median)
                        
                        results = st.session_state.ab_engine.chi_squared_test(
                            control_success, len(control_data),
                            treatment_success, len(treatment_data),
                            alpha=alpha
                        )
                    else:  # Bayesian
                        # Convert to binary for Bayesian
                        st.info("ℹ️ Converting continuous metric to binary (above median = success)")
                        control_median = np.median(control_data)
                        control_success = np.sum(control_data > control_median)
                        treatment_success = np.sum(treatment_data > control_median)
                        
                        results = st.session_state.ab_engine.bayesian_ab_test(
                            control_success, len(control_data),
                            treatment_success, len(treatment_data)
                        )
                    
                    # Display results
                    st.subheader("📊 Test Results")
                    
                    # Metrics
                    if test_type != 'Bayesian':
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Control Mean",
                                f"{results.get('control_mean', 0):.4f}"
                            )
                        
                        with col2:
                            st.metric(
                                "Treatment Mean",
                                f"{results.get('treatment_mean', 0):.4f}"
                            )
                        
                        with col3:
                            st.metric(
                                "Relative Lift",
                                f"{results.get('relative_lift', 0):+.2f}%"
                            )
                        
                        with col4:
                            sig_label = "✅ Significant" if results['significant'] else "❌ Not Significant"
                            st.metric("Result", sig_label)
                        
                        # Interpretation
                        st.subheader("💡 Business Interpretation")
                        interpretation = st.session_state.interpreter.interpret_ab_test_results(results)
                        st.markdown(interpretation)
                        
                        # Visualizations
                        st.subheader("📈 Visualizations")
                        
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            fig_dist = st.session_state.visualizer.plot_distribution_comparison(
                                control_data, treatment_data, metric_col
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)
                        
                        with viz_col2:
                            fig_box = st.session_state.visualizer.plot_box_comparison(
                                df, group_col, metric_col
                            )
                            st.plotly_chart(fig_box, use_container_width=True)
                        
                        fig_ci = st.session_state.visualizer.plot_confidence_interval(
                            results, metric_col
                        )
                        st.plotly_chart(fig_ci, use_container_width=True)
                    
                    else:  # Bayesian results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Probability Treatment Better",
                                f"{results['prob_treatment_better']*100:.1f}%"
                            )
                        
                        with col2:
                            st.metric(
                                "Expected Lift",
                                f"{results['expected_lift']:+.2f}%"
                            )
                        
                        with col3:
                            recommendation = "✅ Implement" if results['prob_treatment_better'] > 0.95 else "⚠️ Uncertain"
                            st.metric("Recommendation", recommendation)
                        
                        # Bayesian interpretation
                        st.subheader("💡 Bayesian Interpretation")
                        bayesian_interp = st.session_state.interpreter.interpret_bayesian_results(results)
                        st.markdown(bayesian_interp)
                        
                        # Posterior distributions
                        st.subheader("📈 Posterior Distributions")
                        fig_posterior = st.session_state.visualizer.plot_bayesian_posteriors(
                            results['control_posterior_alpha'],
                            results['control_posterior_beta'],
                            results['treatment_posterior_alpha'],
                            results['treatment_posterior_beta']
                        )
                        st.plotly_chart(fig_posterior, use_container_width=True)
                    
                    # Store results in session state for export
                    st.session_state.ab_test_results = results
                    st.session_state.ab_test_interpretation = interpretation if test_type != 'Bayesian' else bayesian_interp
        
        # Power Analysis Section
        st.divider()
        st.subheader("📏 Sample Size Calculator (Power Analysis)")
        
        with st.expander("Calculate Required Sample Size"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                baseline_mean = st.number_input(
                    "Baseline Mean",
                    value=100.0,
                    help="Expected mean of control group"
                )
            
            with col2:
                baseline_std = st.number_input(
                    "Baseline Std Dev",
                    value=20.0,
                    help="Expected standard deviation"
                )
            
            with col3:
                mde = st.number_input(
                    "Minimum Detectable Effect (%)",
                    value=5.0,
                    min_value=0.1,
                    help="Smallest effect you want to detect"
                )
            
            power_col1, power_col2 = st.columns(2)
            
            with power_col1:
                power = st.slider("Statistical Power", 0.70, 0.95, 0.80, 0.05)
            
            with power_col2:
                power_alpha = st.slider("Significance Level", 0.01, 0.10, 0.05, 0.01)
            
            if st.button("Calculate Sample Size"):
                power_results = st.session_state.ab_engine.calculate_sample_size(
                    baseline_mean, mde, baseline_std, power_alpha, power
                )
                
                if power_results:
                    st.success(f"✅ Total sample size needed: **{power_results['total_sample_size']:,}**")
                    
                    power_interp = st.session_state.interpreter.interpret_power_analysis(power_results)
                    st.markdown(power_interp)
    
    # ==================== TAB 3: CAUSAL INFERENCE ====================
    with tab3:
        st.header("🎯 Causal Inference Lab")
        
        method = st.selectbox(
            "Select Causal Method",
            options=[
                'Propensity Score Matching (PSM)',
                'Difference-in-Differences (DiD)',
                'Instrumental Variables (IV)'
            ]
        )
        
        if method == 'Propensity Score Matching (PSM)':
            st.subheader("🎯 Propensity Score Matching")
            st.info("PSM creates comparable groups from observational data by matching units with similar characteristics.")
            
            with st.expander("⚙️ PSM Configuration", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    treatment_col = st.selectbox(
                        "Treatment Column",
                        options=st.session_state.data_handler.categorical_columns,
                        help="Binary treatment indicator"
                    )
                
                with col2:
                    outcome_col = st.selectbox(
                        "Outcome Column",
                        options=st.session_state.data_handler.numeric_columns,
                        help="Outcome variable to measure"
                    )
                
                with col3:
                    caliper = st.slider(
                        "Caliper (Max PS Distance)",
                        0.01, 0.50, 0.10, 0.01,
                        help="Maximum allowed propensity score difference for matching"
                    )
                
                covariate_cols = st.multiselect(
                    "Covariate Columns (for matching)",
                    options=st.session_state.data_handler.numeric_columns,
                    help="Variables to control for in matching"
                )
            
            if st.button("🚀 Run PSM Analysis", type="primary"):
                if not covariate_cols:
                    st.error("❌ Please select at least one covariate for matching")
                else:
                    with st.spinner("Running Propensity Score Matching..."):
                        psm_results = st.session_state.causal_lab.propensity_score_matching(
                            st.session_state.data_handler.data,
                            treatment_col,
                            outcome_col,
                            covariate_cols,
                            caliper=caliper
                        )
                        
                        if 'error' in psm_results:
                            st.error(f"❌ {psm_results['error']}")
                        else:
                            # Display results
                            st.subheader("📊 PSM Results")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("ATT (Treatment Effect)", f"{psm_results['att']:+.4f}")
                            
                            with col2:
                                st.metric("P-value", f"{psm_results['p_value']:.4f}")
                            
                            with col3:
                                st.metric("Match Rate", f"{psm_results['match_rate']:.1f}%")
                            
                            with col4:
                                sig = "✅ Significant" if psm_results['p_value'] < 0.05 else "❌ Not Significant"
                                st.metric("Significance", sig)
                            
                            # Interpretation
                            st.subheader("💡 Causal Interpretation")
                            causal_interp = st.session_state.interpreter.interpret_causal_effect(
                                psm_results, "PSM"
                            )
                            st.markdown(causal_interp)
                            
                            # Visualizations
                            st.subheader("📈 Diagnostic Plots")
                            
                            viz_col1, viz_col2 = st.columns(2)
                            
                            with viz_col1:
                                fig_ps = st.session_state.visualizer.plot_propensity_scores(
                                    psm_results['matched_treated'],
                                    psm_results['matched_control']
                                )
                                st.plotly_chart(fig_ps, use_container_width=True)
                            
                            with viz_col2:
                                fig_balance = st.session_state.visualizer.plot_balance_diagnostics(
                                    psm_results['balance_stats']
                                )
                                st.plotly_chart(fig_balance, use_container_width=True)
                            
                            # Balance table
                            st.subheader("⚖️ Covariate Balance")
                            st.dataframe(
                                psm_results['balance_stats'],
                                use_container_width=True
                            )
                            
                            st.session_state.psm_results = psm_results
        
        elif method == 'Difference-in-Differences (DiD)':
            st.subheader("📊 Difference-in-Differences Analysis")
            st.info("DiD estimates causal effects by comparing changes over time between treated and control groups.")
            
            with st.expander("⚙️ DiD Configuration", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    group_col_did = st.selectbox(
                        "Group Column",
                        options=st.session_state.data_handler.categorical_columns,
                        help="Column identifying treated vs control groups"
                    )
                    
                    time_col = st.selectbox(
                        "Time Period Column",
                        options=st.session_state.data_handler.categorical_columns,
                        key='time_col',
                        help="Column identifying pre/post periods"
                    )
                
                with col2:
                    outcome_col_did = st.selectbox(
                        "Outcome Column",
                        options=st.session_state.data_handler.numeric_columns,
                        key='outcome_did',
                        help="Outcome variable"
                    )
                    
                    # Get unique values for selection
                    if group_col_did and time_col:
                        treatment_group = st.selectbox(
                            "Treatment Group Value",
                            options=st.session_state.data_handler.data[group_col_did].unique()
                        )
                        
                        post_period = st.selectbox(
                            "Post-Treatment Period Value",
                            options=st.session_state.data_handler.data[time_col].unique()
                        )
            
            if st.button("🚀 Run DiD Analysis", type="primary"):
                with st.spinner("Running Difference-in-Differences..."):
                    did_results = st.session_state.causal_lab.difference_in_differences(
                        st.session_state.data_handler.data,
                        group_col_did,
                        time_col,
                        outcome_col_did,
                        treatment_group,
                        post_period
                    )
                    
                    # Display results
                    st.subheader("📊 DiD Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("DiD Estimate", f"{did_results['did_estimate']:+.4f}")
                    
                    with col2:
                        st.metric("P-value", f"{did_results['p_value']:.4f}")
                    
                    with col3:
                        st.metric("R-squared", f"{did_results['r_squared']:.4f}")
                    
                    # Interpretation
                    st.subheader("💡 Causal Interpretation")
                    did_interp = st.session_state.interpreter.interpret_causal_effect(
                        did_results, "DiD"
                    )
                    st.markdown(did_interp)
                    
                    # Parallel trends plot
                    st.subheader("📈 Parallel Trends")
                    fig_trends = st.session_state.visualizer.plot_did_trends(
                        did_results['group_time_means']
                    )
                    st.plotly_chart(fig_trends, use_container_width=True)
                    
                    # Model summary
                    with st.expander("📋 Full Regression Output"):
                        st.text(str(did_results['model_summary']))
                    
                    st.session_state.did_results = did_results
        
        else:  # Instrumental Variables
            st.subheader("🎻 Instrumental Variables (2SLS)")
            st.info("IV estimation addresses endogeneity by using an instrument that affects treatment but not the outcome directly.")
            
            with st.expander("⚙️ IV Configuration", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    outcome_col_iv = st.selectbox(
                        "Outcome Variable",
                        options=st.session_state.data_handler.numeric_columns,
                        key='outcome_iv'
                    )
                    
                    treatment_col_iv = st.selectbox(
                        "Treatment Variable (Endogenous)",
                        options=st.session_state.data_handler.numeric_columns,
                        key='treatment_iv'
                    )
                
                with col2:
                    instrument_col = st.selectbox(
                        "Instrumental Variable",
                        options=st.session_state.data_handler.numeric_columns,
                        key='instrument_iv',
                        help="Variable that affects treatment but not outcome directly"
                    )
                    
                    covariate_cols_iv = st.multiselect(
                        "Control Variables (Optional)",
                        options=st.session_state.data_handler.numeric_columns,
                        key='covariates_iv'
                    )
            
            if st.button("🚀 Run IV Analysis", type="primary"):
                with st.spinner("Running Instrumental Variables estimation..."):
                    iv_results = st.session_state.causal_lab.instrumental_variables(
                        st.session_state.data_handler.data,
                        outcome_col_iv,
                        treatment_col_iv,
                        instrument_col,
                        covariate_cols_iv if covariate_cols_iv else None
                    )
                    
                    if iv_results:
                        st.subheader("📊 IV Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("IV Estimate", f"{iv_results['iv_estimate']:+.4f}")
                        
                        with col2:
                            st.metric("P-value", f"{iv_results['p_value']:.4f}")
                        
                        with col3:
                            st.metric("First Stage F-stat", f"{iv_results['first_stage_f_stat']:.2f}")
                        
                        with col4:
                            weak = "⚠️ Weak" if iv_results['weak_instrument'] else "✅ Strong"
                            st.metric("Instrument Strength", weak)
                        
                        if iv_results['weak_instrument']:
                            st.warning(
                                "⚠️ **Weak Instrument Warning:** First-stage F-statistic < 10 suggests "
                                "the instrument may be weak. Consider finding a stronger instrument."
                            )
                        
                        st.session_state.iv_results = iv_results
    
    # ==================== TAB 4: REPORT EXPORT ====================
    with tab4:
        st.header("📄 Report Export")
        
        st.write("Export your analysis results in various formats for sharing and archiving.")
        
        # Check if results exist
        has_ab_results = 'ab_test_results' in st.session_state
        has_psm_results = 'psm_results' in st.session_state
        has_did_results = 'did_results' in st.session_state
        
        if not (has_ab_results or has_psm_results or has_did_results):
            st.info("ℹ️ No analysis results available. Please run an analysis first.")
        else:
            # Select which results to export
            export_options = []
            if has_ab_results:
                export_options.append("A/B Test Results")
            if has_psm_results:
                export_options.append("PSM Results")
            if has_did_results:
                export_options.append("DiD Results")
            
            selected_export = st.selectbox(
                "Select Results to Export",
                options=export_options
            )
            
            export_format = st.radio(
                "Export Format",
                options=['Excel', 'Markdown', 'HTML'],
                horizontal=True
            )
            
            if st.button("📥 Generate Export", type="primary"):
                with st.spinner("Generating export..."):
                    # Get appropriate results
                    if selected_export == "A/B Test Results":
                        results = st.session_state.ab_test_results
                        interpretation = st.session_state.ab_test_interpretation
                        test_type = results.get('test_type', 'A/B Test')
                    elif selected_export == "PSM Results":
                        results = st.session_state.psm_results
                        interpretation = st.session_state.interpreter.interpret_causal_effect(results, "PSM")
                        test_type = "Propensity Score Matching"
                    else:
                        results = st.session_state.did_results
                        interpretation = st.session_state.interpreter.interpret_causal_effect(results, "DiD")
                        test_type = "Difference-in-Differences"
                    
                    # Generate export
                    if export_format == 'Excel':
                        excel_file = st.session_state.report_gen.create_excel_report(
                            results,
                            st.session_state.data_handler.data,
                            test_type
                        )
                        
                        st.download_button(
                            label="⬇️ Download Excel Report",
                            data=excel_file,
                            file_name=f"experiment_report_{st.session_state.report_gen.timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    elif export_format == 'Markdown':
                        md_report = st.session_state.report_gen.create_markdown_report(
                            results,
                            interpretation,
                            test_type
                        )
                        
                        st.download_button(
                            label="⬇️ Download Markdown Report",
                            data=md_report,
                            file_name=f"experiment_report_{st.session_state.report_gen.timestamp}.md",
                            mime="text/markdown"
                        )
                        
                        with st.expander("Preview Markdown"):
                            st.markdown(md_report)
                    
                    else:  # HTML
                        html_report = st.session_state.report_gen.create_html_report(
                            results,
                            interpretation,
                            test_type
                        )
                        
                        st.download_button(
                            label="⬇️ Download HTML Report",
                            data=html_report,
                            file_name=f"experiment_report_{st.session_state.report_gen.timestamp}.html",
                            mime="text/html"
                        )
                        
                        with st.expander("Preview HTML"):
                            st.components.v1.html(html_report, height=600, scrolling=True)

else:
    # Welcome screen when no data uploaded
    st.info("👈 Please upload a dataset using the sidebar to begin your analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🧪 A/B Testing
        - T-tests & Z-tests
        - Chi-squared tests
        - Bayesian A/B testing
        - Power analysis
        - Sample size calculator
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Causal Inference
        - Propensity Score Matching
        - Difference-in-Differences
        - Instrumental Variables
        - Balance diagnostics
        - Parallel trends analysis
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Features
        - Automated health checks
        - SRM detection
        - Interactive visualizations
        - Plain English explanations
        - Multi-format exports
        """)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p>🔬 Experimentation & Causal Analysis Suite | Built with Streamlit & Python</p>
        <p style='font-size: 0.9em;'>Statistical rigor meets business clarity</p>
    </div>
""", unsafe_allow_html=True)
