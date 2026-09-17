"""
Experimentation & Causal Analysis Suite
Main Streamlit Application
"""

from pathlib import Path

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
from modules.ab_advanced import is_binary_metric, format_change
from modules.experiment_design import (
    ROW_IS_UNIT, guess_unit_column, guess_assignment_column, guess_control_label,
    metric_candidates, describe_metric, to_unit_level, experiment_summary
)
from utils import StatisticalInterpreter, ReportGenerator, ship_decision

BINARY_ONLY_TESTS = ('Proportions Z-Test', 'Chi-Squared', 'Bayesian')
AUTO_TEST = 'Auto (recommended for this metric)'
TEST_TYPES = [
    'T-Test', 'Z-Test', 'Proportions Z-Test', 'Chi-Squared', 'Bayesian',
    'Mann-Whitney U', 'Bootstrap', 'CUPED (variance reduction)', 'Sequential (always-valid)'
]


def run_selected_test(engine, test_type, control, treatment, alpha,
                      equal_var=False, control_cov=None, treatment_cov=None):
    """Dispatch one control-vs-treatment comparison to the engine"""
    if test_type in BINARY_ONLY_TESTS:
        if not (is_binary_metric(control) and is_binary_metric(treatment)):
            raise ValueError(
                f"{test_type} needs a 0/1 metric such as a conversion flag. "
                "For continuous metrics use T-Test, Mann-Whitney U, or Bootstrap."
            )
        counts = (int(control.sum()), len(control), int(treatment.sum()), len(treatment))
        if test_type == 'Proportions Z-Test':
            return engine.proportions_test(*counts, alpha=alpha)
        if test_type == 'Chi-Squared':
            return engine.chi_squared_test(*counts, alpha=alpha)
        return engine.bayesian_ab_test(*counts)
    if test_type == 'Z-Test':
        return engine.z_test(control, treatment, alpha=alpha)
    if test_type == 'Mann-Whitney U':
        return engine.mann_whitney_test(control, treatment, alpha=alpha)
    if test_type == 'Bootstrap':
        return engine.bootstrap_test(control, treatment, alpha=alpha)
    if test_type == 'CUPED (variance reduction)':
        return engine.cuped_test(control, treatment, control_cov, treatment_cov, alpha=alpha)
    if test_type == 'Sequential (always-valid)':
        return engine.sequential_test(control, treatment, alpha=alpha)
    return engine.t_test(control, treatment, alpha=alpha, equal_var=equal_var)

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
    
    SAMPLE_DATASETS = {
        "A/B test (revenue, conversion)": "sample_ab_test_data.csv",
        "E-commerce A/B test (funnel)": "ecommerce_ab_test.csv",
        "Difference-in-differences (store sales)": "sample_did_data.csv",
        "Multi-variant checkout test (CUPED, guardrails)": "multivariant_checkout_test.csv",
    }
    sample_choice = st.selectbox(
        "Or try a sample dataset",
        options=["(none)"] + list(SAMPLE_DATASETS),
        help="Bundled datasets so you can explore without uploading anything"
    )
    sample_path = (
        Path(__file__).parent / "data" / SAMPLE_DATASETS[sample_choice]
        if sample_choice != "(none)" else None
    )
    has_data = bool(uploaded_file or sample_path)
    
    if has_data:
        try:
            if uploaded_file:
                data = st.session_state.data_handler.load_data(uploaded_file)
            else:
                data = st.session_state.data_handler.load_path(sample_path)
            st.success(f"✅ Loaded {len(data):,} rows × {len(data.columns)} columns")
            
            st.subheader("📊 Data Info")
            st.write(f"**Numeric columns:** {len(st.session_state.data_handler.numeric_columns)}")
            st.write(f"**Categorical columns:** {len(st.session_state.data_handler.categorical_columns)}")
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.session_state.data_handler.data = None
            has_data = False

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

# Each tab renders inside a function so that a tab can bail out early with `return`.
# `st.stop()` would halt the whole script and blank every tab after it.
def render_power_calculator():
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
            try:
                power_results = st.session_state.ab_engine.calculate_sample_size(
                    baseline_mean, mde, baseline_std, power_alpha, power
                )
            except Exception as e:
                st.error(f"Error calculating sample size: {str(e)}")
                power_results = {}
            
            if power_results:
                st.success(f"✅ Total sample size needed: **{power_results['total_sample_size']:,}**")
                
                power_interp = st.session_state.interpreter.interpret_power_analysis(power_results)
                st.markdown(power_interp)


def render_ab_tab():
    st.header("🧪 A/B Testing Engine")
    
    data = st.session_state.data_handler.data
    all_columns = data.columns.tolist()
    
    # ---------- 1. Who was randomized? ----------
    with st.container(border=True):
        st.subheader("1️⃣ Who was randomized?")
        st.caption(
            "The unit of randomization is whatever was assigned to an arm: usually a user, sometimes "
            "a device, session, or store. Every unit must count once, or the test is overconfident."
        )
        
        unit_guess = guess_unit_column(data)
        unit_options = [ROW_IS_UNIT] + all_columns
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            unit_choice = st.selectbox(
                "Unit of Randomization",
                options=unit_options,
                index=unit_options.index(unit_guess) if unit_guess else 0,
                help="The ID column of the thing that was randomly assigned (user_id, store_id). "
                     "If the table has several rows per unit, they are combined before testing."
            )
        unit_col = None if unit_choice == ROW_IS_UNIT else unit_choice
        
        assignment_options = [
            c for c in all_columns
            if c != unit_col and 2 <= data[c].nunique(dropna=True) <= 10
        ]
        if not assignment_options:
            st.error(
                "❌ No column looks like an arm assignment (2 to 10 distinct values). "
                "A/B analysis needs one, e.g. a 'variant' column with 'control' and 'treatment'. "
                "The other tabs still work with this dataset."
            )
            render_power_calculator()
            return
        assignment_guess = guess_assignment_column(data, exclude=unit_col)
        with r_col2:
            group_col = st.selectbox(
                "Assignment Column",
                options=assignment_options,
                index=assignment_options.index(assignment_guess) if assignment_guess in assignment_options else 0,
                help="The column recording which arm each unit was assigned to"
            )
        
        group_values = data[group_col].dropna().unique().tolist() if group_col else []
        
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            control_group = st.selectbox(
                "Control Arm (baseline experience)",
                options=group_values,
                index=guess_control_label(group_values),
                help="Lift is always reported as the other arm relative to this one"
            )
        with a_col2:
            treatment_group = st.selectbox(
                "Treatment Arm (the change being tested)",
                options=[g for g in group_values if g != control_group],
                help="The arm whose effect you want to measure"
            )
        
        compare_all = False
        correction = 'holm'
        if len(group_values) > 2:
            mv_col1, mv_col2 = st.columns(2)
            with mv_col1:
                compare_all = st.checkbox(
                    f"Compare all {len(group_values) - 1} variants against control",
                    value=True,
                    help="Tests every variant and corrects p-values for the number of comparisons"
                )
            with mv_col2:
                correction = st.selectbox(
                    "Multiple Comparison Correction",
                    options=['holm', 'bonferroni', 'fdr_bh', 'none'],
                    help="Holm and Bonferroni control the chance of any false win; "
                         "fdr_bh controls the share of false wins"
                )
        
        arms_in_test = group_values if compare_all else [control_group, treatment_group]
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            control_share = st.number_input(
                "Planned Traffic to Control (%)",
                min_value=1.0, max_value=99.0,
                value=round(100.0 / max(len(arms_in_test), 2), 1), step=1.0,
                help="What the experiment was configured to send to control. The remaining traffic is "
                     "assumed to be split evenly across the other arms. Used to detect a sample ratio mismatch."
            )
        
        # Rows per unit decide whether aggregation is needed
        agg = 'mean'
        if unit_col:
            rows_per_unit = data.groupby(unit_col).size()
            if (rows_per_unit > 1).any():
                with s_col2:
                    agg_label = st.selectbox(
                        "Combine a Unit's Rows By",
                        options=['Average', 'Total', 'Any (max)'],
                        help="Average: typical value per row. Total: sum over the experiment "
                             "(revenue per user). Any: did it ever happen (converted at least once)."
                    )
                agg = {'Average': 'mean', 'Total': 'sum', 'Any (max)': 'max'}[agg_label]
                st.warning(
                    f"⚠️ **{unit_col}** repeats: {len(rows_per_unit):,} units across {len(data):,} rows "
                    f"(up to {int(rows_per_unit.max())} rows each). Rows from the same unit are not independent, "
                    "so they are combined into one value per unit before testing."
                )
            else:
                st.success(f"✅ One row per **{unit_col}**: {len(rows_per_unit):,} independent units.")
        else:
            st.info(
                f"ℹ️ Treating each of the {len(data):,} rows as an independent unit. If a user can appear "
                "on several rows, pick their ID column above instead."
            )
    
    # ---------- 2. What are you trying to move? ----------
    with st.container(border=True):
        st.subheader("2️⃣ What are you trying to move?")
        st.caption(
            "Pick ONE primary metric before looking at results. It is the only metric the ship decision "
            "is based on. Everything else is a guardrail or context."
        )
        
        metric_options = metric_candidates(data, unit_col, group_col)
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            metric_col = st.selectbox(
                "Primary Metric",
                options=metric_options,
                help="The single outcome this experiment was designed to change"
            )
        with m_col2:
            primary_direction = st.radio(
                "A Win Is When It Goes",
                options=['▲ Up', '▼ Down'],
                horizontal=True,
                help="Up for conversion or revenue. Down for churn, latency, or error rate."
            )
        
        metric_info = describe_metric(data[metric_col]) if metric_col else None
        if metric_info:
            st.markdown(f"📐 **{metric_col}** is {metric_info['summary']}.")
        
        mde_pct = st.number_input(
            "Smallest Lift Worth Shipping (% relative to control)",
            value=2.0, min_value=0.0, step=0.5,
            help="Relative, not percentage points: 2 means control 5.0% → 5.1%. A result whose whole "
                 "confidence interval sits below this is called 'no meaningful effect' instead of 'keep running'."
        )
    
    # ---------- 3. What must not break? ----------
    with st.container(border=True):
        st.subheader("3️⃣ What must not break? (optional)")
        st.caption(
            "Guardrails are metrics you are not trying to improve but refuse to damage. "
            "A significant move in the bad direction blocks shipping even if the primary metric wins."
        )
        guardrail_cols = st.multiselect(
            "Guardrail Metrics",
            options=[c for c in metric_options if c != metric_col],
            help="e.g. page load time, refunds, support tickets, unsubscribe rate"
        )
        guardrail_up_is_good = {}
        for g_col in guardrail_cols:
            bad_direction = st.radio(
                f"{g_col} is harmed when it goes",
                options=['▼ Down', '▲ Up'],
                horizontal=True,
                key=f"guardrail_direction_{g_col}",
                help="Down for things you want to keep high (retention). Up for things you want to keep low (latency)."
            )
            guardrail_up_is_good[g_col] = bad_direction == '▼ Down'
    
    lower_is_better = (
        ([metric_col] if primary_direction == '▼ Down' else [])
        + [g for g, up_is_good in guardrail_up_is_good.items() if not up_is_good]
    )
    
    # ---------- 4. How should it be analyzed? ----------
    with st.container(border=True):
        st.subheader("4️⃣ How should it be analyzed?")
        
        test_choice = st.selectbox(
            "Statistical Test",
            options=[AUTO_TEST] + TEST_TYPES,
            help="Auto picks the test that matches the metric's type. Override only if you have a reason."
        )
        test_type = metric_info['recommended_test'] if test_choice == AUTO_TEST and metric_info else test_choice
        if test_choice == AUTO_TEST and metric_info:
            st.markdown(f"🧭 Using **{test_type}**. {metric_info['reason']}")
        elif metric_info and test_type in BINARY_ONLY_TESTS and metric_info['kind'] != 'binary':
            st.error(f"❌ {test_type} needs a yes/no (0/1) metric. **{metric_col}** is not one.")
        elif metric_info and metric_info['kind'] == 'binary' and test_type in ('T-Test', 'Z-Test'):
            st.warning(
                f"⚠️ **{metric_col}** is a yes/no metric. {test_type} will run, but a Proportions Z-Test "
                "is the standard choice for rates."
            )
        
        order_col = None
        if test_type == 'Sequential (always-valid)':
            ROWS_IN_ORDER = "(rows are already in arrival order)"
            time_like = [c for c in all_columns if any(k in c.lower() for k in ('time', 'date', 'ts', 'created', 'exposed'))]
            order_options = [ROWS_IN_ORDER] + [c for c in all_columns if c not in (group_col, metric_col)]
            order_choice = st.selectbox(
                "Arrival Order",
                options=order_options,
                index=order_options.index(time_like[0]) if time_like and time_like[0] in order_options else 0,
                help="A sequential test replays the experiment as data arrived, so it needs to know the order. "
                     "Pick the exposure timestamp, or confirm the rows are already sorted by time."
            )
            order_col = None if order_choice == ROWS_IN_ORDER else order_choice
            if order_col is None:
                st.caption("⚠️ Interim looks are only meaningful if the file really is sorted by arrival time.")

        cuped_covariate = None
        if test_type == 'CUPED (variance reduction)':
            cuped_covariate = st.selectbox(
                "Pre-Experiment Covariate",
                options=[c for c in metric_options if c != metric_col],
                help="Must be measured BEFORE assignment (e.g. last month's spend), so the treatment "
                     "cannot have affected it. The more it correlates with the metric, the more noise it removes."
            )
        
        with st.expander("Advanced settings"):
            alpha = st.slider(
                "Significance Level (α)",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                help="How often you accept calling a win when nothing changed. 0.05 is the convention."
            )
            assume_equal_var = False
            if test_type == 'T-Test':
                assume_equal_var = st.checkbox(
                    "Assume equal variances (Student's t-test)",
                    value=False,
                    help="Leave off. Welch's t-test stays valid when variances or arm sizes differ; "
                         "Student's does not."
                )
    
    # ---------- Experiment as it will be analyzed ----------
    unit_label = f"{unit_col} units" if unit_col else "rows"
    if metric_col and treatment_group is not None:
        arm_counts = (
            data[data[group_col].isin(arms_in_test)]
            .groupby(group_col)[unit_col if unit_col else group_col]
            .agg('nunique' if unit_col else 'size')
        )
        arm_counts = {str(a): int(arm_counts.get(a, 0)) for a in arms_in_test}
        other_share = (100.0 - control_share) / max(len(arms_in_test) - 1, 1)
        expected_split = {
            str(a): (control_share if a == control_group else other_share) / 100.0 for a in arms_in_test
        }
        st.info(
            "**Your experiment, as it will be analyzed**\n\n" + experiment_summary(
                unit_label=unit_label,
                n_units=sum(arm_counts.values()),
                arms=arm_counts,
                control=str(control_group),
                metric=metric_col,
                metric_kind=metric_info['kind'],
                higher_is_better=primary_direction == '▲ Up',
                mde_pct=mde_pct if mde_pct > 0 else None,
                test_name=test_type if not compare_all else f"{test_type}, {correction}-corrected across variants",
                alpha=alpha,
                guardrails=guardrail_up_is_good,
                expected_split=expected_split
            )
        )
    
    # Validate and run test
    if st.button("🚀 Run A/B Test", type="primary", use_container_width=True):
        
        # Validation
        is_valid, msg = st.session_state.data_handler.validate_ab_test_columns(
            group_col, metric_col
        )
        
        if not is_valid:
            st.error(f"❌ Validation Error: {msg}")
        elif treatment_group is None:
            st.error("❌ Validation Error: pick a treatment group different from control")
        else:
            with st.spinner("Running analysis..."):
                # Prepare data
                extra_cols = [c for c in [cuped_covariate] + guardrail_cols if c]
                unit_df, unit_info = to_unit_level(
                    data, unit_col, group_col, [metric_col] + extra_cols, agg=agg, order_col=order_col
                )
                if unit_info['n_contaminated_units']:
                    st.warning(
                        f"⚠️ {unit_info['n_contaminated_units']:,} units appear in more than one arm "
                        "and were excluded. They saw both experiences, so they cannot be attributed to either."
                    )
                if unit_info['aggregated']:
                    st.info(
                        f"ℹ️ Combined {unit_info['n_rows']:,} rows into {unit_info['n_units']:,} "
                        f"units (one per {unit_col}, by {agg})."
                    )
                full = unit_df.dropna(subset=[group_col, metric_col])
                n_groups = full[group_col].nunique()
                multi = compare_all and n_groups > 2
                
                df = full if multi else full[full[group_col].isin([control_group, treatment_group])]
                if n_groups > 2 and not multi:
                    st.info(
                        f"ℹ️ {n_groups} groups found. Comparing **{treatment_group}** "
                        f"against **{control_group}** only."
                    )
                
                # Run health checks
                st.subheader("🏥 Health Checks")
                health_results = st.session_state.health_checker.run_all_checks(
                    df, group_col, metric_col,
                    expected_ratio={a: expected_split[str(a)] for a in arms_in_test}
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
                
                # Multi-variant: every variant against control, corrected p-values
                if multi:
                    groups = {g: d[metric_col].values for g, d in df.groupby(group_col)}
                    mv_results = st.session_state.ab_engine.multi_variant_test(
                        groups, control_group, alpha=alpha, correction=correction
                    )
                    st.subheader("📊 All Variants vs Control")
                    st.dataframe(pd.DataFrame([{
                        'Variant': c['group'],
                        'Control Mean': c['control_mean'],
                        'Variant Mean': c['treatment_mean'],
                        'Change': format_change(c),
                        'p (raw)': c['p_value_raw'],
                        f'p ({correction})': c['p_value_adjusted'],
                        'Significant': '✅' if c['significant'] else '❌'
                    } for c in mv_results['comparisons']]), use_container_width=True)
                    st.markdown(
                        st.session_state.interpreter.interpret_multi_variant(mv_results)
                    )
                    st.session_state.multi_variant_results = mv_results
                    
                    # Detailed view continues with the best variant, or the selected one
                    if mv_results['best_variant'] is not None:
                        treatment_group = next(
                            g for g in groups if str(g) == mv_results['best_variant']
                        )
                    st.info(f"ℹ️ Detailed results below: **{treatment_group}** vs **{control_group}**")
                
                control_rows = df[df[group_col] == control_group]
                treatment_rows = df[df[group_col] == treatment_group]
                control_data = control_rows[metric_col].values
                treatment_data = treatment_rows[metric_col].values
                
                # Run the selected test
                try:
                    if cuped_covariate:
                        control_rows = control_rows.dropna(subset=[cuped_covariate])
                        treatment_rows = treatment_rows.dropna(subset=[cuped_covariate])
                        control_data = control_rows[metric_col].values
                        treatment_data = treatment_rows[metric_col].values
                    results = run_selected_test(
                        st.session_state.ab_engine, test_type, control_data, treatment_data, alpha,
                        equal_var=assume_equal_var,
                        control_cov=control_rows[cuped_covariate].values if cuped_covariate else None,
                        treatment_cov=treatment_rows[cuped_covariate].values if cuped_covariate else None
                    )
                except ValueError as e:
                    st.error(f"❌ {str(e)}")
                    return
                
                # Guardrails: proportions test for 0/1 metrics, Welch otherwise
                guardrail_results = {}
                for g_col in guardrail_cols:
                    g_control = control_rows[g_col].dropna().values
                    g_treatment = treatment_rows[g_col].dropna().values
                    g_test = (
                        'Proportions Z-Test'
                        if is_binary_metric(g_control) and is_binary_metric(g_treatment) else 'T-Test'
                    )
                    guardrail_results[g_col] = run_selected_test(
                        st.session_state.ab_engine, g_test, g_control, g_treatment, alpha
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
                            format_change(results),
                            help=(
                                f"95% CI: [{results['lift_ci_lower']:+.2f}%, {results['lift_ci_upper']:+.2f}%]"
                                if 'lift_ci_lower' in results else results.get('relative_lift_note')
                            )
                        )
                    
                    with col4:
                        sig_label = "✅ Significant" if results['significant'] else "❌ Not Significant"
                        st.metric("Result", sig_label)
                    
                    # Ship decision: primary metric + guardrails + health
                    decision = ship_decision(
                        results,
                        guardrails=guardrail_results,
                        health=health_results,
                        higher_is_better=metric_col not in lower_is_better,
                        guardrail_higher_is_better={g: g not in lower_is_better for g in guardrail_cols},
                        mde_pct=mde_pct if mde_pct > 0 else None
                    )
                    st.subheader("🚦 Decision")
                    decision_box = {
                        'SHIP': st.success, 'DO NOT SHIP': st.error
                    }.get(decision['decision'], st.warning)
                    decision_box(f"**{decision['decision']}**")
                    for reason in decision['reasons']:
                        st.markdown(f"- {reason}")
                    
                    if guardrail_results:
                        st.dataframe(pd.DataFrame([{
                            'Guardrail': g,
                            'Control': r['control_mean'],
                            'Treatment': r['treatment_mean'],
                            'Change': format_change(r),
                            'p-value': r['p_value'],
                            'Status': '🛑 Harmed' if g in decision['harmed_guardrails'] else '✅ Held'
                        } for g, r in guardrail_results.items()]), use_container_width=True)
                    st.session_state.ab_decision = decision
                    
                    # Interpretation
                    st.subheader("💡 Business Interpretation")
                    interpretation = st.session_state.interpreter.interpret_ab_test_results(
                        results, higher_is_better=metric_col not in lower_is_better
                    )
                    st.markdown(interpretation)
                    
                    if test_type == 'Sequential (always-valid)':
                        st.plotly_chart(
                            st.session_state.visualizer.plot_sequential_path(results),
                            use_container_width=True
                        )
                    
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
                    
                    if 'ci_lower' in results:
                        fig_ci = st.session_state.visualizer.plot_confidence_interval(
                            results, metric_col
                        )
                        st.plotly_chart(fig_ci, use_container_width=True)
                
                else:  # Bayesian results
                    # No frequentist decision for this run; do not export a stale one
                    st.session_state.pop('ab_decision', None)
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
                        if results['prob_treatment_better'] > 0.95:
                            recommendation = "✅ Implement"
                        elif results['prob_treatment_better'] < 0.05:
                            recommendation = "❌ Keep Control"
                        else:
                            recommendation = "⚠️ Uncertain"
                        st.metric("Recommendation", recommendation)
                    
                    st.caption(
                        f"Expected loss if you ship treatment: {results['expected_loss_treatment']*100:.3f} pp · "
                        f"if you keep control: {results['expected_loss_control']*100:.3f} pp"
                    )
                    
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
    
    render_power_calculator()


def render_causal_tab():
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
                    options=st.session_state.data_handler.binary_columns(),
                    help="Column with exactly two values (treated vs not treated)"
                )
                treated_value = st.selectbox(
                    "Treated Value",
                    options=(
                        sorted(
                            st.session_state.data_handler.data[treatment_col].dropna().unique().tolist(),
                            key=lambda v: str(v).lower() not in ('1', 'true', 'treatment', 'treated')
                        ) if treatment_col else []
                    ),
                    help="Which value marks the treated group"
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
                options=[
                    c for c in st.session_state.data_handler.numeric_columns
                    if c not in (treatment_col, outcome_col)
                ],
                help="Variables to control for in matching"
            )
        
        if st.button("🚀 Run PSM Analysis", type="primary"):
            if not covariate_cols:
                st.error("❌ Please select at least one covariate for matching")
            else:
                with st.spinner("Running Propensity Score Matching..."):
                    try:
                        psm_results = st.session_state.causal_lab.propensity_score_matching(
                            st.session_state.data_handler.data,
                            treatment_col,
                            outcome_col,
                            covariate_cols,
                            caliper=caliper,
                            treated_value=treated_value
                        )
                    except Exception as e:
                        psm_results = {'error': str(e)}
                    
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
                    did_treatment_group = st.selectbox(
                        "Treatment Group Value",
                        options=st.session_state.data_handler.data[group_col_did].unique()
                    )
                    
                    post_period = st.selectbox(
                        "Post-Treatment Period Value",
                        options=st.session_state.data_handler.data[time_col].unique()
                    )
            
            cluster_choice = st.selectbox(
                "Cluster Standard Errors By (Optional)",
                options=['(none)'] + [
                    c for c in st.session_state.data_handler.data.columns
                    if c not in (group_col_did, time_col, outcome_col_did)
                ],
                help="Unit observed repeatedly, e.g. store_id or user_id"
            )
            
            event_period_cols = st.multiselect(
                "Time Index Columns for Pre-Trend Test (Optional)",
                options=[
                    c for c in st.session_state.data_handler.data.columns
                    if c not in (group_col_did, time_col, outcome_col_did)
                ],
                help="Columns that order time in finer steps than pre/post, e.g. year then quarter. "
                     "With several pre-treatment periods the parallel-trends assumption can be tested."
            )
            first_treated_label = None
            if event_period_cols:
                ordered_periods = sorted(
                    st.session_state.data_handler.data[event_period_cols]
                    .dropna().apply(tuple, axis=1).unique()
                )
                period_labels = {' / '.join(map(str, k)): k for k in ordered_periods}
                post_rows = st.session_state.data_handler.data[
                    st.session_state.data_handler.data[time_col] == post_period
                ]
                default_first = (
                    min(post_rows[event_period_cols].dropna().apply(tuple, axis=1))
                    if len(post_rows) else ordered_periods[-1]
                )
                first_treated_label = st.selectbox(
                    "First Treated Period",
                    options=list(period_labels),
                    index=ordered_periods.index(default_first)
                )
        
        if st.button("🚀 Run DiD Analysis", type="primary"):
            with st.spinner("Running Difference-in-Differences..."):
                try:
                    did_results = st.session_state.causal_lab.difference_in_differences(
                        st.session_state.data_handler.data,
                        group_col_did,
                        time_col,
                        outcome_col_did,
                        did_treatment_group,
                        post_period,
                        cluster_col=None if cluster_choice == '(none)' else cluster_choice
                    )
                except Exception as e:
                    st.error(f"❌ {str(e)}")
                    return
                
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
                st.subheader("📈 Group Means Before and After")
                fig_trends = st.session_state.visualizer.plot_did_trends(
                    did_results['group_time_means']
                )
                st.plotly_chart(fig_trends, use_container_width=True)
                
                # Event study: test parallel trends when finer time periods are available
                if event_period_cols and first_treated_label:
                    st.subheader("🔎 Pre-Trend Test (Event Study)")
                    try:
                        es_results = st.session_state.causal_lab.event_study(
                            st.session_state.data_handler.data,
                            group_col_did,
                            event_period_cols,
                            outcome_col_did,
                            did_treatment_group,
                            period_labels[first_treated_label],
                            cluster_col=None if cluster_choice == '(none)' else cluster_choice
                        )
                        if es_results['parallel_trends_assumption']:
                            st.success(
                                f"✅ No evidence of diverging pre-trends "
                                f"(joint test p={es_results['pre_trend_p_value']:.4f} across "
                                f"{es_results['n_pre_periods_tested']} pre-periods). "
                                "This supports, but cannot prove, the parallel-trends assumption."
                            )
                        else:
                            st.error(
                                f"🛑 Pre-treatment trends differ between groups "
                                f"(joint test p={es_results['pre_trend_p_value']:.4f}). "
                                "The DiD estimate above is not credible as a causal effect."
                            )
                        st.plotly_chart(
                            st.session_state.visualizer.plot_event_study(es_results['coefficients']),
                            use_container_width=True
                        )
                        st.session_state.event_study_results = es_results
                    except ValueError as e:
                        st.warning(f"⚠️ Pre-trend test skipped: {str(e)}")
                
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
                try:
                    iv_results = st.session_state.causal_lab.instrumental_variables(
                        st.session_state.data_handler.data,
                        outcome_col_iv,
                        treatment_col_iv,
                        instrument_col,
                        covariate_cols_iv if covariate_cols_iv else None
                    )
                except Exception as e:
                    st.error(f"❌ {str(e)}")
                    iv_results = {}
                
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
                    
                    st.subheader("💡 Causal Interpretation")
                    st.markdown(
                        st.session_state.interpreter.interpret_causal_effect(iv_results, "IV")
                    )
                    
                    st.session_state.iv_results = iv_results


# Main content - Tabs
if has_data:
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
        render_ab_tab()
    
    # ==================== TAB 3: CAUSAL INFERENCE ====================
    with tab3:
        render_causal_tab()
    
    # ==================== TAB 4: REPORT EXPORT ====================
    with tab4:
        st.header("📄 Report Export")
        
        st.write("Export your analysis results in various formats for sharing and archiving.")
        
        # Check if results exist
        has_ab_results = 'ab_test_results' in st.session_state
        has_psm_results = 'psm_results' in st.session_state
        has_did_results = 'did_results' in st.session_state
        has_iv_results = 'iv_results' in st.session_state
        has_mv_results = 'multi_variant_results' in st.session_state
        
        if not (has_ab_results or has_psm_results or has_did_results or has_iv_results or has_mv_results):
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
            if has_iv_results:
                export_options.append("IV Results")
            if has_mv_results:
                export_options.append("Multi-Variant Results")
            
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
                    elif selected_export == "IV Results":
                        results = st.session_state.iv_results
                        interpretation = st.session_state.interpreter.interpret_causal_effect(results, "IV")
                        test_type = "Instrumental Variables (2SLS)"
                    elif selected_export == "Multi-Variant Results":
                        results = st.session_state.multi_variant_results
                        interpretation = st.session_state.interpreter.interpret_multi_variant(results)
                        test_type = f"Multi-Variant ({results['correction']} correction)"
                    else:
                        results = st.session_state.did_results
                        interpretation = st.session_state.interpreter.interpret_causal_effect(results, "DiD")
                        test_type = "Difference-in-Differences"
                    
                    decision = (
                        st.session_state.get('ab_decision')
                        if selected_export == "A/B Test Results" else None
                    )
                    
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
                            test_type,
                            decision=decision
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
                            test_type,
                            decision=decision
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
    st.info("👈 Upload a dataset or pick a sample dataset in the sidebar to begin your analysis")
    
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
