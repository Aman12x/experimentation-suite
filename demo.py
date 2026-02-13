#!/usr/bin/env python3
"""
Standalone Demo of the Experimentation & Causal Analysis Suite
Runs core statistical functions without requiring Streamlit
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

print("=" * 70)
print("🔬 EXPERIMENTATION & CAUSAL ANALYSIS SUITE - DEMO")
print("=" * 70)
print()

# Load sample data
print("📁 Loading sample A/B test data...")
df = pd.read_csv('sample_ab_test_data.csv')
print(f"✅ Loaded {len(df):,} rows")
print()

print("📊 DATA OVERVIEW")
print("-" * 70)
print(df.head())
print()
print("Group Summary:")
summary = df.groupby('group').agg({
    'revenue': ['count', 'mean', 'std'],
    'conversion': ['sum', 'mean']
})
print(summary)
print()

# Prepare data
control_data = df[df['group'] == 'control']['revenue'].dropna().values
treatment_data = df[df['group'] == 'treatment']['revenue'].dropna().values

print("=" * 70)
print("🏥 HEALTH CHECKS")
print("=" * 70)
print()

# Sample Ratio Mismatch Check
total = len(df.dropna(subset=['revenue']))
control_n = len(control_data)
treatment_n = len(treatment_data)
expected = np.array([0.5 * total, 0.5 * total])
observed = np.array([control_n, treatment_n])
chi2_stat = np.sum((observed - expected)**2 / expected)
srm_p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)

print(f"Sample Ratio Mismatch Check:")
print(f"  Expected: [0.500, 0.500]")
print(f"  Observed: [{control_n/total:.3f}, {treatment_n/total:.3f}]")
print(f"  P-value: {srm_p_value:.6f}")
print(f"  Status: {'✅ OK' if srm_p_value > 0.001 else '⚠️ CRITICAL'}")
print()

# Outlier check
q1, q3 = np.percentile(df['revenue'].dropna(), [25, 75])
iqr = q3 - q1
outliers = df[(df['revenue'] < q1 - 3*iqr) | (df['revenue'] > q3 + 3*iqr)]
outlier_pct = len(outliers) / len(df.dropna(subset=['revenue'])) * 100
print(f"Outlier Check:")
print(f"  Outliers: {len(outliers)} ({outlier_pct:.1f}%)")
print(f"  Status: {'✅ OK' if outlier_pct < 5 else '⚠️ WARNING'}")
print()

# Variance ratio
control_var = np.var(control_data, ddof=1)
treatment_var = np.var(treatment_data, ddof=1)
var_ratio = max(control_var, treatment_var) / min(control_var, treatment_var)
print(f"Variance Ratio Check:")
print(f"  Ratio: {var_ratio:.2f}")
print(f"  Status: {'✅ OK' if var_ratio < 4 else '⚠️ WARNING - Consider Welch t-test'}")
print()

print("=" * 70)
print("🧪 T-TEST RESULTS: REVENUE METRIC")
print("=" * 70)
print()

# T-test
t_stat, p_value = stats.ttest_ind(treatment_data, control_data)
control_mean = np.mean(control_data)
treatment_mean = np.mean(treatment_data)
control_std = np.std(control_data, ddof=1)
treatment_std = np.std(treatment_data, ddof=1)

# Effect size
pooled_std = np.sqrt(
    ((control_n - 1) * control_var + (treatment_n - 1) * treatment_var) / 
    (control_n + treatment_n - 2)
)
cohens_d = (treatment_mean - control_mean) / pooled_std

# Confidence interval
se = pooled_std * np.sqrt(1/control_n + 1/treatment_n)
dof = control_n + treatment_n - 2
t_critical = stats.t.ppf(0.975, dof)
ci_lower = (treatment_mean - control_mean) - t_critical * se
ci_upper = (treatment_mean - control_mean) + t_critical * se

relative_lift = ((treatment_mean - control_mean) / control_mean) * 100

print(f"Control Group:")
print(f"  Mean: ${control_mean:.2f}")
print(f"  Std Dev: ${control_std:.2f}")
print(f"  Sample Size: {control_n:,}")
print()

print(f"Treatment Group:")
print(f"  Mean: ${treatment_mean:.2f}")
print(f"  Std Dev: ${treatment_std:.2f}")
print(f"  Sample Size: {treatment_n:,}")
print()

print(f"Statistical Results:")
print(f"  Difference: ${treatment_mean - control_mean:.2f}")
print(f"  Relative Lift: {relative_lift:+.2f}%")
print(f"  T-statistic: {t_stat:.4f}")
print(f"  P-value: {p_value:.4f}")
print(f"  Significant: {'✅ YES (p < 0.05)' if p_value < 0.05 else '❌ NO (p ≥ 0.05)'}")
print(f"  95% CI: [${ci_lower:.2f}, ${ci_upper:.2f}]")
print(f"  Cohen's d: {cohens_d:.4f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")
print()

print("=" * 70)
print("💡 BUSINESS INTERPRETATION")
print("=" * 70)
print()

if p_value < 0.05:
    print(f"✅ STATISTICALLY SIGNIFICANT RESULT")
    print()
    print(f"The treatment group shows a {abs(relative_lift):.2f}% {'increase' if relative_lift > 0 else 'decrease'}")
    print(f"in revenue compared to control. With a p-value of {p_value:.4f}, we have strong")
    print(f"evidence this is a real effect, not due to chance.")
    print()
    if abs(relative_lift) > 5:
        print(f"💰 RECOMMENDATION: IMPLEMENT")
        print(f"The {abs(relative_lift):.1f}% lift is both statistically significant and")
        print(f"practically meaningful. Strong candidate for rollout.")
    else:
        print(f"⚠️ RECOMMENDATION: PROCEED WITH CAUTION")
        print(f"While significant, the {abs(relative_lift):.1f}% effect is modest.")
        print(f"Weigh implementation costs against the gains.")
else:
    print(f"❌ NOT STATISTICALLY SIGNIFICANT")
    print()
    print(f"While we observed a {abs(relative_lift):.2f}% {'increase' if relative_lift > 0 else 'decrease'},")
    print(f"the p-value of {p_value:.4f} means this could easily be due to random chance.")
    print()
    print(f"🛑 RECOMMENDATION: DO NOT IMPLEMENT")
    print(f"Insufficient evidence of a real effect. Consider running a larger test.")

print()

print("=" * 70)
print("📏 POWER ANALYSIS")
print("=" * 70)
print()

# Calculate sample size for 5% MDE
mde = 5.0  # 5% minimum detectable effect
effect_size_abs = (mde / 100) * control_mean
effect_size_d = effect_size_abs / pooled_std

try:
    from statsmodels.stats.power import tt_ind_solve_power
    n_required = tt_ind_solve_power(
        effect_size=effect_size_d,
        alpha=0.05,
        power=0.80,
        ratio=1.0,
        alternative='two-sided'
    )
    total_required = int(np.ceil(n_required * 2))
    
    print(f"To detect a {mde}% effect with 80% power and α=0.05:")
    print(f"  Required per group: {int(np.ceil(n_required)):,}")
    print(f"  Total required: {total_required:,}")
    print(f"  Current sample: {control_n + treatment_n:,}")
    print(f"  Status: {'✅ SUFFICIENT' if (control_n + treatment_n) >= total_required else '⚠️ UNDERPOWERED'}")
except ImportError:
    print("⚠️ Power analysis requires statsmodels (not installed in demo environment)")
    print(f"   Current sample: {control_n + treatment_n:,}")
    print(f"   Effect detected: {abs(relative_lift):.1f}% lift with Cohen's d = {cohens_d:.3f}")

print()

print("=" * 70)
print("🎯 CHI-SQUARED TEST: CONVERSION RATE")
print("=" * 70)
print()

control_conversions = df[df['group'] == 'control']['conversion'].sum()
treatment_conversions = df[df['group'] == 'treatment']['conversion'].sum()
control_total = len(df[df['group'] == 'control'])
treatment_total = len(df[df['group'] == 'treatment'])

# Contingency table
observed = np.array([
    [treatment_conversions, treatment_total - treatment_conversions],
    [control_conversions, control_total - control_conversions]
])

chi2, p_val_chi, dof, expected = stats.chi2_contingency(observed)

control_rate = control_conversions / control_total
treatment_rate = treatment_conversions / treatment_total
rate_lift = ((treatment_rate - control_rate) / control_rate) * 100

print(f"Control: {control_conversions}/{control_total} = {control_rate*100:.2f}%")
print(f"Treatment: {treatment_conversions}/{treatment_total} = {treatment_rate*100:.2f}%")
print(f"Relative Lift: {rate_lift:+.2f}%")
print(f"Chi-squared: {chi2:.4f}")
print(f"P-value: {p_val_chi:.4f}")
print(f"Significant: {'✅ YES' if p_val_chi < 0.05 else '❌ NO'}")
print()

print("=" * 70)
print("🎲 BAYESIAN A/B TEST: CONVERSION RATE")
print("=" * 70)
print()

# Bayesian with Beta-Binomial
prior_alpha, prior_beta = 1, 1
control_alpha = prior_alpha + control_conversions
control_beta = prior_beta + (control_total - control_conversions)
treatment_alpha = prior_alpha + treatment_conversions
treatment_beta = prior_beta + (treatment_total - treatment_conversions)

# Monte Carlo simulation
np.random.seed(42)
n_sims = 100000
control_samples = np.random.beta(control_alpha, control_beta, n_sims)
treatment_samples = np.random.beta(treatment_alpha, treatment_beta, n_sims)

prob_treatment_better = np.mean(treatment_samples > control_samples)
lift_samples = (treatment_samples - control_samples) / control_samples
expected_lift = np.mean(lift_samples) * 100
lift_ci_lower = np.percentile(lift_samples, 2.5) * 100
lift_ci_upper = np.percentile(lift_samples, 97.5) * 100

print(f"Probability Treatment Better: {prob_treatment_better*100:.1f}%")
print(f"Expected Lift: {expected_lift:+.2f}%")
print(f"95% Credible Interval: [{lift_ci_lower:.2f}%, {lift_ci_upper:.2f}%]")
print()

if prob_treatment_better > 0.95:
    print("✅ HIGH CONFIDENCE: Very strong evidence in favor of treatment")
elif prob_treatment_better > 0.90:
    print("✅ GOOD CONFIDENCE: Good evidence in favor of treatment")
elif prob_treatment_better < 0.10:
    print("❌ HIGH CONFIDENCE TREATMENT IS WORSE")
else:
    print("⚠️ UNCERTAIN: Not enough evidence either way")

print()

# Load DiD data
print("=" * 70)
print("📊 DIFFERENCE-IN-DIFFERENCES DEMO")
print("=" * 70)
print()

did_df = pd.read_csv('sample_did_data.csv')
print(f"📁 Loaded DiD data: {len(did_df):,} rows")
print()

# Calculate DiD manually
means = did_df.groupby(['region', 'period'])['sales'].mean().unstack()
print("Group × Time Means:")
print(means)
print()

pre_diff = means.loc['treatment', 'pre'] - means.loc['control', 'pre']
post_diff = means.loc['treatment', 'post'] - means.loc['control', 'post']
did_estimate = post_diff - pre_diff

print(f"Pre-treatment difference: {pre_diff:.2f}")
print(f"Post-treatment difference: {post_diff:.2f}")
print(f"DiD Estimate (Treatment Effect): {did_estimate:.2f}")
print()

# Regression approach
did_df['treated'] = (did_df['region'] == 'treatment').astype(int)
did_df['post'] = (did_df['period'] == 'post').astype(int)
did_df['treated_post'] = did_df['treated'] * did_df['post']

try:
    import statsmodels.formula.api as smf
    model = smf.ols('sales ~ treated + post + treated_post', data=did_df).fit()

    print("Regression Results:")
    print(f"  DiD Coefficient: {model.params['treated_post']:.2f}")
    print(f"  P-value: {model.pvalues['treated_post']:.4f}")
    print(f"  R-squared: {model.rsquared:.4f}")
    print(f"  Significant: {'✅ YES' if model.pvalues['treated_post'] < 0.05 else '❌ NO'}")
    print()
except ImportError:
    print("⚠️ Regression analysis requires statsmodels (not installed in demo environment)")
    print(f"   DiD Estimate from means: {did_estimate:.2f}")
    print()

parallel_trends = abs(pre_diff) < abs(did_estimate) * 0.1
print(f"Parallel Trends Check: {'✅ PASS' if parallel_trends else '⚠️ WARNING'}")
print()

print("=" * 70)
print("✅ DEMO COMPLETE!")
print("=" * 70)
print()
print("This demo showcased:")
print("  ✓ Sample Ratio Mismatch detection")
print("  ✓ Outlier analysis")
print("  ✓ T-test with effect sizes")
print("  ✓ Power analysis")
print("  ✓ Chi-squared test for proportions")
print("  ✓ Bayesian A/B testing")
print("  ✓ Difference-in-Differences")
print()
print("To use the full interactive Streamlit UI:")
print("  1. pip install -r requirements.txt")
print("  2. streamlit run app.py")
print("  3. Upload the sample CSV files provided")
print()
