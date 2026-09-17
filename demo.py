#!/usr/bin/env python3
"""
Command-line tour of the Experimentation & Causal Analysis Suite

Runs the same engine the Streamlit app and the REST API use, on the bundled
sample data. No server needed:

    python demo.py
"""

from pathlib import Path

import pandas as pd

from modules import ABTestingEngine, CausalInferenceLab, HealthChecker
from utils import ship_decision

DATA_DIR = Path(__file__).parent / 'data'
engine, lab = ABTestingEngine(), CausalInferenceLab()


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------- multi-variant
section("1. THREE-ARM CHECKOUT TEST: health checks, multi-variant, guardrails")
df = pd.read_csv(DATA_DIR / 'multivariant_checkout_test.csv')
print(f"{len(df):,} users across {df.variant.nunique()} variants")

health = HealthChecker().run_all_checks(df, 'variant', 'revenue')
for line in health['checks_passed'] + health['warnings']:
    print(f"  {line}")

groups = {g: d.revenue.values for g, d in df.groupby('variant')}
mv = engine.multi_variant_test(groups, 'control', correction='holm')
print(f"\nRevenue vs control ({mv['correction']}-corrected):")
for c in mv['comparisons']:
    print(f"  {c['group']:<20} lift {c['relative_lift']:+6.2f}%   "
          f"p_raw={c['p_value_raw']:.4f}   p_adj={c['p_value_adjusted']:.4f}   "
          f"{'significant' if c['significant'] else 'not significant'}")
print(f"  Best variant: {mv['best_variant']}")

control = df[df.variant == 'control']
for variant in ('one_page_checkout', 'express_pay'):
    arm = df[df.variant == variant]
    primary = engine.t_test(control.revenue.values, arm.revenue.values)
    latency = engine.t_test(control.page_load_ms.values, arm.page_load_ms.values)
    decision = ship_decision(
        primary, guardrails={'page_load_ms': latency},
        guardrail_higher_is_better={'page_load_ms': False}, health=health
    )
    print(f"\n  {variant}: {decision['decision']}")
    for reason in decision['reasons']:
        print(f"    - {reason}")

# ------------------------------------------------------------------------ CUPED
section("2. CUPED: same data, tighter interval")
arm = df[df.variant == 'one_page_checkout']
cuped = engine.cuped_test(
    control.revenue.values, arm.revenue.values,
    control.pre_revenue.values, arm.pre_revenue.values
)
print(f"  Covariate correlation:   {cuped['covariate_correlation']:.2f}")
print(f"  Variance removed:        {cuped['variance_reduction_pct']:.1f}%")
print(f"  Plain Welch  CI: [{cuped['unadjusted_ci_lower']:+.2f}, {cuped['unadjusted_ci_upper']:+.2f}]  "
      f"p={cuped['unadjusted_p_value']:.4f}")
print(f"  CUPED        CI: [{cuped['ci_lower']:+.2f}, {cuped['ci_upper']:+.2f}]  p={cuped['p_value']:.4f}")

# ------------------------------------------------------------------- sequential
section("3. SEQUENTIAL MONITORING: safe to peek")
seq = engine.sequential_test(control.revenue.values, df[df.variant == 'express_pay'].revenue.values)
total = seq['control_n'] + seq['treatment_n']
if seq['could_stop_at_look']:
    print(f"  Always-valid p-value crossed α at look {seq['could_stop_at_look']} of {len(seq['looks'])} "
          f"({seq['could_stop_at_n']:,} of {total:,} users)")
else:
    print(f"  No look crossed α; final always-valid p = {seq['p_value']:.4f}")

# ------------------------------------------------------------------ conversions
section("4. CONVERSION RATE: frequentist and Bayesian")
ab = pd.read_csv(DATA_DIR / 'sample_ab_test_data.csv')
counts = [
    int(ab[ab.group == g].conversion.sum()) if stat == 'sum' else int((ab.group == g).sum())
    for g in ('control', 'treatment') for stat in ('sum', 'n')
]
prop = engine.proportions_test(*counts)
bayes = engine.bayesian_ab_test(*counts)
print(f"  Control {prop['control_rate']:.2%} vs treatment {prop['treatment_rate']:.2%}")
print(f"  Lift {prop['relative_lift']:+.1f}% (95% CI {prop['lift_ci_lower']:+.1f}% to {prop['lift_ci_upper']:+.1f}%), "
      f"p={prop['p_value']:.4f}")
print(f"  P(treatment better) = {bayes['prob_treatment_better']:.1%}")

# -------------------------------------------------------------------------- DiD
section("5. DIFFERENCE-IN-DIFFERENCES with a pre-trend test")
did_df = pd.read_csv(DATA_DIR / 'sample_did_data.csv')
did = lab.difference_in_differences(
    did_df, 'region', 'period', 'sales', 'treatment', 'post', cluster_col='store_id'
)
print(f"  DiD estimate: {did['did_estimate']:+.2f}  "
      f"(95% CI {did['ci_lower']:+.2f} to {did['ci_upper']:+.2f}, {did['se_type']})")
es = lab.event_study(
    did_df, 'region', ['year', 'quarter'], 'sales', 'treatment', (2024, 1), cluster_col='store_id'
)
verdict = "no evidence of diverging pre-trends" if es['parallel_trends_assumption'] else "PRE-TRENDS DIFFER"
print(f"  Pre-trend joint test over {es['n_pre_periods_tested']} quarters: "
      f"p={es['pre_trend_p_value']:.4f} -> {verdict}")

print("\nDone. Launch the UI with `streamlit run app.py` or the API with `python api_server.py`.")
