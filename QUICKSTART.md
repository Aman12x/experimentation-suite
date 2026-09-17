# 🚀 Quick Start

## Install and launch

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## First analysis: a three-arm test with a guardrail

1. In the sidebar, under **Or try a sample dataset**, pick **Multi-variant checkout test (CUPED, guardrails)**.
2. Open the **🧪 A/B Testing** tab. `variant` and `control` are selected for you.
3. Set **Metric Column** to `revenue`.
4. Under **Guardrail Metrics**, add `page_load_ms`, and add it again under **Metrics Where Lower Is Better**.
5. Click **🚀 Run A/B Test**.

What to look for:
- **Health checks** run first. A sample ratio mismatch would invalidate everything below it.
- **All Variants vs Control** shows raw and Holm-corrected p-values side by side.
- **Decision** reads `DO NOT SHIP` for `express_pay`: it lifts revenue but slows pages.

## See CUPED tighten an interval

Same dataset. Untick **Compare all variants against control**, set **Treatment Group** to `one_page_checkout`, **Test Type** to **CUPED (variance reduction)**, and **Pre-Experiment Covariate** to `pre_revenue`. The plain t-test on this arm is not significant; with CUPED it is.

## Difference-in-differences with a pre-trend test

1. Pick the **Difference-in-differences (store sales)** sample.
2. **🎯 Causal Inference** tab, method **Difference-in-Differences (DiD)**.
3. Group `region`, time `period`, outcome `sales`, treatment group `treatment`, post period `post`.
4. **Cluster Standard Errors By**: `store_id`.
5. **Time Index Columns for Pre-Trend Test**: `year`, then `quarter`. First treated period defaults to `2024 / 1`.
6. Click **🚀 Run DiD Analysis**.

The event-study chart should show pre-treatment gaps around zero and a jump at treatment.

## Your own data

Upload a CSV or Parquet file with one row per unit. A/B tests need a group column and a numeric metric. Tests marked for binary metrics (Proportions, Chi-Squared, Bayesian) need a 0/1 column.

## Reading the results

| Term | Meaning |
|---|---|
| p-value | If there were no real difference, how often a result this extreme would appear |
| Relative lift CI | Range of plausible lifts. If it includes 0, direction is not settled |
| SRM | Traffic split differs from plan. Assignment or logging is broken; do not trust the result |
| Always-valid p-value | Safe to check at any time; a normal p-value is only valid if you look once |
| Adjusted p-value | Corrected for having tested several variants |

## Troubleshooting

- **Module not found**: `pip install --upgrade -r requirements.txt`
- **Port in use**: `streamlit run app.py --server.port 8502`
