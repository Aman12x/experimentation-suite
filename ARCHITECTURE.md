# 🏗️ Architecture

## One engine, four front ends

```
  Streamlit UI (app.py)   REST API (api_server.py)   MCP server (mcp_server.py)   CLI (demo.py, validate.py)
            │                          │                            │
            └──────────────┬───────────┴────────────────────────────┘
                           ▼
        modules/                                  utils/
        ├─ ab_testing.py        core tests        ├─ decision.py          ship verdict
        ├─ ab_advanced.py       CUPED, sequential ├─ interpreters.py      plain English
        ├─ causal_inference.py  PSM, DiD, IV      └─ report_generator.py  exports
        ├─ health_checks.py     SRM, data quality
        ├─ data_handler.py      loading
        └─ visualizations.py    Plotly figures
                           │
                           ▼
              NumPy · SciPy · statsmodels · scikit-learn · pandas
```

The modules never import Streamlit or FastAPI. They take arrays or DataFrames, return plain dictionaries of native Python types, and raise `ValueError` on bad input. Each front end decides how to show an error: the UI renders it, the API turns it into a 400. This is what lets the UI and the API share one implementation, and what makes the statistics testable without a server.

## Result contract

Every two-sample test returns the same core keys, so interpretation, the ship decision, reports, and charts work with any of them:

`test_type`, `p_value`, `significant`, `alpha`, `control_mean`, `treatment_mean`, `control_n`, `treatment_n`, `mean_difference`, `relative_lift`, and, where the method defines them, `ci_lower` / `ci_upper` (absolute) and `lift_ci_lower` / `lift_ci_upper` (relative).

Method-specific extras sit alongside (`variance_reduction_pct` for CUPED, `looks` for sequential, `comparisons` for multi-variant).

## Statistical choices

| Choice | Reason |
|---|---|
| Welch's t-test is the default | The pooled test's false-positive rate breaks down when group sizes and variances both differ, which traffic splits like 90/10 produce routinely |
| Delta method for relative lift | Dividing the absolute interval by the control mean treats that mean as a known constant |
| Binary-only tests refuse continuous metrics | Thresholding revenue at the median to feed a proportions test answers a different question than the one asked |
| SRM at α = 0.001 | It is a data-quality alarm run on every analysis, so it must rarely cry wolf |
| A check that cannot run is a warning | "Could not check" must never render as "healthy" |
| Mixture SPRT for sequential testing | Always-valid p-values allow stopping at any look; the price is lower power than a single fixed-horizon test |
| Holm as the default correction | Controls family-wise error like Bonferroni and is never less powerful |
| 2SLS through `statsmodels` `IV2SLS` | Running the second stage as ordinary OLS gives the right coefficient and the wrong standard errors |
| Partial F for instrument strength | The overall first-stage F also credits the covariates, so a useless instrument can look strong |
| Matched-pair standard errors for PSM | Matched units are not independent samples |
| Two-period DiD reports parallel trends as untestable | A gap in levels is allowed; only trends matter, and two periods carry no trend information |
| Analysis runs at the unit of randomization | Rows from the same user are correlated; treating sessions as independent users makes standard errors too small |
| The test defaults to Auto | The metric's type decides the test (rate → proportions, mean → Welch, heavy tail → bootstrap), so a user cannot silently pair a yes/no metric with the wrong test |
| Ratio metrics use the delta method on per-unit totals | The business number is total numerator / total denominator; the mean of per-unit ratios down-weights heavy users and is a different estimand |
| Relative lift can be `None` | Against a zero, negative, or noise-level baseline a percentage is meaningless or has the wrong sign; every consumer falls back to the absolute change |
| The bootstrap resamples in chunks, scales its resample count, and refuses very large inputs | Memory stays flat on a 1 GiB instance; beyond a few hundred thousand units Welch gives the same answer |
| A single day with a broken traffic split invalidates the decision | Overall counts can average out a day where assignment or logging failed |
| Dependencies are locked | The image is rebuilt on every push; without a lock each deploy could pick up untested library releases |
| Tests also run from summary statistics | Means, variances and covariances are sufficient for every parametric test here, so a warehouse can aggregate any number of rows and ship back a handful of numbers. Equality with the row-level path is enforced by running the generated SQL on DuckDB in the tests |
| The SQL builder validates identifiers and takes no free-text SQL | Its callers include LLM agents; whatever they are told, the only statement it can emit is the aggregate query |
| The suite is an MCP server, not an MCP client | MCP is for a model choosing tools. The agent pairs this server with a warehouse server; a form that already knows its query would use a plain database driver instead |
| MCP tool errors carry the validation message | The SDK hides the text of unexpected exceptions from clients. Bad input is raised as a `ToolError` so the agent can read what to fix |
| HTTP mode refuses to start without a bearer token | A statistics endpoint is harmless; an unauthenticated public compute endpoint is not |
| Missing values raise | A NaN would otherwise propagate into a NaN statistic |

## Decision precedence

`utils/decision.py` applies, in order: sample ratio mismatch → any guardrail significantly harmed → primary significantly worse → primary significantly better → interval rules out the minimum lift of interest (stop) → otherwise keep running. Direction is configurable per metric so that latency or churn can be guardrails.

## Testing strategy

| Layer | What it proves | Where |
|---|---|---|
| Reference checks | Output equals SciPy / statsmodels on the same input | `test_statistical_validity.py`, `test_advanced_methods.py` |
| Simulations | Long-run guarantees hold: false-positive rate, interval coverage, family-wise error, peeking, CUPED bias and power, 2SLS coverage | same files, marked `slow` |
| Known-effect recovery | PSM, DiD, event study, and IV recover a planted effect that a naive comparison misses | `test_causal_inference.py`, `test_decision_and_event_study.py` |
| Properties | Generated inputs never produce NaN, infinities, invalid JSON, a wrong-direction recommendation, or a lost unit | `test_properties.py` |
| Behaviour | Recommendations respect direction; guardrails block; reports escape input | `test_health_and_interpretation.py`, `test_reports.py` |
| Warehouse path | Generated SQL run on DuckDB, fed to the from-stats methods, equals the row-level analysis to 1e-9; injection attempts are refused | `test_from_stats_and_sql.py` |
| MCP | The agent workflow end to end against DuckDB, tool errors, a real stdio subprocess, bearer auth over HTTP | `test_mcp_server.py` |
| API | Every advertised route exists, validates input, and returns JSON-safe output | `test_api.py` |
| End to end | The Streamlit app is driven headlessly through each flow on the sample data | `test_app.py` |

`validate.py` prints the simulation results as a table for the README.

## Extending

- **New test**: add a method to `AdvancedABMethods` that returns the result contract above. It then works with `ship_decision`, the interpreter, and exports. Add it to `TEST_TYPES` and `run_selected_test` in `app.py`, and a route in `api_server.py`.
- **New causal method**: add to `CausalInferenceLab`, plus a branch in `StatisticalInterpreter.interpret_causal_effect`.
- **New export format**: add a `create_*_report` method to `ReportGenerator`.

## Limits

- Data is held in memory; the app is meant for experiment-sized extracts, not warehouse tables.
- PSM is greedy 1:1 matching, O(treated × control).
- The API has no authentication. Put it behind your own gateway before exposing it.
