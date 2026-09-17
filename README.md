# 🔬 Experimentation & Causal Analysis Suite

[![Tests](https://github.com/Aman12x/experimentation-suite/actions/workflows/tests.yml/badge.svg)](https://github.com/Aman12x/experimentation-suite/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An A/B testing and causal inference toolkit that answers the question a product team actually asks: **ship it, don't ship it, or keep running?** One statistics engine sits behind a Streamlit app, a REST API, and a command-line demo, and every method is checked by simulation against the guarantee it claims.

**Live app:** https://experimentation-suite-526867948326.us-east4.run.app

---

## What it does

**A/B testing**
- Welch's t-test by default (Student's is opt-in), z-test, two-proportion z-test, chi-squared
- Confidence interval on **relative lift** via the delta method, not just on the absolute difference
- **CUPED** variance reduction with a pre-experiment covariate
- **Sequential testing** with always-valid p-values (mixture SPRT), so checking results daily does not inflate false positives
- **Multiple variants** against one control with Holm, Bonferroni, or Benjamini-Hochberg correction
- Mann-Whitney U and percentile bootstrap for skewed metrics such as revenue
- Bayesian Beta-Binomial test with probability of being better, credible interval on lift, and expected loss per arm
- Power analysis and sample-size calculator

**Guided experiment definition**
- The app asks four design questions instead of showing a wall of column pickers: *who was randomized, what are you trying to move, what must not break, how should it be analyzed*
- **Unit of randomization**: pick the ID of whatever was assigned. If the table has several rows per unit (sessions, orders), they are combined to one row per unit before testing, and units seen in more than one arm are dropped as contaminated
- **One primary metric**, with its direction (up or down is a win) stated explicitly. The app says what the metric is (a rate, a mean, a heavy-tailed count) and picks the matching test; a mismatched manual choice is called out before it runs
- Planned traffic split feeds the sample ratio check
- A plain-English statement of the experiment is shown before you run it

**Decision layer**
- Guardrail metrics, each with its own direction (higher or lower is better)
- A single verdict with reasons: `SHIP`, `DO NOT SHIP`, `KEEP RUNNING`, `STOP - NO MEANINGFUL EFFECT`, or `INVALID - FIX THE EXPERIMENT`
- Health checks that run before any result is shown: sample ratio mismatch (any number of arms, any expected split), outliers, missing data, variance ratio, normality

**Causal inference for observational data**
- Propensity score matching (1:1 nearest neighbour within a caliper) with balance before and after
- Difference-in-differences with heteroskedasticity-robust or cluster-robust standard errors
- Event-study DiD with a joint **pre-trend test** when several pre-treatment periods exist
- Instrumental variables by 2SLS, with a weak-instrument check on the instrument's **partial** first-stage F

**Output**
- Plain-English interpretation of every result
- Excel, Markdown, and HTML reports that lead with the decision

---

## Does the statistics hold up?

`python validate.py` simulates data where the truth is known and measures each method's long-run behaviour next to the naive alternative. Output from the current code:

| Check | Target | Measured | Naive alternative |
|---|---|---|---|
| t-test false-positive rate (unequal n and variance) | 5% | 5.5% | Student's pooled: 44.3% |
| Relative-lift 95% CI coverage | 95% | 94.7% | absolute CI / control mean: 93.7% |
| CUPED power to detect a true +1.5 effect (n=400/arm) | higher is better | 75.6% (variance cut 83%) | plain Welch: 16.6% |
| Chance of any false win, 5 null variants | ≤ 5% | 3.2% | uncorrected: 18.5% |
| False-positive rate with 20 interim looks | ≤ 5% | 1.2% | peeking at fixed-horizon p: 25.4% |
| 2SLS 95% CI coverage with an unobserved confounder | 95% | 95.0% | OLS within ±0.1 of truth: 0.0% |
| False-positive rate when users are randomized but rows are sessions | 5% | 4.5% (one row per user) | sessions treated as independent: 45.2% |

The same properties are enforced with tolerances in `tests/`, so a regression in any of them fails CI.

---

## Quick start

```bash
git clone https://github.com/Aman12x/experimentation-suite.git
cd experimentation-suite
pip install -r requirements.txt

streamlit run app.py        # UI on http://localhost:8501
python api_server.py        # API on http://localhost:8000, docs at /docs
python demo.py              # command-line tour, no server
python validate.py          # simulation report shown above
```

No data needed: pick one of the bundled datasets under **"Or try a sample dataset"** in the sidebar. The *multi-variant checkout test* shows the most in one run: three arms, a covariate for CUPED, and a variant that wins on revenue while slowing page loads.

### Docker

```bash
docker compose up           # UI on :8501, API on :8000
```

---

## The demo in 20 lines

`python demo.py`, abridged:

```
Revenue vs control (holm-corrected):
  express_pay          lift  +4.72%   p_raw=0.0085   p_adj=0.0170   significant
  one_page_checkout    lift  +2.01%   p_raw=0.2526   p_adj=0.2526   not significant

  one_page_checkout: KEEP RUNNING
    - No significant effect yet (p=0.2526), and the interval [-1.47%, +5.48%] still allows a meaningful lift.

  express_pay: DO NOT SHIP
    - Guardrail 'page_load_ms' got significantly worse (+12.17%, p=0.0000).
    - Primary metric improved (+4.72%), but not at the cost of a guardrail.

CUPED: same data, tighter interval
  Covariate correlation:   0.97
  Variance removed:        93.3%
  Plain Welch  CI: [-1.21, +4.61]  p=0.2526
  CUPED        CI: [+2.74, +4.26]  p=0.0000
```

`one_page_checkout` looks like a null result until CUPED removes the noise that pre-experiment spend already explains. `express_pay` wins the primary metric and is still blocked, because it costs page speed.

---

## Python API

```python
from modules import ABTestingEngine
from utils import ship_decision

engine = ABTestingEngine()

primary = engine.cuped_test(control_revenue, treatment_revenue,
                            control_pre_revenue, treatment_pre_revenue)
latency = engine.t_test(control_latency, treatment_latency)

verdict = ship_decision(
    primary,
    guardrails={"latency": latency},
    guardrail_higher_is_better={"latency": False},
    mde_pct=2.0,
)
print(verdict["decision"], verdict["reasons"])
```

```python
from modules import CausalInferenceLab

lab = CausalInferenceLab()
did = lab.difference_in_differences(df, "region", "period", "sales",
                                    treatment_group="treatment", post_period="post",
                                    cluster_col="store_id")
pre_trends = lab.event_study(df, "region", ["year", "quarter"], "sales",
                             treatment_group="treatment", first_treated_period=(2024, 1),
                             cluster_col="store_id")
```

## REST API

| Endpoint | Purpose |
|---|---|
| `POST /api/ab-test/t-test`, `/z-test`, `/proportions`, `/chi-squared` | Frequentist tests |
| `POST /api/ab-test/mann-whitney`, `/bootstrap` | Distribution-free tests |
| `POST /api/ab-test/cuped` | Variance reduction |
| `POST /api/ab-test/multi-variant` | Many variants, corrected p-values |
| `POST /api/ab-test/sequential` | Always-valid monitoring |
| `POST /api/ab-test/bayesian`, `/power-analysis` | Bayesian test, sample size |
| `POST /api/health-check` | SRM and data-quality checks |
| `POST /api/decision` | Ship decision from test outputs |
| `POST /api/causal/psm`, `/did`, `/iv` | Causal methods on row records |

```bash
curl -X POST http://localhost:8000/api/ab-test/proportions \
  -H "Content-Type: application/json" \
  -d '{"control_success": 50, "control_total": 1000, "treatment_success": 65, "treatment_total": 1000}'
```

Invalid input returns `400` or `422` with the reason. Interactive docs are at `/docs`.

---

## Project layout

```
app.py                  Streamlit UI
api_server.py           FastAPI server (same engine as the UI)
demo.py                 Command-line tour
validate.py             Simulation report
modules/
  ab_testing.py         Core tests, power analysis, Bayesian
  ab_advanced.py        Proportions, lift CI, CUPED, bootstrap, multi-variant, sequential
  causal_inference.py   PSM, DiD, event study, IV
  health_checks.py      SRM and data-quality checks
  experiment_design.py  Unit-level aggregation, metric typing, test recommendation
  data_handler.py       Loading and validation
  visualizations.py     Plotly charts
utils/
  decision.py           Ship decision
  interpreters.py       Plain-English explanations
  report_generator.py   Excel / Markdown / HTML export
data/                   Sample datasets and the generator for the synthetic one
tests/                  Unit, simulation, API, and headless end-to-end app tests
```

The modules have no Streamlit dependency, so they import cleanly into notebooks, jobs, or the API. See [ARCHITECTURE.md](ARCHITECTURE.md) for design notes and [QUICKSTART.md](QUICKSTART.md) for a guided first run.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                                   # everything
pytest -m "not slow"                     # skip the simulations
pytest --cov=modules --cov=utils         # CI fails under 80% coverage
```

## Assumptions worth knowing

- **Welch's t-test** compares means; with heavy skew and small samples prefer the bootstrap.
- **CUPED** needs a covariate measured *before* assignment. A post-assignment covariate biases the estimate.
- **Sequential test** uses a normal mixing prior whose width defaults to 10% of the pooled standard deviation. It trades some power for the right to stop at any look.
- **PSM** only balances what you measured. Unobserved confounders remain.
- **DiD** rests on parallel trends. Two periods cannot test it; the event study can support it but never prove it.
- **IV** needs an instrument that moves the treatment and touches the outcome through nothing else. Only the first part is testable.

## References

- Deng, Xu, Kohavi, Walker (2013). *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data* (CUPED)
- Johari, Koomen, Pekelis, Walsh (2017). *Peeking at A/B Tests: Why It Matters, and What to Do About It* (always-valid inference)
- Fabijan et al. (2019). *Diagnosing Sample Ratio Mismatch in Online Controlled Experiments*
- Holm (1979); Benjamini & Hochberg (1995). Multiple-comparison corrections
- Angrist & Pischke (2009). *Mostly Harmless Econometrics*
- Imbens & Rubin (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*
- Gelman et al. (2013). *Bayesian Data Analysis*

## License

MIT. See [LICENSE](LICENSE).
