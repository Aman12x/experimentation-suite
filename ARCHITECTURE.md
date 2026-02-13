# 🏗️ System Architecture

## Overview

The Experimentation & Causal Analysis Suite follows a modular, layered architecture designed for maintainability, testability, and extensibility.

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI Layer                      │
│                        (app.py)                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   Data   │  │   A/B    │  │  Causal  │  │  Report  │  │
│  │ Overview │  │  Testing │  │Inference │  │  Export  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Business Logic Layer                      │
│                       (modules/)                            │
│                                                             │
│  ┌───────────────────┐    ┌──────────────────────┐        │
│  │  Data Handler     │    │  Health Checker      │        │
│  │  - Load data      │    │  - SRM detection     │        │
│  │  - Validate       │    │  - Outlier checks    │        │
│  │  - Transform      │    │  - Normality tests   │        │
│  └───────────────────┘    └──────────────────────┘        │
│                                                             │
│  ┌───────────────────┐    ┌──────────────────────┐        │
│  │  A/B Testing      │    │  Causal Inference    │        │
│  │  - T-test         │    │  - PSM               │        │
│  │  - Z-test         │    │  - DiD               │        │
│  │  - Chi-squared    │    │  - IV/2SLS           │        │
│  │  - Bayesian       │    │  - Balance checks    │        │
│  │  - Power analysis │    │  - Trend analysis    │        │
│  └───────────────────┘    └──────────────────────┘        │
│                                                             │
│  ┌───────────────────────────────────────────────┐        │
│  │              Visualizer                        │        │
│  │  - Distribution plots  - Balance diagnostics  │        │
│  │  - Box plots          - Power curves          │        │
│  │  - CI plots           - Bayesian posteriors   │        │
│  │  - DiD trends         - Propensity scores     │        │
│  └───────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Utility Layer                             │
│                      (utils/)                               │
│                                                             │
│  ┌────────────────────────┐  ┌──────────────────────┐     │
│  │  Statistical           │  │  Report Generator    │     │
│  │  Interpreter           │  │  - Excel export      │     │
│  │  - P-value explains    │  │  - Markdown export   │     │
│  │  - CI interpretation   │  │  - HTML export       │     │
│  │  - Effect size guide   │  │  - Summary tables    │     │
│  │  - Business summaries  │  │                      │     │
│  └────────────────────────┘  └──────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Statistical Libraries                      │
│   SciPy  │  Statsmodels  │  NumPy  │  Pandas  │  Plotly   │
└─────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### 1. UI Layer (app.py)
**Purpose**: User interaction and workflow orchestration

**Responsibilities**:
- File upload handling
- Tab-based navigation
- User input collection
- Result presentation
- Error handling and user feedback

**Design Patterns**:
- Session state management for persistence
- Lazy loading of heavy computations
- Progressive disclosure of complexity

### 2. Data Handler (modules/data_handler.py)
**Purpose**: Data ingestion and validation

**Key Methods**:
- `load_data()`: CSV/Parquet import
- `validate_ab_test_columns()`: Pre-flight checks
- `prepare_ab_data()`: Data cleaning
- `get_group_stats()`: Descriptive statistics

**Validation Rules**:
- Minimum 2 groups
- Numeric metric columns
- Sufficient sample sizes
- Data type verification

### 3. A/B Testing Engine (modules/ab_testing.py)
**Purpose**: Statistical hypothesis testing

**Implemented Tests**:

| Test | Use Case | Assumptions |
|------|----------|-------------|
| T-Test | Continuous metrics, small-medium samples | Normality, independence |
| Z-Test | Large samples (n>30) | CLT applies, independence |
| Chi-Squared | Categorical outcomes | Independence, expected freq ≥5 |
| Bayesian | Binary outcomes with priors | Conjugate Beta-Binomial |

**Power Analysis**:
- Sample size calculation
- Effect size estimation
- Power curves generation

### 4. Causal Inference Lab (modules/causal_inference.py)
**Purpose**: Estimating causal effects from observational data

**Methods**:

#### Propensity Score Matching (PSM)
```
1. Fit logistic regression: P(Treatment=1|X)
2. Calculate propensity scores for all units
3. Match treated units to controls using nearest neighbor
4. Calculate ATT on matched sample
5. Assess covariate balance
```

**Assumptions**:
- Unconfoundedness (no unmeasured confounders)
- Overlap/common support
- Stable unit treatment value

#### Difference-in-Differences (DiD)
```
Model: Y_it = β₀ + β₁*Treated_i + β₂*Post_t + β₃*Treated_i×Post_t + ε_it

DiD Estimate = β₃ (coefficient on interaction term)
```

**Assumptions**:
- Parallel trends (key assumption)
- No anticipation effects
- Stable composition

#### Instrumental Variables (IV)
```
First Stage: Treatment ~ Instrument + Covariates
Second Stage: Outcome ~ Predicted_Treatment + Covariates

IV Estimate = Second stage coefficient on predicted treatment
```

**Assumptions**:
- Instrument relevance (F-stat > 10)
- Exclusion restriction
- Monotonicity

### 5. Health Checker (modules/health_checks.py)
**Purpose**: Automated data quality assurance

**Checks Performed**:

| Check | Criterion | Action if Failed |
|-------|-----------|------------------|
| Sample Ratio Mismatch | p < 0.001 | Critical warning |
| Outliers | >5% beyond 3×IQR | Warning |
| Missing Data | >10% | Warning |
| Variance Ratio | >4.0 | Suggest Welch's t-test |
| Normality | Shapiro-Wilk p < 0.05 | Note CLT or suggest non-parametric |

### 6. Visualizer (modules/visualizations.py)
**Purpose**: Interactive data visualization

**Chart Types**:
- **Distribution Comparison**: Overlapping histograms
- **Box Plots**: Quartile comparison with outliers
- **Confidence Intervals**: Effect size with uncertainty
- **Power Curves**: Sample size planning
- **Bayesian Posteriors**: Probability distributions
- **Propensity Scores**: Matching quality
- **DiD Trends**: Parallel trends assessment
- **Balance Diagnostics**: Covariate balance

**Design Principles**:
- Publication-quality aesthetics
- Interactive tooltips
- Color-blind friendly palette
- Responsive layouts

### 7. Statistical Interpreter (utils/interpreters.py)
**Purpose**: Plain English explanations

**Translation Examples**:

| Statistical Concept | Plain English |
|--------------------|---------------|
| p-value = 0.03 | "We have moderate evidence that the observed difference is real and not due to random chance" |
| Cohen's d = 0.6 | "This represents a medium increase in the metric" |
| CI: [0.5, 1.2] excludes 0 | "We can be confident the treatment has a positive effect" |
| ATT = +200 | "On average, the treatment caused a 200-unit increase" |

### 8. Report Generator (utils/report_generator.py)
**Purpose**: Multi-format export

**Export Formats**:

**Excel**:
- Summary sheet with key metrics
- Raw data sheet
- Formatted headers and styling

**Markdown**:
- Header with metadata
- Business interpretation
- Detailed results table
- Methodology notes

**HTML**:
- Professional styling
- Responsive design
- Embedded visualizations (placeholder)
- Print-friendly layout

## Data Flow

### A/B Test Workflow
```
1. User uploads CSV/Parquet
   ↓
2. DataHandler validates structure
   ↓
3. User selects group & metric columns
   ↓
4. HealthChecker runs diagnostics
   ↓
5. ABTestingEngine performs test
   ↓
6. Interpreter generates explanation
   ↓
7. Visualizer creates charts
   ↓
8. Results displayed in UI
   ↓
9. ReportGenerator creates export
```

### PSM Workflow
```
1. User uploads observational data
   ↓
2. DataHandler validates treatment/outcome/covariates
   ↓
3. CausalInferenceLab fits propensity model
   ↓
4. Matching algorithm creates pairs
   ↓
5. ATT calculated on matched sample
   ↓
6. Balance diagnostics computed
   ↓
7. Visualizer shows PS distributions & SMD
   ↓
8. Interpreter provides causal explanation
```

## Error Handling Strategy

### Validation Errors
- **Detection**: Pre-flight checks in DataHandler
- **Response**: Clear error messages with remediation steps
- **User Impact**: Prevents invalid analysis attempts

### Statistical Errors
- **Detection**: Try-catch in statistical methods
- **Response**: Graceful degradation or alternative methods
- **User Impact**: Suggests modifications or different approaches

### Data Quality Issues
- **Detection**: Health checks with threshold-based warnings
- **Response**: Warnings displayed prominently
- **User Impact**: Informed decision-making about result validity

## Performance Considerations

### Optimization Strategies
1. **Lazy Computation**: Results computed only when requested
2. **Caching**: Session state stores results across interactions
3. **Vectorization**: NumPy operations for large datasets
4. **Sampling**: Option to sample large datasets for EDA

### Scalability Limits
- **Maximum rows**: ~1M (limited by Streamlit memory)
- **Matching operations**: O(n²) for PSM, optimized with spatial indexing
- **Bayesian simulations**: 100K Monte Carlo samples

## Security Considerations

### Data Privacy
- **Local Processing**: All computation client-side
- **No Persistence**: Data cleared on session end
- **No External APIs**: Pure Python libraries

### Input Validation
- File type restrictions (CSV, Parquet only)
- Column type validation
- Range checks for parameters
- SQL injection prevention (not applicable, no database)

## Extension Points

### Adding New Tests
```python
# In modules/ab_testing.py
def new_test(self, control, treatment, **kwargs):
    # 1. Validate inputs
    # 2. Compute test statistic
    # 3. Calculate p-value
    # 4. Compute effect size
    # 5. Return standardized dict
    return {
        'test_type': 'New Test',
        'p_value': p_value,
        'effect': effect,
        # ... other metrics
    }
```

### Adding New Visualizations
```python
# In modules/visualizations.py
def plot_new_viz(self, data, **kwargs):
    fig = go.Figure()
    # Add traces
    # Configure layout
    return fig
```

### Adding New Export Formats
```python
# In utils/report_generator.py
def create_new_format_report(self, results, **kwargs):
    # Generate report content
    return output
```

## Testing Strategy

### Unit Tests (Recommended for Production)
```python
# tests/test_ab_testing.py
def test_t_test_known_result():
    engine = ABTestingEngine()
    control = np.array([1, 2, 3, 4, 5])
    treatment = np.array([2, 3, 4, 5, 6])
    result = engine.t_test(control, treatment)
    assert result['p_value'] < 0.05
```

### Integration Tests
- End-to-end workflow validation
- File upload → analysis → export pipeline
- Cross-validation of methods

### Statistical Validation
- Simulation studies for power analysis accuracy
- Known-result tests for statistical methods
- Robustness checks for edge cases

## Dependencies

### Core Libraries
- **Streamlit**: Web framework (1.31.0)
- **Pandas**: Data manipulation (2.1.4)
- **NumPy**: Numerical computing (1.26.3)
- **SciPy**: Statistical functions (1.11.4)
- **Statsmodels**: Advanced statistics (0.14.1)
- **Plotly**: Interactive visualizations (5.18.0)

### Specialized Libraries
- **DoWhy**: Causal inference (0.11.1) [Not currently used, placeholder]
- **CausalML**: ML-based causal methods (0.15.0) [Not currently used, placeholder]
- **PyMC**: Bayesian modeling (5.10.3)

## Design Patterns Used

1. **Facade Pattern**: ABTestingEngine, CausalInferenceLab provide simple interfaces to complex statistical operations

2. **Strategy Pattern**: Different test types (T-test, Z-test, etc.) implement common interface

3. **Template Method**: Health checks follow consistent structure

4. **Builder Pattern**: Report generation builds complex outputs step-by-step

5. **Singleton Pattern**: Session state for application-wide state management

## Future Enhancements

### Short-term
- [ ] Multi-arm bandit support
- [ ] Sequential testing (always-valid p-values)
- [ ] Regression discontinuity design
- [ ] Synthetic control methods

### Medium-term
- [ ] Automated experiment design
- [ ] Sample size optimizer with cost constraints
- [ ] Advanced heterogeneous treatment effects
- [ ] Integration with experiment platforms

### Long-term
- [ ] Real-time monitoring dashboard
- [ ] A/B test meta-analysis
- [ ] Machine learning integration for uplift modeling
- [ ] Multi-language support

---

**Last Updated**: 2025-02-13
