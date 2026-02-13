# 🔬 Experimentation & Causal Analysis Suite

A professional-grade statistical testing and causal inference platform built with Python and Streamlit.

## ✨ Features

### A/B Testing Engine
- **Statistical Tests**: T-tests, Z-tests, Chi-squared tests
- **Bayesian A/B Testing**: Beta-Binomial conjugate priors with posterior distributions
- **Power Analysis**: Calculate required sample sizes for desired effect detection
- **Effect Size Metrics**: Cohen's d, relative lift, confidence intervals

### Causal Inference Lab
- **Propensity Score Matching (PSM)**: Handle observational data with covariate balancing
- **Difference-in-Differences (DiD)**: Analyze intervention effects over time
- **Instrumental Variables (IV)**: Two-Stage Least Squares estimation for endogeneity
- **Balance Diagnostics**: Standardized mean differences and covariate balance checks

### Health Checks & Quality Assurance
- **Sample Ratio Mismatch (SRM) Detection**: Identify data quality issues
- **Outlier Detection**: IQR-based outlier identification
- **Missing Data Analysis**: Assess data completeness by group
- **Variance Ratio Checks**: Levene's test for equal variances
- **Normality Testing**: Shapiro-Wilk and Kolmogorov-Smirnov tests

### Explainability
- **Plain English Summaries**: Business-friendly interpretations of p-values, confidence intervals, and ATT
- **Automated Recommendations**: Clear guidance on whether to implement changes
- **Interactive Visualizations**: Publication-quality Plotly charts

### Export & Reporting
- **Multiple Formats**: Excel, Markdown, HTML
- **Comprehensive Reports**: Include statistical details, interpretations, and recommendations

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the repository**

2. **Create a virtual environment (recommended)**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📊 Usage

### Starting the Application

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Data Requirements

#### For A/B Testing:
- **CSV or Parquet format**
- **Group Column**: Categorical column indicating control/treatment assignment
- **Metric Column**: Numeric column with the metric to analyze
- Minimum 2 groups (control and treatment)

Example structure:
```
user_id,group,conversion,revenue
1,control,1,25.50
2,treatment,0,0.00
3,control,1,42.30
4,treatment,1,38.90
```

#### For Causal Inference:

**Propensity Score Matching:**
- Treatment indicator (binary)
- Outcome variable (numeric)
- Covariates for matching (numeric)

**Difference-in-Differences:**
- Group identifier (treated vs control)
- Time period indicator (pre vs post)
- Outcome variable (numeric)

**Instrumental Variables:**
- Outcome variable (numeric)
- Treatment variable (numeric/binary)
- Instrumental variable (numeric)
- Optional control variables

## 📖 Workflow

1. **Upload Data**: Use the sidebar to upload your CSV or Parquet file
2. **Explore Data**: Review summary statistics and data quality in the "Data Overview" tab
3. **Configure Analysis**: Select your test type and parameters
4. **Review Health Checks**: Examine automated data quality checks
5. **Interpret Results**: Read plain English explanations of statistical findings
6. **Export Reports**: Download results in your preferred format

## 🎯 Example Use Cases

### 1. A/B Test Analysis
```
Scenario: Testing a new website design
- Upload: user_data.csv with columns [user_id, variant, conversion_rate]
- Select: group_col='variant', metric_col='conversion_rate'
- Run: T-Test to compare variants
- Result: Get statistical significance, effect size, and business recommendation
```

### 2. Marketing Campaign Evaluation (DiD)
```
Scenario: Evaluating a regional marketing campaign
- Upload: sales_data.csv with [region, time_period, sales]
- Configure: group='region', time='time_period', outcome='sales'
- Run: DiD analysis to isolate campaign effect
- Result: Causal effect estimate with parallel trends visualization
```

### 3. Observational Study (PSM)
```
Scenario: Analyzing effect of a feature without randomization
- Upload: user_data.csv with [user_id, used_feature, outcome, age, tenure]
- Configure: treatment='used_feature', outcome='outcome', covariates=['age','tenure']
- Run: PSM to create comparable groups
- Result: ATT estimate with covariate balance diagnostics
```

## 📊 Statistical Methods

### Hypothesis Testing
- **T-Test**: Comparing means of two groups (assumes normality for small samples)
- **Z-Test**: Large sample comparison (Central Limit Theorem applies)
- **Chi-Squared**: Comparing proportions/frequencies between groups

### Bayesian Analysis
- **Prior**: Beta(1,1) uniform prior (configurable)
- **Posterior**: Beta-Binomial conjugate updates
- **Decision Metric**: Probability that treatment > control

### Effect Size
- **Cohen's d**: Standardized mean difference
  - Small: 0.2, Medium: 0.5, Large: 0.8
- **Relative Lift**: Percentage change from baseline

### Causal Inference
- **PSM**: Logistic regression propensity scores with nearest neighbor matching
- **DiD**: Fixed effects regression: Y = β₀ + β₁*Treated + β₂*Post + β₃*Treated×Post + ε
- **IV**: Two-Stage Least Squares with weak instrument diagnostics

## 🔧 Configuration

### Health Check Thresholds
- **SRM Alpha**: 0.001 (very stringent)
- **Outlier Threshold**: 3.0 × IQR
- **Missing Data Warning**: >10%
- **Variance Ratio Warning**: >4.0
- **Normality**: Sample size threshold of 30

### Power Analysis Defaults
- **Alpha**: 0.05 (5% Type I error rate)
- **Power**: 0.80 (80% power / 20% Type II error rate)
- **Ratio**: 1:1 (equal allocation)

## 📁 Project Structure

```
experimentation-suite/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── modules/
│   ├── __init__.py
│   ├── data_handler.py        # Data loading and validation
│   ├── ab_testing.py          # A/B testing engine
│   ├── causal_inference.py    # Causal inference methods
│   ├── health_checks.py       # Data quality checks
│   └── visualizations.py      # Plotly visualizations
└── utils/
    ├── __init__.py
    ├── interpreters.py        # Plain English explanations
    └── report_generator.py    # Export functionality
```

## 🎨 Key Design Principles

1. **Statistical Rigor**: Proper hypothesis testing, multiple comparison corrections, and effect size reporting
2. **Business Clarity**: Plain English interpretations accessible to non-statisticians
3. **Transparency**: Full disclosure of assumptions, limitations, and diagnostics
4. **Reproducibility**: Exportable reports with complete methodology
5. **Usability**: Intuitive interface with helpful guidance and warnings

## 🚨 Important Considerations

### When to Use Each Test:
- **T-Test**: Continuous metrics, approximately normal distribution, equal variances
- **Z-Test**: Large samples (n>30), known population variance
- **Chi-Squared**: Categorical outcomes, count data
- **Bayesian**: When you want probabilistic statements or have prior information

### Causal Inference Assumptions:
- **PSM**: No unmeasured confounders, overlap in propensity scores
- **DiD**: Parallel trends assumption, no anticipation effects
- **IV**: Instrument relevance, exclusion restriction, monotonicity

### Sample Size Matters:
- Small samples (<30): Higher risk of Type I and Type II errors
- Unbalanced samples: May reduce statistical power
- Use power analysis BEFORE running experiments

## 🤝 Contributing

This is a professional demonstration project. For production use, consider:
- Adding unit tests for all statistical functions
- Implementing Monte Carlo simulations for power analysis
- Adding support for multi-arm bandits
- Including sequential testing capabilities
- Adding more sophisticated causal inference methods (regression discontinuity, synthetic control)

## 📚 References

### Statistical Testing
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Deng, A. et al. (2017). "Continuous Monitoring of A/B Tests without Pain"

### Causal Inference
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference
- Angrist, J. & Pischke, J. (2009). Mostly Harmless Econometrics
- Imbens, G. & Rubin, D. (2015). Causal Inference for Statistics

### Bayesian Methods
- Gelman, A. et al. (2013). Bayesian Data Analysis

## 📄 License

This project is provided as-is for educational and professional demonstration purposes.

## 💡 Tips for Best Results

1. **Always check health diagnostics** before interpreting results
2. **Use power analysis** to determine sample sizes prospectively
3. **Consider practical significance** alongside statistical significance
4. **Document assumptions** and limitations in reports
5. **Validate results** with sensitivity analyses when possible

---

**Built with ❤️ using Python, Streamlit, SciPy, Statsmodels, and Plotly**
