# 🔬 Experimentation & Causal Analysis Suite

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg)](https://streamlit.io)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](Dockerfile)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> Professional A/B testing & causal inference platform with automated health checks, Bayesian analysis, and plain-English business interpretations

<p align="center">
  <img src="https://img.shields.io/badge/Statistical_Methods-8-blue" alt="8 Statistical Methods">
  <img src="https://img.shields.io/badge/Causal_Inference-3_Methods-purple" alt="3 Causal Methods">
  <img src="https://img.shields.io/badge/Health_Checks-5-orange" alt="5 Health Checks">
  <img src="https://img.shields.io/badge/Export_Formats-3-green" alt="3 Export Formats">
</p>

---

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

---

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)

```bash
# Pull and run the Docker container
docker pull your-username/experimentation-suite:latest
docker run -p 8501:8501 -p 8000:8000 experimentation-suite

# Access the app
# UI: http://localhost:8501
# API: http://localhost:8000/docs
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/your-username/experimentation-suite.git
cd experimentation-suite

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Or run the API server
python api_server.py
```

### Option 3: Quick Demo (No Installation)

```bash
# Run the standalone demo
python demo.py
```

---

## 📊 Usage

### Web Interface (Streamlit)

1. Upload your CSV/Parquet dataset
2. Select analysis type (A/B Test or Causal Inference)
3. Configure parameters
4. Review automated health checks
5. Interpret results with plain English explanations
6. Export reports in your preferred format

### API (REST)

```bash
# Run the API server
python api_server.py

# Example: Run T-test via API
curl -X POST "http://localhost:8000/api/ab-test/t-test" \
  -H "Content-Type: application/json" \
  -d '{
    "control": [98, 102, 95, 105, 99],
    "treatment": [110, 115, 108, 112, 109],
    "alpha": 0.05
  }'
```

**API Documentation**: Visit `http://localhost:8000/docs` for interactive Swagger docs

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=modules --cov=utils --cov-report=html

# Run specific test file
pytest tests/test_ab_testing.py -v

# Run tests with markers
pytest -m "unit"  # Only unit tests
pytest -m "integration"  # Only integration tests
```

**Test Coverage**: See `htmlcov/index.html` after running coverage

---

## 🐳 Docker Usage

### Build Image

```bash
docker build -t experimentation-suite .
```

### Run Container

```bash
# Run Streamlit UI only
docker run -p 8501:8501 experimentation-suite

# Run both UI and API
docker run -p 8501:8501 -p 8000:8000 experimentation-suite

# Run with volume mount for data persistence
docker run -p 8501:8501 -v $(pwd)/data:/app/data experimentation-suite
```

### Docker Compose

```bash
# Start all services
docker-compose up

# Start in detached mode
docker-compose up -d

# Stop services
docker-compose down
```

---

## 📖 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical design details
- **[API_DOCS.md](API_DOCS.md)** - REST API documentation
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

---

## 🎯 Example Use Cases

### 1. A/B Test Analysis
```python
from modules.ab_testing import ABTestingEngine

engine = ABTestingEngine()
results = engine.t_test(
    control=[100, 102, 98, 105],
    treatment=[110, 115, 108, 112],
    alpha=0.05
)
print(f"P-value: {results['p_value']:.4f}")
print(f"Significant: {results['significant']}")
```

### 2. Propensity Score Matching
```python
from modules.causal_inference import CausalInferenceLab

lab = CausalInferenceLab()
results = lab.propensity_score_matching(
    df=data,
    treatment_col='received_treatment',
    outcome_col='revenue',
    covariate_cols=['age', 'tenure', 'region']
)
print(f"ATT: {results['att']:.2f}")
```

### 3. Via REST API
```bash
curl -X POST http://localhost:8000/api/ab-test/bayesian \
  -H "Content-Type: application/json" \
  -d '{
    "control_success": 50,
    "control_total": 1000,
    "treatment_success": 65,
    "treatment_total": 1000
  }'
```

---

## 📁 Project Structure

```
experimentation-suite/
├── app.py                      # Main Streamlit application
├── api_server.py               # FastAPI REST server
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── requirements.txt            # Python dependencies
├── requirements-dev.txt        # Development dependencies
├── pytest.ini                  # Pytest configuration
├── .dockerignore              # Docker ignore file
├── README.md                   # This file
│
├── modules/                    # Core business logic
│   ├── __init__.py
│   ├── data_handler.py        # Data loading & validation
│   ├── ab_testing.py          # Statistical tests
│   ├── causal_inference.py    # PSM, DiD, IV methods
│   ├── health_checks.py       # Quality assurance
│   └── visualizations.py      # Plotly charts
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── interpreters.py        # Plain English explanations
│   └── report_generator.py    # Export functionality
│
├── tests/                      # Unit & integration tests
│   ├── __init__.py
│   ├── test_ab_testing.py
│   ├── test_causal_inference.py
│   ├── test_health_checks.py
│   ├── test_api.py
│   └── conftest.py            # Pytest fixtures
│
└── data/                       # Sample datasets
    ├── sample_ab_test_data.csv
    ├── sample_did_data.csv
    └── ecommerce_ab_test.csv
```

---

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Streamlit Configuration
STREAMLIT_PORT=8501
STREAMLIT_THEME=light

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log
```

### Docker Environment

Create a `.env` file:
```env
PYTHONUNBUFFERED=1
API_PORT=8000
STREAMLIT_PORT=8501
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Run tests (`pytest`)
4. Commit your changes (`git commit -m 'Add AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

---

## 📊 Key Design Principles

1. **Statistical Rigor**: Proper hypothesis testing, multiple comparison corrections, and effect size reporting
2. **Business Clarity**: Plain English interpretations accessible to non-statisticians
3. **Transparency**: Full disclosure of assumptions, limitations, and diagnostics
4. **Reproducibility**: Exportable reports with complete methodology
5. **Usability**: Intuitive interface with helpful guidance and warnings

---

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

---

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

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/experimentation-suite&type=Date)](https://star-history.com/#your-username/experimentation-suite&Date)

---

## 💡 Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- Statistical computing powered by [SciPy](https://scipy.org) and [Statsmodels](https://www.statsmodels.org)
- Visualizations created with [Plotly](https://plotly.com)
- API framework by [FastAPI](https://fastapi.tiangolo.com)

---

**Built with ❤️ using Python, Streamlit, FastAPI, SciPy, Statsmodels, and Plotly**

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-running-tests">Tests</a> •
  <a href="#-docker-usage">Docker</a> •
  <a href="#-contributing">Contributing</a>
</p>
