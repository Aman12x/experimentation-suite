"""
FastAPI REST API Server for Experimentation Suite
Provides RESTful endpoints for statistical testing and causal inference
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
import numpy as np
import pandas as pd
import logging

# Make the local packages importable when run as a script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Experimentation & Causal Analysis API",
    description="Professional A/B testing and causal inference REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # no cookies or auth here, and browsers reject "*" with credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============== PYDANTIC MODELS ===============

class TTestRequest(BaseModel):
    """Request model for T-test"""
    control: List[float] = Field(..., description="Control group data", min_length=2)
    treatment: List[float] = Field(..., description="Treatment group data", min_length=2)
    alpha: float = Field(0.05, ge=0.01, le=0.10, description="Significance level")
    alternative: Literal["two-sided", "greater", "less"] = Field(
        "two-sided", description="Alternative hypothesis"
    )
    equal_var: bool = Field(False, description="True = Student's pooled t-test, False = Welch's t-test")
    
    class Config:
        json_schema_extra = {
            "example": {
                "control": [98, 102, 95, 105, 99, 101],
                "treatment": [110, 115, 108, 112, 109, 111],
                "alpha": 0.05,
                "alternative": "two-sided"
            }
        }


class ZTestRequest(BaseModel):
    """Request model for Z-test"""
    control: List[float] = Field(..., min_length=30)
    treatment: List[float] = Field(..., min_length=30)
    alpha: float = Field(0.05, ge=0.01, le=0.10)
    alternative: Literal["two-sided", "greater", "less"] = Field("two-sided")


class ChiSquaredRequest(BaseModel):
    """Request model for Chi-squared test"""
    control_success: int = Field(..., ge=0, description="Number of successes in control")
    control_total: int = Field(..., gt=0, description="Total observations in control")
    treatment_success: int = Field(..., ge=0, description="Number of successes in treatment")
    treatment_total: int = Field(..., gt=0, description="Total observations in treatment")
    alpha: float = Field(0.05, ge=0.01, le=0.10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "control_success": 50,
                "control_total": 1000,
                "treatment_success": 65,
                "treatment_total": 1000,
                "alpha": 0.05
            }
        }


class BayesianRequest(BaseModel):
    """Request model for Bayesian A/B test"""
    control_success: int = Field(..., ge=0)
    control_total: int = Field(..., gt=0)
    treatment_success: int = Field(..., ge=0)
    treatment_total: int = Field(..., gt=0)
    prior_alpha: float = Field(1.0, gt=0, description="Beta prior alpha")
    prior_beta: float = Field(1.0, gt=0, description="Beta prior beta")


class PowerAnalysisRequest(BaseModel):
    """Request model for power analysis"""
    baseline_mean: float = Field(..., description="Expected mean of control group")
    mde: float = Field(..., gt=0, le=100, description="Minimum detectable effect (%)")
    baseline_std: float = Field(..., gt=0, description="Expected standard deviation")
    alpha: float = Field(0.05, ge=0.01, le=0.10)
    power: float = Field(0.80, ge=0.5, le=0.99)
    ratio: float = Field(1.0, gt=0, description="Treatment to control ratio")
    
    class Config:
        json_schema_extra = {
            "example": {
                "baseline_mean": 100.0,
                "mde": 5.0,
                "baseline_std": 20.0,
                "alpha": 0.05,
                "power": 0.80,
                "ratio": 1.0
            }
        }


class HealthCheckRequest(BaseModel):
    """Request model for health checks"""
    group: List[str] = Field(..., description="Group assignment per observation", min_length=2)
    metric: List[float] = Field(..., description="Metric value per observation", min_length=2)
    expected_ratio: Optional[Dict[str, float]] = Field(
        None, description="Expected traffic share per group label. Defaults to an equal split."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "group": ["control", "control", "treatment", "treatment"],
                "metric": [10.0, 12.0, 11.0, 13.0],
                "expected_ratio": {"control": 0.5, "treatment": 0.5}
            }
        }


class ProportionsRequest(BaseModel):
    """Request model for the two-proportion z-test"""
    control_success: int = Field(..., ge=0)
    control_total: int = Field(..., gt=0)
    treatment_success: int = Field(..., ge=0)
    treatment_total: int = Field(..., gt=0)
    alpha: float = Field(0.05, ge=0.01, le=0.10)
    alternative: Literal["two-sided", "greater", "less"] = Field("two-sided")


class TwoSampleRequest(BaseModel):
    """Request model for Mann-Whitney U and bootstrap tests"""
    control: List[float] = Field(..., min_length=2)
    treatment: List[float] = Field(..., min_length=2)
    alpha: float = Field(0.05, ge=0.01, le=0.10)


class RatioMetricRequest(BaseModel):
    """Request model for ratio metrics (one numerator and one denominator total per unit)"""
    control_numerator: List[float] = Field(..., min_length=2, description="e.g. revenue per user")
    control_denominator: List[float] = Field(..., min_length=2, description="e.g. sessions per user")
    treatment_numerator: List[float] = Field(..., min_length=2)
    treatment_denominator: List[float] = Field(..., min_length=2)
    alpha: float = Field(0.05, ge=0.01, le=0.10)


class WarehouseQueryRequest(BaseModel):
    """Request model for the aggregate SQL a warehouse should run"""
    dialect: Literal["bigquery", "snowflake", "postgres", "duckdb"]
    table: str
    assignment_col: str
    metric_col: str
    analysis: Literal["mean", "proportion", "cuped", "ratio"] = "mean"
    unit_col: Optional[str] = None
    second_col: Optional[str] = Field(None, description="Covariate for cuped, denominator for ratio")
    unit_agg: Literal["mean", "sum", "max"] = "mean"
    time_col: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class AggregatesRequest(BaseModel):
    """Request model for analysis from per-arm aggregate rows"""
    rows: List[Dict[str, Any]] = Field(..., min_length=2, description="One row per arm, as returned by the generated SQL")
    control_label: str
    analysis: Literal["mean", "proportion", "cuped", "ratio"] = "mean"
    higher_is_better: bool = True
    alpha: float = Field(0.05, ge=0.01, le=0.10)
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm"
    mde_pct: Optional[float] = Field(None, ge=0)
    expected_split: Optional[Dict[str, float]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "rows": [{"arm": "control", "n": 2010, "mean": 51.2, "var": 410.5},
                         {"arm": "treatment", "n": 1990, "mean": 53.0, "var": 422.1}],
                "control_label": "control",
                "mde_pct": 2.0
            }
        }


class CupedRequest(BaseModel):
    """Request model for CUPED variance reduction"""
    control: List[float] = Field(..., min_length=2)
    treatment: List[float] = Field(..., min_length=2)
    control_covariate: List[float] = Field(..., description="Pre-experiment covariate, aligned with control")
    treatment_covariate: List[float] = Field(..., description="Pre-experiment covariate, aligned with treatment")
    alpha: float = Field(0.05, ge=0.01, le=0.10)


class MultiVariantRequest(BaseModel):
    """Request model for multi-variant comparison"""
    groups: Dict[str, List[float]] = Field(..., description="Metric values per group label, control included")
    control_label: str = Field(..., description="Which label is the control")
    alpha: float = Field(0.05, ge=0.01, le=0.10)
    correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = Field("holm")
    
    class Config:
        json_schema_extra = {
            "example": {
                "groups": {"control": [10, 11, 9, 10], "a": [11, 12, 10, 12], "b": [13, 14, 12, 13]},
                "control_label": "control",
                "correction": "holm"
            }
        }


class SequentialRequest(BaseModel):
    """Request model for the always-valid sequential test"""
    control: List[float] = Field(..., description="Control observations in arrival order")
    treatment: List[float] = Field(..., description="Treatment observations in arrival order")
    alpha: float = Field(0.05, ge=0.01, le=0.10)
    n_looks: int = Field(20, ge=1, le=200)
    tau: Optional[float] = Field(None, gt=0, description="Mixing prior std dev; defaults to 10% of pooled std")


class PSMRequest(BaseModel):
    """Request model for propensity score matching"""
    data: List[Dict[str, Any]] = Field(..., description="Rows as records", min_length=4)
    treatment_col: str
    outcome_col: str
    covariate_cols: List[str] = Field(..., min_length=1)
    treated_value: Optional[Any] = Field(None, description="Label marking the treated group if the column is not 0/1")
    caliper: float = Field(0.1, gt=0, le=1)


class DiDRequest(BaseModel):
    """Request model for difference-in-differences"""
    data: List[Dict[str, Any]] = Field(..., description="Rows as records", min_length=4)
    group_col: str
    time_col: str
    outcome_col: str
    treatment_group: Any
    post_period: Any
    cluster_col: Optional[str] = None


class IVRequest(BaseModel):
    """Request model for instrumental variables (2SLS)"""
    data: List[Dict[str, Any]] = Field(..., description="Rows as records", min_length=4)
    outcome_col: str
    treatment_col: str
    instrument_col: str
    covariate_cols: Optional[List[str]] = None


class DecisionRequest(BaseModel):
    """Request model for the ship decision"""
    primary: Dict[str, Any] = Field(..., description="Result object from any A/B test endpoint")
    guardrails: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    health: Optional[Dict[str, Any]] = Field(None, description="Result object from /api/health-check")
    higher_is_better: bool = True
    guardrail_higher_is_better: Dict[str, bool] = Field(default_factory=dict)
    mde_pct: Optional[float] = Field(None, ge=0)


# =============== ENGINE ===============

from modules.ab_testing import ABTestingEngine
from modules.causal_inference import CausalInferenceLab
from modules.health_checks import HealthChecker
from utils.decision import ship_decision
from utils.warehouse import analyze_aggregates
from modules.sql_templates import aggregate_sql

_ab_engine = ABTestingEngine()
_causal_lab = CausalInferenceLab()


def _jsonable(value: Any) -> Any:
    """Convert numpy / pandas values into plain JSON types; drop what cannot be serialised"""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()
                if not isinstance(v, pd.DataFrame) or k in ('balance_stats', 'coefficients')}
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict(orient='records'))
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _run(name: str, fn, *args, **kwargs) -> JSONResponse:
    """Run an analysis; bad input is a 400, anything else is a real server error"""
    try:
        return JSONResponse(content=_jsonable(fn(*args, **kwargs)))
    except (ValueError, KeyError) as e:
        logger.warning(f"{name} rejected: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


def get_ab_engine() -> ABTestingEngine:
    """Shared A/B testing engine (same code path as the Streamlit app)"""
    return _ab_engine


# =============== API ENDPOINTS ===============

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Experimentation & Causal Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "t-test": "/api/ab-test/t-test",
            "z-test": "/api/ab-test/z-test",
            "chi-squared": "/api/ab-test/chi-squared",
            "bayesian": "/api/ab-test/bayesian",
            "power-analysis": "/api/ab-test/power-analysis",
            "proportions": "/api/ab-test/proportions",
            "mann-whitney": "/api/ab-test/mann-whitney",
            "bootstrap": "/api/ab-test/bootstrap",
            "cuped": "/api/ab-test/cuped",
            "ratio-metric": "/api/ab-test/ratio-metric",
            "warehouse-query": "/api/warehouse/query",
            "warehouse-analyze": "/api/warehouse/analyze",
            "multi-variant": "/api/ab-test/multi-variant",
            "sequential": "/api/ab-test/sequential",
            "health-check": "/api/health-check",
            "decision": "/api/decision",
            "psm": "/api/causal/psm",
            "did": "/api/causal/did",
            "iv": "/api/causal/iv"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "experimentation-api"}


@app.post("/api/ab-test/t-test")
async def run_t_test(request: TTestRequest):
    """
    Run independent samples t-test
    
    - **control**: Array of control group values
    - **treatment**: Array of treatment group values
    - **alpha**: Significance level (default: 0.05)
    - **alternative**: 'two-sided', 'greater', or 'less'
    """
    try:
        engine = get_ab_engine()
        results = engine.t_test(
            control=np.array(request.control),
            treatment=np.array(request.treatment),
            alpha=request.alpha,
            alternative=request.alternative,
            equal_var=request.equal_var
        )
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"T-test error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/ab-test/z-test")
async def run_z_test(request: ZTestRequest):
    """Run Z-test for large samples (n >= 30 recommended)"""
    try:
        engine = get_ab_engine()
        results = engine.z_test(
            control=np.array(request.control),
            treatment=np.array(request.treatment),
            alpha=request.alpha,
            alternative=request.alternative
        )
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Z-test error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/ab-test/chi-squared")
async def run_chi_squared(request: ChiSquaredRequest):
    """
    Run Chi-squared test for proportions
    
    Used for testing conversion rates, click-through rates, etc.
    """
    try:
        engine = get_ab_engine()
        results = engine.chi_squared_test(
            control_success=request.control_success,
            control_total=request.control_total,
            treatment_success=request.treatment_success,
            treatment_total=request.treatment_total,
            alpha=request.alpha
        )
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Chi-squared error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/ab-test/bayesian")
async def run_bayesian(request: BayesianRequest):
    """
    Run Bayesian A/B test using Beta-Binomial conjugate priors
    
    Returns probability that treatment is better than control
    """
    try:
        engine = get_ab_engine()
        results = engine.bayesian_ab_test(
            control_success=request.control_success,
            control_total=request.control_total,
            treatment_success=request.treatment_success,
            treatment_total=request.treatment_total,
            prior_alpha=request.prior_alpha,
            prior_beta=request.prior_beta
        )
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Bayesian test error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/ab-test/power-analysis")
async def run_power_analysis(request: PowerAnalysisRequest):
    """
    Calculate required sample size for A/B test
    
    Determines how many samples needed to detect desired effect size
    """
    try:
        engine = get_ab_engine()
        results = engine.calculate_sample_size(
            baseline_mean=request.baseline_mean,
            mde=request.mde,
            baseline_std=request.baseline_std,
            alpha=request.alpha,
            power=request.power,
            ratio=request.ratio
        )
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Power analysis error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/ab-test/proportions")
async def run_proportions(request: ProportionsRequest):
    """Two-proportion z-test for conversion-style metrics, with a CI on relative lift"""
    return _run("proportions", get_ab_engine().proportions_test, **request.model_dump())


@app.post("/api/ab-test/mann-whitney")
async def run_mann_whitney(request: TwoSampleRequest):
    """Mann-Whitney U test for skewed or heavy-tailed metrics"""
    return _run(
        "mann-whitney", get_ab_engine().mann_whitney_test,
        np.array(request.control), np.array(request.treatment), alpha=request.alpha
    )


@app.post("/api/ab-test/bootstrap")
async def run_bootstrap(request: TwoSampleRequest):
    """Percentile bootstrap for the difference in means and relative lift"""
    return _run(
        "bootstrap", get_ab_engine().bootstrap_test,
        np.array(request.control), np.array(request.treatment), alpha=request.alpha
    )


@app.post("/api/ab-test/ratio-metric")
async def run_ratio_metric(request: RatioMetricRequest):
    """Delta-method test for a ratio of totals, e.g. revenue per session with users randomized"""
    return _run("ratio-metric", get_ab_engine().ratio_metric_test, **request.model_dump())


@app.post("/api/ab-test/cuped")
async def run_cuped(request: CupedRequest):
    """CUPED variance reduction using a pre-experiment covariate"""
    return _run(
        "cuped", get_ab_engine().cuped_test,
        np.array(request.control), np.array(request.treatment),
        np.array(request.control_covariate), np.array(request.treatment_covariate),
        alpha=request.alpha
    )


@app.post("/api/ab-test/multi-variant")
async def run_multi_variant(request: MultiVariantRequest):
    """Every variant against control, with p-values corrected for multiple comparisons"""
    return _run(
        "multi-variant", get_ab_engine().multi_variant_test,
        {k: np.array(v) for k, v in request.groups.items()},
        request.control_label, alpha=request.alpha, correction=request.correction
    )


@app.post("/api/ab-test/sequential")
async def run_sequential(request: SequentialRequest):
    """Always-valid sequential test (mixture SPRT): safe to check at every look"""
    return _run(
        "sequential", get_ab_engine().sequential_test,
        np.array(request.control), np.array(request.treatment),
        alpha=request.alpha, tau=request.tau, n_looks=request.n_looks
    )


@app.post("/api/warehouse/query")
async def warehouse_query(request: WarehouseQueryRequest):
    """The one aggregate SQL statement to run in your warehouse; returns one row per arm"""
    return _run("warehouse-query", aggregate_sql, **request.model_dump())


@app.post("/api/warehouse/analyze")
async def warehouse_analyze(request: AggregatesRequest):
    """Tests, sample-ratio check and ship decision from the aggregate rows. No row-level data needed."""
    return _run("warehouse-analyze", analyze_aggregates, **request.model_dump())


@app.post("/api/health-check")
async def run_health_check(request: HealthCheckRequest):
    """Sample ratio mismatch, outliers, missing data, variance ratio, and normality checks"""
    if len(request.group) != len(request.metric):
        raise HTTPException(status_code=400, detail="group and metric must be the same length")
    df = pd.DataFrame({'group': request.group, 'metric': request.metric})
    return _run(
        "health-check", HealthChecker().run_all_checks,
        df, 'group', 'metric', expected_ratio=request.expected_ratio
    )


@app.post("/api/decision")
async def run_decision(request: DecisionRequest):
    """Combine primary metric, guardrails, and health checks into one ship decision"""
    return _run("decision", ship_decision, **request.model_dump())


@app.post("/api/causal/psm")
async def run_psm(request: PSMRequest):
    """Propensity score matching (1:1 nearest neighbour within a caliper)"""
    return _run(
        "psm", _causal_lab.propensity_score_matching,
        pd.DataFrame(request.data), request.treatment_col, request.outcome_col,
        request.covariate_cols, caliper=request.caliper, treated_value=request.treated_value
    )


@app.post("/api/causal/did")
async def run_did(request: DiDRequest):
    """Difference-in-differences with robust or cluster-robust standard errors"""
    return _run(
        "did", _causal_lab.difference_in_differences,
        pd.DataFrame(request.data), request.group_col, request.time_col, request.outcome_col,
        request.treatment_group, request.post_period, cluster_col=request.cluster_col
    )


@app.post("/api/causal/iv")
async def run_iv(request: IVRequest):
    """Two-stage least squares with a partial-F weak-instrument check"""
    return _run(
        "iv", _causal_lab.instrumental_variables,
        pd.DataFrame(request.data), request.outcome_col, request.treatment_col,
        request.instrument_col, request.covariate_cols
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
