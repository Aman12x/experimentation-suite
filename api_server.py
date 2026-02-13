"""
FastAPI REST API Server for Experimentation Suite
Provides RESTful endpoints for statistical testing and causal inference
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from io import BytesIO
import logging

# Import our modules (without streamlit dependency)
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============== PYDANTIC MODELS ===============

class TTestRequest(BaseModel):
    """Request model for T-test"""
    control: List[float] = Field(..., description="Control group data", min_items=2)
    treatment: List[float] = Field(..., description="Treatment group data", min_items=2)
    alpha: float = Field(0.05, ge=0.01, le=0.10, description="Significance level")
    alternative: str = Field("two-sided", description="Alternative hypothesis: two-sided, greater, or less")
    
    class Config:
        schema_extra = {
            "example": {
                "control": [98, 102, 95, 105, 99, 101],
                "treatment": [110, 115, 108, 112, 109, 111],
                "alpha": 0.05,
                "alternative": "two-sided"
            }
        }


class ZTestRequest(BaseModel):
    """Request model for Z-test"""
    control: List[float] = Field(..., min_items=30)
    treatment: List[float] = Field(..., min_items=30)
    alpha: float = Field(0.05, ge=0.01, le=0.10)
    alternative: str = Field("two-sided")


class ChiSquaredRequest(BaseModel):
    """Request model for Chi-squared test"""
    control_success: int = Field(..., ge=0, description="Number of successes in control")
    control_total: int = Field(..., gt=0, description="Total observations in control")
    treatment_success: int = Field(..., ge=0, description="Number of successes in treatment")
    treatment_total: int = Field(..., gt=0, description="Total observations in treatment")
    alpha: float = Field(0.05, ge=0.01, le=0.10)
    
    class Config:
        schema_extra = {
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
        schema_extra = {
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
    group: List[str] = Field(..., description="Group assignments")
    metric: List[float] = Field(..., description="Metric values")
    expected_ratio: List[float] = Field([0.5, 0.5], description="Expected group proportions")


# =============== LAZY-LOADED MODULES ===============

_ab_engine = None
_health_checker = None
_interpreter = None

def get_ab_engine():
    """Lazy load AB testing engine"""
    global _ab_engine
    if _ab_engine is None:
        # Import here to avoid streamlit dependency at startup
        try:
            from modules_standalone.ab_testing_standalone import ABTestingEngine
            _ab_engine = ABTestingEngine()
        except ImportError:
            # Fallback to inline implementation
            _ab_engine = create_standalone_ab_engine()
    return _ab_engine


def create_standalone_ab_engine():
    """Create standalone AB engine without streamlit"""
    from scipy import stats
    import numpy as np
    
    class StandaloneABEngine:
        def t_test(self, control, treatment, alpha=0.05, alternative='two-sided'):
            control = np.array(control)
            treatment = np.array(treatment)
            
            t_stat, p_value = stats.ttest_ind(treatment, control, alternative=alternative)
            
            control_mean = np.mean(control)
            treatment_mean = np.mean(treatment)
            pooled_std = np.sqrt(
                ((len(control) - 1) * np.var(control, ddof=1) + 
                 (len(treatment) - 1) * np.var(treatment, ddof=1)) / 
                (len(control) + len(treatment) - 2)
            )
            
            cohens_d = (treatment_mean - control_mean) / pooled_std
            se = pooled_std * np.sqrt(1/len(control) + 1/len(treatment))
            df = len(control) + len(treatment) - 2
            t_critical = stats.t.ppf(1 - alpha/2, df)
            ci_lower = (treatment_mean - control_mean) - t_critical * se
            ci_upper = (treatment_mean - control_mean) + t_critical * se
            relative_lift = ((treatment_mean - control_mean) / control_mean) * 100
            
            return {
                'test_type': 't-test',
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'control_mean': float(control_mean),
                'treatment_mean': float(treatment_mean),
                'control_std': float(np.std(control, ddof=1)),
                'treatment_std': float(np.std(treatment, ddof=1)),
                'control_n': int(len(control)),
                'treatment_n': int(len(treatment)),
                'mean_difference': float(treatment_mean - control_mean),
                'cohens_d': float(cohens_d),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'relative_lift': float(relative_lift),
                'significant': bool(p_value < alpha),
                'alpha': alpha
            }
        
        def z_test(self, control, treatment, alpha=0.05, alternative='two-sided'):
            control = np.array(control)
            treatment = np.array(treatment)
            
            control_mean = np.mean(control)
            treatment_mean = np.mean(treatment)
            control_std = np.std(control, ddof=1)
            treatment_std = np.std(treatment, ddof=1)
            
            se = np.sqrt((control_std**2 / len(control)) + (treatment_std**2 / len(treatment)))
            z_stat = (treatment_mean - control_mean) / se
            
            if alternative == 'two-sided':
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            elif alternative == 'greater':
                p_value = 1 - stats.norm.cdf(z_stat)
            else:
                p_value = stats.norm.cdf(z_stat)
            
            z_critical = stats.norm.ppf(1 - alpha/2)
            ci_lower = (treatment_mean - control_mean) - z_critical * se
            ci_upper = (treatment_mean - control_mean) + z_critical * se
            
            pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)
            cohens_d = (treatment_mean - control_mean) / pooled_std
            relative_lift = ((treatment_mean - control_mean) / control_mean) * 100
            
            return {
                'test_type': 'z-test',
                'z_statistic': float(z_stat),
                'p_value': float(p_value),
                'control_mean': float(control_mean),
                'treatment_mean': float(treatment_mean),
                'control_std': float(control_std),
                'treatment_std': float(treatment_std),
                'control_n': int(len(control)),
                'treatment_n': int(len(treatment)),
                'mean_difference': float(treatment_mean - control_mean),
                'cohens_d': float(cohens_d),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'relative_lift': float(relative_lift),
                'significant': bool(p_value < alpha),
                'alpha': alpha
            }
        
        def chi_squared_test(self, control_success, control_total, treatment_success, treatment_total, alpha=0.05):
            observed = np.array([
                [treatment_success, treatment_total - treatment_success],
                [control_success, control_total - control_success]
            ])
            
            chi2, p_value, dof, expected = stats.chi2_contingency(observed)
            
            control_rate = control_success / control_total
            treatment_rate = treatment_success / treatment_total
            
            se = np.sqrt(
                (control_rate * (1 - control_rate) / control_total) +
                (treatment_rate * (1 - treatment_rate) / treatment_total)
            )
            z_critical = stats.norm.ppf(1 - alpha/2)
            diff = treatment_rate - control_rate
            ci_lower = diff - z_critical * se
            ci_upper = diff + z_critical * se
            
            relative_lift = ((treatment_rate - control_rate) / control_rate) * 100 if control_rate > 0 else 0
            
            return {
                'test_type': 'chi-squared',
                'chi2_statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'control_rate': float(control_rate),
                'treatment_rate': float(treatment_rate),
                'control_n': int(control_total),
                'treatment_n': int(treatment_total),
                'rate_difference': float(diff),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'relative_lift': float(relative_lift),
                'significant': bool(p_value < alpha),
                'alpha': alpha
            }
        
        def bayesian_ab_test(self, control_success, control_total, treatment_success, treatment_total,
                           prior_alpha=1.0, prior_beta=1.0):
            control_alpha = prior_alpha + control_success
            control_beta = prior_beta + (control_total - control_success)
            treatment_alpha = prior_alpha + treatment_success
            treatment_beta = prior_beta + (treatment_total - treatment_success)
            
            control_mean = control_alpha / (control_alpha + control_beta)
            treatment_mean = treatment_alpha / (treatment_alpha + treatment_beta)
            
            np.random.seed(42)
            n_samples = 100000
            control_samples = np.random.beta(control_alpha, control_beta, n_samples)
            treatment_samples = np.random.beta(treatment_alpha, treatment_beta, n_samples)
            
            prob_treatment_better = np.mean(treatment_samples > control_samples)
            lift_samples = (treatment_samples - control_samples) / control_samples
            expected_lift = np.mean(lift_samples) * 100
            lift_ci_lower = np.percentile(lift_samples, 2.5) * 100
            lift_ci_upper = np.percentile(lift_samples, 97.5) * 100
            
            return {
                'test_type': 'bayesian',
                'control_posterior_alpha': float(control_alpha),
                'control_posterior_beta': float(control_beta),
                'treatment_posterior_alpha': float(treatment_alpha),
                'treatment_posterior_beta': float(treatment_beta),
                'control_mean': float(control_mean),
                'treatment_mean': float(treatment_mean),
                'prob_treatment_better': float(prob_treatment_better),
                'expected_lift': float(expected_lift),
                'lift_ci_lower': float(lift_ci_lower),
                'lift_ci_upper': float(lift_ci_upper),
                'control_n': int(control_total),
                'treatment_n': int(treatment_total)
            }
        
        def calculate_sample_size(self, baseline_mean, mde, baseline_std, alpha=0.05, power=0.80, ratio=1.0):
            try:
                from statsmodels.stats.power import tt_ind_solve_power
                
                effect_size = (mde / 100) * baseline_mean
                cohens_d = effect_size / baseline_std
                
                n_control = tt_ind_solve_power(
                    effect_size=cohens_d,
                    alpha=alpha,
                    power=power,
                    ratio=ratio,
                    alternative='two-sided'
                )
                
                n_treatment = n_control * ratio
                total_n = n_control + n_treatment
                
                return {
                    'n_control': int(np.ceil(n_control)),
                    'n_treatment': int(np.ceil(n_treatment)),
                    'total_sample_size': int(np.ceil(total_n)),
                    'cohens_d': float(cohens_d),
                    'mde_absolute': float(effect_size),
                    'mde_percentage': float(mde),
                    'alpha': alpha,
                    'power': power,
                    'ratio': ratio
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Power analysis error: {str(e)}")
    
    return StandaloneABEngine()


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
            "power-analysis": "/api/ab-test/power-analysis"
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
            alternative=request.alternative
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
