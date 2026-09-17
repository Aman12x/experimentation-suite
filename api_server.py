"""
FastAPI REST API Server for Experimentation Suite
Provides RESTful endpoints for statistical testing and causal inference
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
import numpy as np
import pandas as pd
from io import BytesIO
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
    allow_credentials=True,
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
    group: List[str] = Field(..., description="Group assignments")
    metric: List[float] = Field(..., description="Metric values")
    expected_ratio: List[float] = Field([0.5, 0.5], description="Expected group proportions")


# =============== ENGINE ===============

from modules.ab_testing import ABTestingEngine

_ab_engine = ABTestingEngine()


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
