"""
Integration tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_server import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test suite for API endpoints"""
    
    @pytest.mark.integration
    def test_root_endpoint(self):
        """Test root endpoint returns metadata"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    @pytest.mark.integration
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    @pytest.mark.integration
    def test_t_test_endpoint(self):
        """Test T-test API endpoint"""
        payload = {
            "control": [98, 102, 95, 105, 99, 101, 97, 103],
            "treatment": [110, 115, 108, 112, 109, 111, 107, 113],
            "alpha": 0.05,
            "alternative": "two-sided"
        }
        
        response = client.post("/api/ab-test/t-test", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "p_value" in data
        assert "control_mean" in data
        assert "treatment_mean" in data
        assert "significant" in data
        assert data["test_type"] == "t-test"
    
    @pytest.mark.integration
    def test_z_test_endpoint(self):
        """Test Z-test API endpoint"""
        # Generate large enough samples for Z-test
        import numpy as np
        np.random.seed(42)
        
        payload = {
            "control": np.random.normal(100, 15, 100).tolist(),
            "treatment": np.random.normal(105, 15, 100).tolist(),
            "alpha": 0.05,
            "alternative": "two-sided"
        }
        
        response = client.post("/api/ab-test/z-test", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "z_statistic" in data
        assert "p_value" in data
        assert data["test_type"] == "z-test"
    
    @pytest.mark.integration
    def test_chi_squared_endpoint(self):
        """Test Chi-squared API endpoint"""
        payload = {
            "control_success": 50,
            "control_total": 1000,
            "treatment_success": 65,
            "treatment_total": 1000,
            "alpha": 0.05
        }
        
        response = client.post("/api/ab-test/chi-squared", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "chi2_statistic" in data
        assert "p_value" in data
        assert "control_rate" in data
        assert "treatment_rate" in data
        assert data["test_type"] == "chi-squared"
    
    @pytest.mark.integration
    def test_bayesian_endpoint(self):
        """Test Bayesian A/B test endpoint"""
        payload = {
            "control_success": 50,
            "control_total": 1000,
            "treatment_success": 65,
            "treatment_total": 1000,
            "prior_alpha": 1.0,
            "prior_beta": 1.0
        }
        
        response = client.post("/api/ab-test/bayesian", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "prob_treatment_better" in data
        assert "expected_lift" in data
        assert 0 <= data["prob_treatment_better"] <= 1
        assert data["test_type"] == "bayesian"
    
    @pytest.mark.integration
    def test_power_analysis_endpoint(self):
        """Test power analysis endpoint"""
        payload = {
            "baseline_mean": 100.0,
            "mde": 5.0,
            "baseline_std": 20.0,
            "alpha": 0.05,
            "power": 0.80,
            "ratio": 1.0
        }
        
        response = client.post("/api/ab-test/power-analysis", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "total_sample_size" in data
        assert "n_control" in data
        assert "n_treatment" in data
        assert data["total_sample_size"] > 0
    
    @pytest.mark.integration
    def test_invalid_t_test_request(self):
        """Test T-test with invalid data"""
        payload = {
            "control": [1],  # Too few samples
            "treatment": [2],
            "alpha": 0.05
        }
        
        response = client.post("/api/ab-test/t-test", json=payload)
        # Should fail validation or return error
        assert response.status_code in [400, 422]
    
    @pytest.mark.integration
    def test_invalid_alpha_value(self):
        """Test with invalid alpha value"""
        payload = {
            "control": [98, 102, 95, 105],
            "treatment": [110, 115, 108, 112],
            "alpha": 1.5  # Invalid (should be 0-1)
        }
        
        response = client.post("/api/ab-test/t-test", json=payload)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.integration
    def test_chi_squared_zero_total(self):
        """Test chi-squared with zero total"""
        payload = {
            "control_success": 0,
            "control_total": 0,  # Invalid
            "treatment_success": 5,
            "treatment_total": 100
        }
        
        response = client.post("/api/ab-test/chi-squared", json=payload)
        assert response.status_code == 422  # Validation error


class TestAPIDocumentation:
    """Test API documentation endpoints"""
    
    @pytest.mark.integration
    def test_swagger_docs_available(self):
        """Test Swagger documentation is accessible"""
        response = client.get("/docs")
        assert response.status_code == 200
    
    @pytest.mark.integration
    def test_redoc_available(self):
        """Test ReDoc documentation is accessible"""
        response = client.get("/redoc")
        assert response.status_code == 200
    
    @pytest.mark.integration
    def test_openapi_schema(self):
        """Test OpenAPI schema is valid"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "/api/ab-test/t-test" in schema["paths"]
