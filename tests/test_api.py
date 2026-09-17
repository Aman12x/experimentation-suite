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


    @pytest.mark.integration
    def test_identical_arrays_return_valid_json(self):
        """Zero variance used to produce NaN, which is not valid JSON"""
        payload = {"control": [5, 5, 5, 5], "treatment": [5, 5, 5, 5]}
        
        response = client.post("/api/ab-test/t-test", json=payload)
        assert response.status_code == 200
        assert response.json()["p_value"] == 1.0
    
    @pytest.mark.integration
    def test_unknown_alternative_is_a_validation_error(self):
        payload = {"control": [1, 2, 3, 4], "treatment": [2, 3, 4, 5], "alternative": "bogus"}
        
        response = client.post("/api/ab-test/t-test", json=payload)
        assert response.status_code == 422
    
    @pytest.mark.integration
    def test_api_and_app_share_one_engine(self):
        from api_server import get_ab_engine
        from modules.ab_testing import ABTestingEngine
        
        assert isinstance(get_ab_engine(), ABTestingEngine)
    
    @pytest.mark.integration
    def test_t_test_defaults_to_welch(self):
        payload = {"control": [98, 102, 95, 105, 99], "treatment": [110, 115, 108, 112, 109]}
        
        assert client.post("/api/ab-test/t-test", json=payload).json()["variant"] == "welch"
        payload["equal_var"] = True
        assert client.post("/api/ab-test/t-test", json=payload).json()["variant"] == "student"


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


class TestAdvancedAndCausalEndpoints:
    """Endpoints added alongside the advanced methods"""
    
    @staticmethod
    def samples(seed=0, n=400, effect=0.0):
        import numpy as np
        rng = np.random.default_rng(seed)
        return rng.normal(100, 10, n).tolist(), rng.normal(100 + effect, 10, n).tolist()
    
    @pytest.mark.integration
    def test_proportions_endpoint(self):
        r = client.post("/api/ab-test/proportions", json={
            "control_success": 50, "control_total": 1000,
            "treatment_success": 65, "treatment_total": 1000})
        assert r.status_code == 200
        assert r.json()["relative_lift"] == pytest.approx(30.0)
        assert "lift_ci_lower" in r.json()
    
    @pytest.mark.integration
    @pytest.mark.parametrize("path", ["mann-whitney", "bootstrap"])
    def test_two_sample_endpoints(self, path):
        control, treatment = self.samples(effect=3)
        r = client.post(f"/api/ab-test/{path}", json={"control": control, "treatment": treatment})
        assert r.status_code == 200
        assert r.json()["significant"] is True
    
    @pytest.mark.integration
    def test_cuped_endpoint(self):
        import numpy as np
        rng = np.random.default_rng(1)
        pre_c, pre_t = rng.normal(100, 20, 500), rng.normal(100, 20, 500)
        r = client.post("/api/ab-test/cuped", json={
            "control": (pre_c + rng.normal(0, 5, 500)).tolist(),
            "treatment": (pre_t + rng.normal(1, 5, 500)).tolist(),
            "control_covariate": pre_c.tolist(), "treatment_covariate": pre_t.tolist()})
        assert r.status_code == 200
        assert r.json()["variance_reduction_pct"] > 80
    
    @pytest.mark.integration
    def test_ratio_metric_endpoint(self):
        r = client.post("/api/ab-test/ratio-metric", json={
            "control_numerator": [10, 40, 25, 5], "control_denominator": [1, 4, 2, 1],
            "treatment_numerator": [12, 50, 30, 8], "treatment_denominator": [1, 4, 2, 1]})
        assert r.status_code == 200
        assert r.json()["control_mean"] == pytest.approx(80 / 8)
        assert r.json()["treatment_mean"] == pytest.approx(100 / 8)
    
    @pytest.mark.integration
    def test_constant_arms_return_valid_json(self):
        """Zero variance with different means used to produce an infinite t statistic"""
        r = client.post("/api/ab-test/t-test", json={"control": [0, 0, 0], "treatment": [1, 1, 1]})
        assert r.status_code == 200
        assert r.json()["p_value"] == 0.0 and r.json()["t_statistic"] is None
    
    @pytest.mark.integration
    def test_cuped_misaligned_covariate_is_400(self):
        r = client.post("/api/ab-test/cuped", json={
            "control": [1, 2, 3], "treatment": [1, 2, 3],
            "control_covariate": [1, 2], "treatment_covariate": [1, 2, 3]})
        assert r.status_code == 400
    
    @pytest.mark.integration
    def test_multi_variant_endpoint(self):
        control, a = self.samples(seed=2, effect=0)
        _, b = self.samples(seed=3, effect=4)
        r = client.post("/api/ab-test/multi-variant", json={
            "groups": {"control": control, "a": a, "b": b}, "control_label": "control"})
        assert r.status_code == 200
        body = r.json()
        assert body["best_variant"] == "b"
        assert {c["group"] for c in body["comparisons"]} == {"a", "b"}
    
    @pytest.mark.integration
    def test_sequential_endpoint(self):
        control, treatment = self.samples(seed=4, n=2000, effect=2)
        r = client.post("/api/ab-test/sequential", json={"control": control, "treatment": treatment})
        assert r.status_code == 200
        assert len(r.json()["looks"]) == 20
    
    @pytest.mark.integration
    def test_health_check_endpoint_flags_srm(self):
        r = client.post("/api/health-check", json={
            "group": ["control"] * 1000 + ["treatment"] * 700,
            "metric": [1.0, 2.0] * 850})
        assert r.status_code == 200
        body = r.json()
        assert body["sample_ratio_mismatch"]["has_srm"] is True
        assert body["overall_health"] is False
    
    @pytest.mark.integration
    def test_health_check_length_mismatch_is_400(self):
        r = client.post("/api/health-check", json={"group": ["a", "b", "a"], "metric": [1.0, 2.0]})
        assert r.status_code == 400
    
    @pytest.mark.integration
    def test_decision_endpoint_chains_from_test_output(self):
        control, treatment = self.samples(seed=5, n=3000, effect=4)
        primary = client.post("/api/ab-test/t-test", json={"control": control, "treatment": treatment}).json()
        latency_c, latency_t = self.samples(seed=6, n=3000, effect=3)
        latency = client.post("/api/ab-test/t-test", json={"control": latency_c, "treatment": latency_t}).json()
        
        r = client.post("/api/decision", json={
            "primary": primary, "guardrails": {"latency": latency},
            "guardrail_higher_is_better": {"latency": False}})
        assert r.status_code == 200
        assert r.json()["decision"] == "DO NOT SHIP"
        assert r.json()["harmed_guardrails"] == ["latency"]
    
    @pytest.mark.integration
    def test_psm_endpoint(self):
        import numpy as np
        rng = np.random.default_rng(7)
        x = rng.normal(size=800)
        treated = rng.random(800) < 1 / (1 + np.exp(-x))
        y = 5 * treated + 3 * x + rng.normal(size=800)
        rows = [{"arm": "exposed" if t else "holdout", "x": float(a), "y": float(b)}
                for t, a, b in zip(treated, x, y)]
        
        r = client.post("/api/causal/psm", json={
            "data": rows, "treatment_col": "arm", "outcome_col": "y",
            "covariate_cols": ["x"], "treated_value": "exposed"})
        assert r.status_code == 200
        body = r.json()
        assert body["att"] == pytest.approx(5.0, abs=1.0)
        assert "matched_treated" not in body
        assert body["balance_stats"][0]["covariate"] == "x"
    
    @pytest.mark.integration
    def test_did_endpoint(self, sample_did_data):
        r = client.post("/api/causal/did", json={
            "data": sample_did_data.to_dict(orient="records"),
            "group_col": "group", "time_col": "period", "outcome_col": "outcome",
            "treatment_group": "treatment", "post_period": "post", "cluster_col": "unit_id"})
        assert r.status_code == 200
        assert r.json()["did_estimate"] == pytest.approx(20.0, abs=4)
        assert r.json()["parallel_trends_assumption"] is None
    
    @pytest.mark.integration
    def test_iv_endpoint_and_bad_column(self):
        import numpy as np
        rng = np.random.default_rng(8)
        z, u = rng.normal(size=1500), rng.normal(size=1500)
        t = 0.6 * z + u + rng.normal(size=1500)
        y = 2 * t + 3 * u + rng.normal(size=1500)
        rows = [{"y": float(a), "t": float(b), "z": float(c)} for a, b, c in zip(y, t, z)]
        
        ok = client.post("/api/causal/iv", json={
            "data": rows, "outcome_col": "y", "treatment_col": "t", "instrument_col": "z"})
        assert ok.status_code == 200
        assert ok.json()["iv_estimate"] == pytest.approx(2.0, abs=0.4)
        
        bad = client.post("/api/causal/iv", json={
            "data": rows, "outcome_col": "y", "treatment_col": "t", "instrument_col": "missing"})
        assert bad.status_code == 400
    
    @pytest.mark.integration
    def test_every_advertised_endpoint_exists(self):
        advertised = client.get("/").json()["endpoints"].values()
        paths = client.get("/openapi.json").json()["paths"]
        assert all(path in paths for path in advertised)


class TestWarehouseEndpoints:
    
    @pytest.mark.integration
    def test_query_then_analyze_round_trip(self):
        duckdb = pytest.importorskip("duckdb")
        csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "multivariant_checkout_test.csv")
        plan = client.post("/api/warehouse/query", json={
            "dialect": "duckdb", "table": "checkout", "assignment_col": "variant",
            "metric_col": "converted", "analysis": "proportion"})
        assert plan.status_code == 200
        
        con = duckdb.connect()
        con.execute(f"CREATE TABLE checkout AS SELECT * FROM read_csv_auto('{csv}')")
        rows = con.execute(plan.json()["sql"]).df().to_dict("records")
        
        out = client.post("/api/warehouse/analyze", json={
            "rows": rows, "control_label": "control", "analysis": "proportion"})
        assert out.status_code == 200
        body = out.json()
        assert {c["variant"] for c in body["comparisons"]} == {"one_page_checkout", "express_pay"}
        assert body["sample_ratio_check"]["has_srm"] is False
    
    @pytest.mark.integration
    def test_injection_attempt_is_a_400(self):
        r = client.post("/api/warehouse/query", json={
            "dialect": "postgres", "table": "t; DROP TABLE users", "assignment_col": "v", "metric_col": "m"})
        assert r.status_code == 400
