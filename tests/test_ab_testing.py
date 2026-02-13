"""
Unit tests for A/B Testing Engine
"""

import pytest
import numpy as np
from scipy import stats
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.ab_testing import ABTestingEngine


class TestABTestingEngine:
    """Test suite for A/B Testing Engine"""
    
    @pytest.fixture
    def engine(self):
        """Create an ABTestingEngine instance"""
        return ABTestingEngine()
    
    # =============== T-TEST TESTS ===============
    
    @pytest.mark.unit
    def test_t_test_basic(self, engine, control_treatment_arrays):
        """Test basic t-test functionality"""
        control, treatment = control_treatment_arrays
        
        results = engine.t_test(control, treatment, alpha=0.05)
        
        assert 'test_type' in results
        assert results['test_type'] == 't-test'
        assert 'p_value' in results
        assert 'control_mean' in results
        assert 'treatment_mean' in results
        assert isinstance(results['significant'], bool)
    
    @pytest.mark.unit
    @pytest.mark.statistical
    def test_t_test_known_result(self, engine):
        """Test t-test with known significant difference"""
        control = np.array([1, 2, 3, 4, 5])
        treatment = np.array([6, 7, 8, 9, 10])
        
        results = engine.t_test(control, treatment)
        
        # Large difference should be significant
        assert results['significant'] is True
        assert results['p_value'] < 0.05
        assert results['treatment_mean'] > results['control_mean']
    
    @pytest.mark.unit
    def test_t_test_no_difference(self, engine):
        """Test t-test with no real difference"""
        np.random.seed(42)
        control = np.random.normal(100, 10, 100)
        treatment = np.random.normal(100, 10, 100)
        
        results = engine.t_test(control, treatment)
        
        # Should not be significant (most of the time)
        assert 'p_value' in results
        assert results['control_mean'] == pytest.approx(results['treatment_mean'], abs=5)
    
    @pytest.mark.unit
    def test_t_test_confidence_interval(self, engine, control_treatment_arrays):
        """Test confidence interval calculation"""
        control, treatment = control_treatment_arrays
        
        results = engine.t_test(control, treatment, alpha=0.05)
        
        assert 'ci_lower' in results
        assert 'ci_upper' in results
        assert results['ci_lower'] < results['ci_upper']
        
        # CI should contain the mean difference
        mean_diff = results['mean_difference']
        assert results['ci_lower'] <= mean_diff <= results['ci_upper']
    
    @pytest.mark.unit
    def test_t_test_effect_size(self, engine, control_treatment_arrays):
        """Test Cohen's d calculation"""
        control, treatment = control_treatment_arrays
        
        results = engine.t_test(control, treatment)
        
        assert 'cohens_d' in results
        assert isinstance(results['cohens_d'], (int, float))
        
        # For our test data, effect should be positive
        assert results['cohens_d'] > 0
    
    # =============== Z-TEST TESTS ===============
    
    @pytest.mark.unit
    def test_z_test_basic(self, engine):
        """Test basic z-test functionality"""
        np.random.seed(42)
        control = np.random.normal(100, 15, 1000)
        treatment = np.random.normal(105, 15, 1000)
        
        results = engine.z_test(control, treatment)
        
        assert results['test_type'] == 'z-test'
        assert 'z_statistic' in results
        assert 'p_value' in results
    
    @pytest.mark.unit
    @pytest.mark.statistical
    def test_z_test_large_sample(self, engine):
        """Test z-test with large sample sizes"""
        np.random.seed(42)
        # Large samples with small real difference
        control = np.random.normal(100, 20, 5000)
        treatment = np.random.normal(102, 20, 5000)  # 2% lift
        
        results = engine.z_test(control, treatment)
        
        # With large sample, should detect even small differences
        assert 'p_value' in results
        assert results['treatment_mean'] > results['control_mean']
    
    # =============== CHI-SQUARED TEST ===============
    
    @pytest.mark.unit
    def test_chi_squared_basic(self, engine):
        """Test basic chi-squared test"""
        results = engine.chi_squared_test(
            control_success=50,
            control_total=1000,
            treatment_success=65,
            treatment_total=1000,
            alpha=0.05
        )
        
        assert results['test_type'] == 'chi-squared'
        assert 'chi2_statistic' in results
        assert 'p_value' in results
        assert 'control_rate' in results
        assert 'treatment_rate' in results
    
    @pytest.mark.unit
    def test_chi_squared_proportions(self, engine):
        """Test chi-squared conversion rate calculations"""
        results = engine.chi_squared_test(
            control_success=50,
            control_total=1000,
            treatment_success=60,
            treatment_total=1000
        )
        
        assert results['control_rate'] == 0.05
        assert results['treatment_rate'] == 0.06
        assert results['relative_lift'] == pytest.approx(20.0, abs=0.1)
    
    @pytest.mark.unit
    @pytest.mark.statistical
    def test_chi_squared_significant_difference(self, engine):
        """Test chi-squared with significant difference"""
        # Large difference should be significant
        results = engine.chi_squared_test(
            control_success=50,
            control_total=1000,
            treatment_success=100,
            treatment_total=1000,
            alpha=0.05
        )
        
        assert results['significant'] is True
        assert results['p_value'] < 0.05
    
    # =============== BAYESIAN A/B TEST ===============
    
    @pytest.mark.unit
    def test_bayesian_ab_basic(self, engine):
        """Test basic Bayesian A/B test"""
        results = engine.bayesian_ab_test(
            control_success=50,
            control_total=1000,
            treatment_success=60,
            treatment_total=1000
        )
        
        assert results['test_type'] == 'bayesian'
        assert 'prob_treatment_better' in results
        assert 'expected_lift' in results
        assert 0 <= results['prob_treatment_better'] <= 1
    
    @pytest.mark.unit
    @pytest.mark.statistical
    def test_bayesian_clear_winner(self, engine):
        """Test Bayesian with clear winner"""
        # Large difference should have high probability
        results = engine.bayesian_ab_test(
            control_success=50,
            control_total=1000,
            treatment_success=150,
            treatment_total=1000
        )
        
        assert results['prob_treatment_better'] > 0.99
    
    @pytest.mark.unit
    def test_bayesian_credible_interval(self, engine):
        """Test Bayesian credible interval"""
        results = engine.bayesian_ab_test(
            control_success=50,
            control_total=1000,
            treatment_success=60,
            treatment_total=1000
        )
        
        assert 'lift_ci_lower' in results
        assert 'lift_ci_upper' in results
        assert results['lift_ci_lower'] < results['lift_ci_upper']
    
    # =============== POWER ANALYSIS ===============
    
    @pytest.mark.unit
    def test_power_analysis_basic(self, engine):
        """Test basic power analysis"""
        results = engine.calculate_sample_size(
            baseline_mean=100,
            mde=5.0,  # 5% MDE
            baseline_std=20,
            alpha=0.05,
            power=0.80
        )
        
        assert 'total_sample_size' in results
        assert 'n_control' in results
        assert 'n_treatment' in results
        assert results['total_sample_size'] > 0
    
    @pytest.mark.unit
    @pytest.mark.statistical
    def test_power_analysis_higher_power_needs_more_samples(self, engine):
        """Test that higher power requires larger sample size"""
        results_80 = engine.calculate_sample_size(
            baseline_mean=100, mde=5.0, baseline_std=20,
            alpha=0.05, power=0.80
        )
        
        results_90 = engine.calculate_sample_size(
            baseline_mean=100, mde=5.0, baseline_std=20,
            alpha=0.05, power=0.90
        )
        
        assert results_90['total_sample_size'] > results_80['total_sample_size']
    
    @pytest.mark.unit
    def test_power_analysis_smaller_effect_needs_more_samples(self, engine):
        """Test that smaller MDE requires larger sample size"""
        results_5pct = engine.calculate_sample_size(
            baseline_mean=100, mde=5.0, baseline_std=20
        )
        
        results_2pct = engine.calculate_sample_size(
            baseline_mean=100, mde=2.0, baseline_std=20
        )
        
        assert results_2pct['total_sample_size'] > results_5pct['total_sample_size']
    
    # =============== EDGE CASES ===============
    
    @pytest.mark.unit
    def test_empty_arrays(self, engine):
        """Test handling of empty arrays"""
        control = np.array([])
        treatment = np.array([1, 2, 3])
        
        with pytest.raises((ValueError, ZeroDivisionError)):
            engine.t_test(control, treatment)
    
    @pytest.mark.unit
    def test_single_value_arrays(self, engine):
        """Test handling of single-value arrays"""
        control = np.array([100])
        treatment = np.array([110])
        
        # Should not crash, but results may not be meaningful
        results = engine.t_test(control, treatment)
        assert 'p_value' in results
    
    @pytest.mark.unit
    def test_identical_arrays(self, engine):
        """Test with identical control and treatment"""
        data = np.array([100, 100, 100, 100, 100])
        
        results = engine.t_test(data, data.copy())
        
        # P-value should be 1.0 (no difference)
        assert results['p_value'] == pytest.approx(1.0, abs=0.01)
        assert results['mean_difference'] == 0.0
    
    @pytest.mark.unit
    def test_zero_successes_chi_squared(self, engine):
        """Test chi-squared with zero successes"""
        results = engine.chi_squared_test(
            control_success=0,
            control_total=1000,
            treatment_success=10,
            treatment_total=1000
        )
        
        assert results['control_rate'] == 0.0
        assert results['treatment_rate'] == 0.01


# =============== INTEGRATION TESTS ===============

class TestABTestingIntegration:
    """Integration tests for A/B testing with real-world scenarios"""
    
    @pytest.fixture
    def engine(self):
        return ABTestingEngine()
    
    @pytest.mark.integration
    def test_full_ab_test_workflow(self, engine, sample_ab_data):
        """Test complete A/B test workflow"""
        control_data = sample_ab_data[
            sample_ab_data['group'] == 'control'
        ]['metric'].values
        
        treatment_data = sample_ab_data[
            sample_ab_data['group'] == 'treatment'
        ]['metric'].values
        
        # Run t-test
        results = engine.t_test(control_data, treatment_data)
        
        # Validate comprehensive results
        assert results['significant'] is True  # Should detect 10% lift
        assert results['relative_lift'] > 5  # At least 5% lift
        assert 0 < results['p_value'] < 0.05
        assert results['cohens_d'] > 0
        assert results['ci_lower'] < results['ci_upper']
    
    @pytest.mark.integration
    @pytest.mark.statistical
    def test_multiple_test_consistency(self, engine, control_treatment_arrays):
        """Test that different tests give consistent results"""
        control, treatment = control_treatment_arrays
        
        t_results = engine.t_test(control, treatment)
        z_results = engine.z_test(control, treatment)
        
        # Both should agree on direction
        assert (t_results['significant'] == z_results['significant']) or \
               (abs(t_results['p_value'] - 0.05) < 0.01)  # Edge case near threshold
        
        # Means should be identical
        assert t_results['control_mean'] == z_results['control_mean']
        assert t_results['treatment_mean'] == z_results['treatment_mean']
