"""
Pytest configuration and fixtures for testing
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_ab_data():
    """Generate sample A/B test data"""
    np.random.seed(42)
    n = 1000
    
    data = pd.DataFrame({
        'user_id': range(1, 2*n + 1),
        'group': ['control'] * n + ['treatment'] * n,
        'metric': np.concatenate([
            np.random.normal(100, 20, n),
            np.random.normal(110, 20, n)  # 10% lift
        ]),
        'conversion': np.concatenate([
            np.random.binomial(1, 0.05, n),
            np.random.binomial(1, 0.06, n)
        ])
    })
    
    return data


@pytest.fixture
def sample_did_data():
    """Generate sample DiD data"""
    np.random.seed(123)
    n = 500
    
    data = pd.DataFrame({
        'unit_id': list(range(1, n+1)) * 2 + list(range(n+1, 2*n+1)) * 2,
        'group': ['control']*n*2 + ['treatment']*n*2,
        'period': ['pre']*n + ['post']*n + ['pre']*n + ['post']*n,
        'outcome': np.concatenate([
            np.random.normal(100, 15, n),  # Control pre
            np.random.normal(105, 15, n),  # Control post (trend)
            np.random.normal(100, 15, n),  # Treatment pre
            np.random.normal(125, 15, n)   # Treatment post (trend + effect)
        ])
    })
    
    return data


@pytest.fixture
def control_treatment_arrays():
    """Simple control and treatment arrays for testing"""
    np.random.seed(42)
    control = np.array([100, 102, 98, 105, 99, 101, 97, 103])
    treatment = np.array([110, 112, 108, 115, 109, 111, 107, 113])
    return control, treatment


@pytest.fixture
def temp_csv_file(tmp_path, sample_ab_data):
    """Create a temporary CSV file for testing"""
    csv_path = tmp_path / "test_data.csv"
    sample_ab_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions for testing without Streamlit"""
    class MockStreamlit:
        @staticmethod
        def error(msg):
            print(f"ERROR: {msg}")
        
        @staticmethod
        def warning(msg):
            print(f"WARNING: {msg}")
        
        @staticmethod
        def info(msg):
            print(f"INFO: {msg}")
        
        @staticmethod
        def success(msg):
            print(f"SUCCESS: {msg}")
    
    return MockStreamlit()
