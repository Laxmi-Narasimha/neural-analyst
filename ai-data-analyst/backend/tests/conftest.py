# AI Enterprise Data Analyst - Pytest Configuration
# Shared fixtures and configuration for all tests

import sys
import os
from datetime import datetime
from typing import Generator

import numpy as np
import pandas as pd
import pytest

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# =============================================================================
# Synthetic Data Generator Fixtures
# =============================================================================

@pytest.fixture(scope='session')
def synthetic_generator():
    """Session-scoped synthetic data generator."""
    from tests.synthetic_data_generator import SyntheticDataGenerator
    return SyntheticDataGenerator(seed=42)


@pytest.fixture(scope='session')
def customer_data(synthetic_generator):
    """Session-scoped customer dataset."""
    return synthetic_generator.generate_customer_data(n_customers=500)


@pytest.fixture(scope='session')
def transaction_data(synthetic_generator):
    """Session-scoped transaction dataset."""
    return synthetic_generator.generate_transaction_data(n_transactions=2000, n_customers=500)


@pytest.fixture(scope='session')
def time_series_data(synthetic_generator):
    """Session-scoped time series dataset."""
    return synthetic_generator.generate_time_series_data(n_points=365)


@pytest.fixture(scope='session')
def text_data(synthetic_generator):
    """Session-scoped text dataset."""
    return synthetic_generator.generate_text_data(n_records=200)


@pytest.fixture(scope='session')
def ml_classification_data(synthetic_generator):
    """Session-scoped ML classification dataset."""
    X, y = synthetic_generator.generate_ml_dataset(
        n_samples=500, n_features=10, task='classification'
    )
    return X, y


@pytest.fixture(scope='session')
def ml_regression_data(synthetic_generator):
    """Session-scoped ML regression dataset."""
    X, y = synthetic_generator.generate_ml_dataset(
        n_samples=500, n_features=10, task='regression'
    )
    return X, y


@pytest.fixture(scope='session')
def financial_data(synthetic_generator):
    """Session-scoped financial dataset."""
    return synthetic_generator.generate_financial_data(n_records=100)


@pytest.fixture(scope='session')
def session_data(synthetic_generator):
    """Session-scoped user session dataset."""
    return synthetic_generator.generate_user_session_data(n_events=5000, n_users=200)


@pytest.fixture(scope='session')
def inventory_data(synthetic_generator):
    """Session-scoped inventory dataset."""
    return synthetic_generator.generate_inventory_data(n_products=200)


# =============================================================================
# Edge Case Fixtures
# =============================================================================

@pytest.fixture
def empty_dataframe():
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def single_row_dataframe():
    """Single-row DataFrame for edge case testing."""
    return pd.DataFrame({
        'a': [1],
        'b': ['text'],
        'c': [datetime.now()]
    })


@pytest.fixture
def all_null_dataframe():
    """All-null DataFrame for edge case testing."""
    return pd.DataFrame({
        'col_a': [None] * 100,
        'col_b': [np.nan] * 100,
        'col_c': [None] * 100
    })


@pytest.fixture
def constant_column_dataframe():
    """DataFrame with constant column for edge case testing."""
    return pd.DataFrame({
        'constant': [42] * 100,
        'variable': np.random.normal(0, 1, 100)
    })


@pytest.fixture
def high_cardinality_dataframe():
    """DataFrame with high cardinality column."""
    return pd.DataFrame({
        'id': range(1000),
        'unique_values': [f'val_{i}' for i in range(1000)]
    })


@pytest.fixture
def mixed_types_dataframe():
    """DataFrame with mixed types in columns."""
    return pd.DataFrame({
        'mixed': [1, 'text', 3.14, None, True, datetime.now()] * 10
    })


@pytest.fixture
def extreme_values_dataframe():
    """DataFrame with extreme values."""
    return pd.DataFrame({
        'normal': np.random.normal(100, 15, 100),
        'with_extremes': [1e10, -1e10, 0, np.inf, -np.inf, np.nan] + [10] * 94
    })


# =============================================================================
# Numeric Fixtures for Statistical Tests
# =============================================================================

@pytest.fixture
def normal_distribution():
    """Normal distribution data."""
    np.random.seed(42)
    return pd.Series(np.random.normal(100, 15, 500))


@pytest.fixture
def skewed_distribution():
    """Right-skewed distribution data."""
    np.random.seed(42)
    return pd.Series(np.random.exponential(1, 500))


@pytest.fixture
def bimodal_distribution():
    """Bimodal distribution data."""
    np.random.seed(42)
    return pd.Series(np.concatenate([
        np.random.normal(20, 5, 250),
        np.random.normal(80, 5, 250)
    ]))


@pytest.fixture
def correlated_variables():
    """Highly correlated variables."""
    np.random.seed(42)
    x = np.random.normal(0, 1, 200)
    y = x * 0.9 + np.random.normal(0, 0.3, 200)
    return pd.DataFrame({'x': x, 'y': y})


@pytest.fixture
def uncorrelated_variables():
    """Uncorrelated variables."""
    np.random.seed(42)
    return pd.DataFrame({
        'x': np.random.normal(0, 1, 200),
        'y': np.random.normal(0, 1, 200)
    })


# =============================================================================
# Funnel Data Fixture
# =============================================================================

@pytest.fixture
def funnel_data():
    """Funnel conversion data."""
    n_users = 1000
    data = {
        'user_id': range(n_users),
        'step_1_home': [True] * n_users,
        'step_2_product': [True] * 800 + [False] * 200,
        'step_3_cart': [True] * 500 + [False] * 500,
        'step_4_checkout': [True] * 300 + [False] * 700,
        'step_5_purchase': [True] * 150 + [False] * 850
    }
    return pd.DataFrame(data)


# =============================================================================
# Cluster Data Fixture
# =============================================================================

@pytest.fixture
def cluster_data():
    """Well-separated cluster data."""
    np.random.seed(42)
    data = pd.DataFrame({
        'x': np.concatenate([
            np.random.normal(0, 1, 100),
            np.random.normal(5, 1, 100),
            np.random.normal(10, 1, 100)
        ]),
        'y': np.concatenate([
            np.random.normal(0, 1, 100),
            np.random.normal(5, 1, 100),
            np.random.normal(0, 1, 100)
        ])
    })
    labels = np.array([0] * 100 + [1] * 100 + [2] * 100)
    return data, labels


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "edge_case: marks tests for edge case scenarios"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add slow marker to tests with 'slow' in name
    for item in items:
        if 'slow' in item.name.lower():
            item.add_marker(pytest.mark.slow)


# =============================================================================
# Utility Functions
# =============================================================================

@pytest.fixture
def assert_dataframe_equal():
    """Fixture for comparing DataFrames."""
    def _assert_equal(df1, df2, check_dtype=True):
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
    return _assert_equal


@pytest.fixture
def clean_numeric_data():
    """Fixture to clean numeric data from mixed datasets."""
    def _clean(df, columns=None):
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        result = df[columns].copy()
        
        # Replace inf with nan
        result = result.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with all nan
        result = result.dropna(how='all')
        
        return result
    
    return _clean
