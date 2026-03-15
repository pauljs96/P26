"""
Pytest configuration and shared fixtures
Sistema Tesis Multi-Tenant
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables"""
    test_env = {
        'SUPABASE_URL': 'https://test.supabase.co',
        'SUPABASE_KEY': 'test-key-12345',
        'AWS_ACCESS_KEY_ID': 'test-aws-key',
        'AWS_SECRET_ACCESS_KEY': 'test-aws-secret',
        'AWS_REGION': 'us-east-1',
        'S3_BUCKET': 'test-bucket',
        'LOG_LEVEL': 'INFO',
        'ENVIRONMENT': 'test'
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    yield
    
    # Cleanup
    for key in test_env:
        os.environ.pop(key, None)


# Fixture: Mock Supabase client
@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client"""
    with patch('src.db.supabase_v2.create_client') as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


# Fixture: Mock S3 client
@pytest.fixture
def mock_s3_client():
    """Mock S3 client"""
    with patch('boto3.client') as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


# Fixture: Temporary directory for test files
@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# Fixture: Sample data
@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing"""
    import pandas as pd
    return pd.DataFrame({
        'id': range(1000),
        'value': range(1000, 2000),
        'category': ['A', 'B', 'C'] * 333 + ['A'],
        'date': pd.date_range('2020-01-01', periods=1000, freq='D')
    })


# Fixture: Mock session state
@pytest.fixture
def mock_streamlit_session():
    """Mock Streamlit session state"""
    with patch('streamlit.session_state') as mock:
        session = MagicMock()
        session.user_id = 'test_user'
        session.user_role = 'viewer'
        session.current_org = 'test_org'
        session.user_orgs = ['test_org']
        mock.__getitem__.side_effect = lambda x: getattr(session, x, None)
        mock.__setitem__.side_effect = lambda x, y: setattr(session, x, y)
        yield session


# Fixture: Test database
@pytest.fixture
def test_database():
    """Create in-memory DuckDB for testing"""
    import duckdb
    
    db = duckdb.in_memory()
    
    # Create test tables
    db.execute("""
        CREATE TABLE test_data (
            id INTEGER,
            value INTEGER,
            category VARCHAR,
            date DATE
        )
    """)
    
    # Insert sample data
    import pandas as pd
    df = pd.DataFrame({
        'id': range(100),
        'value': range(100, 200),
        'category': ['A', 'B', 'C'] * 33 + ['A'],
        'date': pd.date_range('2020-01-01', periods=100, freq='D')
    })
    
    db.register('test_data', df)
    
    yield db


# Pytest markers
def pytest_configure(config):
    """Register custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "load: mark test as a load test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as a slow test"
    )


# Hook: Report test summary
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print custom test summary"""
    terminalreporter.write_sep("=", "SISTEMA TESIS TEST SUMMARY", bold=True)
