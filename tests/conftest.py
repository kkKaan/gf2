"""
Pytest configuration and fixtures for comprehensive binpy testing.

This module provides:
- Matrix generation fixtures for various sizes and types
- Performance measurement utilities
- Custom pytest markers for test categorization
- Configuration for coverage reporting and parallel execution
"""

import random
import time

import hypothesis.strategies as st
import pytest
from hypothesis import settings

from binpy.generators import identity, ones, random_sparse, zeros
from binpy.sparse import SparseGF2Matrix

# Configure hypothesis for property-based testing
settings.register_profile("default", max_examples=50, deadline=5000)
settings.register_profile("ci", max_examples=100, deadline=10000)
settings.register_profile("dev", max_examples=20, deadline=2000)
settings.load_profile("default")


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual functions")
    config.addinivalue_line("markers", "integration: Integration tests for workflows")
    config.addinivalue_line("markers", "property: Property-based tests using hypothesis")
    config.addinivalue_line("markers", "performance: Performance and benchmarking tests")
    config.addinivalue_line("markers", "stress: Stress tests and fuzzing")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line("markers", "memory: Memory usage and efficiency tests")


# Matrix dimension strategies for hypothesis
@st.composite
def matrix_dimensions(draw, min_size=1, max_size=20):
    """Generate valid matrix dimensions."""
    rows = draw(st.integers(min_value=min_size, max_value=max_size))
    cols = draw(st.integers(min_value=min_size, max_value=max_size))
    return (rows, cols)


@st.composite
def square_matrix_size(draw, min_size=1, max_size=15):
    """Generate square matrix dimensions."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    return n


@st.composite
def sparsity_level(draw):
    """Generate realistic sparsity levels."""
    return draw(st.floats(min_value=0.01, max_value=0.99))


@st.composite
def binary_vector(draw, min_length=1, max_length=20):
    """Generate binary vectors."""
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    return draw(st.lists(st.integers(min_value=0, max_value=1), min_size=length, max_size=length))


# Small matrices for quick unit tests
@pytest.fixture
def small_matrices():
    """Generate small test matrices for quick tests."""
    return {
        'identity_3x3': identity(3),
        'identity_5x5': identity(5),
        'zero_3x3': zeros(3, 3),
        'zero_4x5': zeros(4, 5),
        'ones_2x2': ones(2, 2),
        'ones_3x4': ones(3, 4),
        'sparse_5x5_low': random_sparse(5, 5, 0.2, seed=42),
        'sparse_5x5_med': random_sparse(5, 5, 0.5, seed=43),
        'sparse_5x5_high': random_sparse(5, 5, 0.8, seed=44),
        'sparse_4x6': random_sparse(4, 6, 0.3, seed=45),
        'sparse_6x4': random_sparse(6, 4, 0.4, seed=46),
    }


# Medium matrices for integration tests
@pytest.fixture
def medium_matrices():
    """Generate medium-sized matrices for integration tests."""
    return {
        'identity_50': identity(50),
        'identity_100': identity(100),
        'zero_50x50': zeros(50, 50),
        'zero_75x100': zeros(75, 100),
        'sparse_50x50_low': random_sparse(50, 50, 0.05, seed=100),
        'sparse_50x50_med': random_sparse(50, 50, 0.3, seed=101),
        'sparse_100x50': random_sparse(100, 50, 0.1, seed=102),
        'sparse_50x100': random_sparse(50, 100, 0.15, seed=103),
        'dense_20x20': random_sparse(20, 20, 0.7, seed=104),
        'dense_30x25': random_sparse(30, 25, 0.8, seed=105),
    }


# Large matrices for performance tests
@pytest.fixture
def large_matrices():
    """Generate large matrices for performance and scalability tests."""
    return {
        'identity_500': identity(500),
        'identity_1000': identity(1000),
        'zero_500x500': zeros(500, 500),
        'zero_1000x1000': zeros(1000, 1000),
        'sparse_500x500_very_low': random_sparse(500, 500, 0.01, seed=200),
        'sparse_1000x1000_low': random_sparse(1000, 1000, 0.02, seed=201),
        'sparse_500x1000': random_sparse(500, 1000, 0.03, seed=202),
        'sparse_1000x500': random_sparse(1000, 500, 0.025, seed=203),
        'medium_dense_200x200': random_sparse(200, 200, 0.4, seed=204),
    }


# Test vectors for linear systems
@pytest.fixture
def test_vectors():
    """Generate test vectors for linear system solving."""
    random.seed(42)
    return {
        'binary_3': [random.randint(0, 1) for _ in range(3)],
        'binary_5': [random.randint(0, 1) for _ in range(5)],
        'binary_10': [random.randint(0, 1) for _ in range(10)],
        'binary_50': [random.randint(0, 1) for _ in range(50)],
        'zero_5': [0] * 5,
        'ones_5': [1] * 5,
        'alternating_10': [i % 2 for i in range(10)],
    }


# Matrix pairs for testing binary operations
@pytest.fixture
def matrix_pairs():
    """Generate pairs of compatible matrices for binary operations."""
    return {
        'same_size_3x3': (random_sparse(3, 3, 0.4, seed=300), random_sparse(3, 3, 0.5, seed=301)),
        'same_size_5x5': (random_sparse(5, 5, 0.3, seed=302), random_sparse(5, 5, 0.6, seed=303)),
        'multiplicable_3x4_4x5': (random_sparse(3, 4, 0.4, seed=304), random_sparse(4, 5, 0.5, seed=305)),
        'multiplicable_5x3_3x7': (random_sparse(5, 3, 0.3, seed=306), random_sparse(3, 7, 0.4, seed=307)),
        'identity_pairs': (identity(4), identity(4)),
        'zero_pairs': (zeros(3, 3), zeros(3, 3)),
    }


# Edge case matrices
@pytest.fixture
def edge_case_matrices():
    """Generate edge case matrices for stress testing."""
    return {
        'empty_0x0': SparseGF2Matrix(0, 0),
        'single_element_1x1_zero': zeros(1, 1),
        'single_element_1x1_one': ones(1, 1),
        'single_row_1x10': random_sparse(1, 10, 0.5, seed=400),
        'single_col_10x1': random_sparse(10, 1, 0.5, seed=401),
        'very_sparse': random_sparse(100, 100, 0.001, seed=402),
        'very_dense': random_sparse(50, 50, 0.99, seed=403),
        'all_zeros_large': zeros(100, 100),
        'all_ones_small': ones(5, 5),
    }


# Performance measurement utilities
class PerformanceContext:
    """Context manager for measuring operation performance."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.execution_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.execution_time = self.end_time - self.start_time


@pytest.fixture
def performance_context():
    """Fixture for measuring performance in tests."""
    return PerformanceContext


# Test data generators for specific scenarios
@pytest.fixture
def coding_theory_matrices():
    """Generate matrices commonly used in coding theory."""
    # Simple parity check matrix (3,1) repetition code
    parity_3_1 = SparseGF2Matrix(2, 3)
    parity_3_1.set(0, 0, 1)
    parity_3_1.set(0, 1, 1)
    parity_3_1.set(1, 1, 1)
    parity_3_1.set(1, 2, 1)

    # Simple Hamming (7,4) parity check matrix
    hamming_7_4 = SparseGF2Matrix(3, 7)
    # H = [1 1 0 1 1 0 0]
    #     [1 0 1 1 0 1 0]
    #     [0 1 1 1 0 0 1]
    positions = [(0, 0), (0, 1), (0, 3), (0, 4), (1, 0), (1, 2), (1, 3), (1, 5), (2, 1), (2, 2), (2, 3),
                 (2, 6)]
    for r, c in positions:
        hamming_7_4.set(r, c, 1)

    return {
        'parity_check_3_1': parity_3_1,
        'hamming_7_4': hamming_7_4,
        'random_ldpc_small': random_sparse(10, 20, 0.15, seed=500),
        'random_ldpc_medium': random_sparse(50, 100, 0.08, seed=501),
    }


# Quantum code matrices
@pytest.fixture
def quantum_code_matrices():
    """Generate matrices for quantum error correction testing."""
    # Simple 3-qubit repetition code stabilizers
    stabilizer_3_qubit = SparseGF2Matrix(2, 6)  # 2 stabilizers, 6 Pauli operators (X,Z for each qubit)
    # S1 = Z1 Z2, S2 = Z2 Z3
    stabilizer_3_qubit.set(0, 3, 1)  # Z1
    stabilizer_3_qubit.set(0, 4, 1)  # Z2
    stabilizer_3_qubit.set(1, 4, 1)  # Z2
    stabilizer_3_qubit.set(1, 5, 1)  # Z3

    return {
        'three_qubit_repetition': stabilizer_3_qubit,
        'random_css_x': random_sparse(5, 10, 0.3, seed=600),
        'random_css_z': random_sparse(5, 10, 0.3, seed=601),
    }


# Memory usage tracking
@pytest.fixture
def memory_tracker():
    """Fixture for tracking memory usage during tests."""
    import os

    import psutil

    class MemoryTracker:

        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.initial_memory = self.process.memory_info().rss

        def get_current_usage(self):
            """Get current memory usage in bytes."""
            return self.process.memory_info().rss

        def get_memory_increase(self):
            """Get memory increase since initialization."""
            return self.get_current_usage() - self.initial_memory

    return MemoryTracker()


# Parametrized test data
@pytest.fixture(params=[1, 2, 3, 5, 8, 10])
def small_square_size(request):
    """Parametrized fixture for small square matrix sizes."""
    return request.param


@pytest.fixture(params=[0.1, 0.3, 0.5, 0.7, 0.9])
def density_levels(request):
    """Parametrized fixture for different density levels."""
    return request.param


@pytest.fixture(params=[(3, 3), (4, 5), (5, 4), (10, 8), (8, 10)])
def matrix_dimensions_param(request):
    """Parametrized fixture for various matrix dimensions."""
    return request.param


# Cleanup utilities
@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    random.seed(42)
    yield
    # Cleanup after test if needed


# Test result collection
@pytest.fixture(scope="session")
def test_results():
    """Session-scoped fixture to collect test results."""
    results = {
        'performance_data': [],
        'memory_usage': [],
        'coverage_data': {},
        'failed_tests': [],
    }
    yield results
    # Could save results to file here if needed
