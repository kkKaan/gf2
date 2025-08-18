"""Property-based tests for binpy using Hypothesis."""

import hypothesis.strategies as st
from hypothesis import given, settings

from binpy.core import add, multiply, rank, transpose
from binpy.generators import identity, random_sparse, zeros
from binpy.solvers import solve


@given(st.integers(min_value=1, max_value=10))
def test_identity_properties(n):
    """Test basic properties of identity matrices."""
    I = identity(n)

    # Identity matrix should have full rank
    assert rank(I) == n

    # I * I = I for binary matrices
    I_squared = multiply(I, I)
    for i in range(n):
        assert I.get_row_bitwise(i) == I_squared.get_row_bitwise(i)


@given(st.integers(min_value=1, max_value=8), st.integers(min_value=1, max_value=8))
def test_zero_matrix_properties(rows, cols):
    """Test properties of zero matrices."""
    Z = zeros(rows, cols)

    # All rows should be zero
    for i in range(rows):
        assert Z.get_row_bitwise(i) == 0


@given(
    st.integers(min_value=1, max_value=8),
    st.integers(min_value=1, max_value=8),
    st.floats(min_value=0.1, max_value=0.9),
)
@settings(max_examples=20)  # Limit for performance
def test_addition_properties(rows, cols, density):
    """Test properties of matrix addition in GF(2)."""
    A = random_sparse(rows, cols, density, seed=42)
    B = random_sparse(rows, cols, density, seed=43)

    # A + A = 0 in GF(2)
    A_plus_A = add(A, A)
    Z = zeros(rows, cols)

    for i in range(rows):
        assert A_plus_A.get_row_bitwise(i) == Z.get_row_bitwise(i)

    # Commutativity: A + B = B + A
    AB = add(A, B)
    BA = add(B, A)

    for i in range(rows):
        assert AB.get_row_bitwise(i) == BA.get_row_bitwise(i)


@given(st.integers(min_value=1, max_value=6))
def test_transpose_properties(n):
    """Test properties of matrix transpose."""
    A = random_sparse(n, n, 0.3, seed=42)

    # (A^T)^T = A
    AT = transpose(A)
    ATT = transpose(AT)

    for i in range(n):
        assert A.get_row_bitwise(i) == ATT.get_row_bitwise(i)


@given(st.integers(min_value=1, max_value=5))
def test_solve_identity_system_property(n):
    """Test solving systems with identity matrix."""
    I = identity(n)

    # Generate a random binary vector
    import random

    random.seed(42)
    b = [random.randint(0, 1) for _ in range(n)]

    # Solve I * x = b
    x = solve(I, b)

    # Solution should equal b for identity matrix
    assert x == b


@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5),
)
@settings(max_examples=10)  # Limit for performance
def test_multiplication_properties(m, n, p):
    """Test properties of matrix multiplication."""
    A = random_sparse(m, n, 0.4, seed=42)
    B = random_sparse(n, p, 0.4, seed=43)
    C = random_sparse(p, 2, 0.4, seed=44)

    # Test associativity: (A * B) * C = A * (B * C)
    AB = multiply(A, B)
    BC = multiply(B, C)
    AB_C = multiply(AB, C)
    A_BC = multiply(A, BC)

    for i in range(m):
        assert AB_C.get_row_bitwise(i) == A_BC.get_row_bitwise(i)


@given(st.integers(min_value=1, max_value=6))
def test_rank_properties(n):
    """Test rank properties."""
    # Rank of identity matrix is n
    I = identity(n)
    assert rank(I) == n

    # Rank of zero matrix is 0
    Z = zeros(n, n)
    assert rank(Z) == 0

    # Rank is invariant under transpose
    A = random_sparse(n, n, 0.3, seed=42)
    AT = transpose(A)
    assert rank(A) == rank(AT)
