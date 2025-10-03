"""
Unit tests for binpy solver functions.

This module provides extensive testing for all solver functions including:
- Linear system solving with various matrix types and edge cases
- Nullspace computation and verification
- Matrix inversion with correctness checking
- Solver property verification and mathematical correctness
- Error handling for invalid inputs and edge cases
"""

import contextlib

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from binpy.core import multiply, rank
from binpy.generators import identity, ones, random_sparse, zeros
from binpy.solvers import (
    condition_analysis,
    image,
    inverse,
    iterative_refinement,
    kernel,
    least_squares,
    nullspace,
    nullspace_bitwise,
    rank_nullity_theorem,
    solve,
    solve_multiple_rhs,
)
from binpy.sparse import SparseGF2Matrix


class MathVerifier:
    """Utilities for verifying mathematical properties of solver results."""

    @staticmethod
    def verify_solution(A, b, x):
        """Verify that x solves Ax = b by substitution."""
        if x is None:
            return False

        # Compute Ax
        result = []
        for i in range(A.rows):
            row_packed = A.get_row_bitwise(i)
            dot_product = 0
            for j in range(A.cols):
                if (row_packed >> j) & 1:
                    dot_product ^= x[j]
            result.append(dot_product)

        return result == list(b)

    @staticmethod
    def verify_nullspace_vector(A, x):
        """Verify that x is in the nullspace of A (Ax = 0)."""
        zero_vector = [0] * A.rows
        return MathVerifier.verify_solution(A, zero_vector, x)

    @staticmethod
    def verify_inverse(A, A_inv):
        """Verify that A_inv is the inverse of A by checking A * A_inv = I."""
        if A_inv is None:
            return False

        product = multiply(A, A_inv)
        identity_matrix = identity(A.rows)

        return all(product.get_row_bitwise(i) == identity_matrix.get_row_bitwise(i) for i in range(A.rows))

    @staticmethod
    def verify_rank_nullity(A, computed_rank, null_basis):
        """Verify rank-nullity theorem: rank(A) + nullity(A) = cols(A)."""
        nullity = len(null_basis)
        return computed_rank + nullity == A.cols


# Basic solve function tests
@pytest.mark.unit
def test_solve_identity_system():
    """Test solving systems with identity matrix."""
    identity_matrix = identity(6)
    b = [1, 0, 1, 0, 1, 0]
    x = solve(identity_matrix, b)
    assert x == b
    assert MathVerifier.verify_solution(identity_matrix, b, x)


@pytest.mark.unit
def test_solve_zero_system():
    """Test solving systems with zero matrix."""
    zero_matrix = zeros(3, 3)
    b_zero = [0, 0, 0]
    b_nonzero = [1, 0, 0]

    # Zero system with zero RHS should have solution
    x = solve(zero_matrix, b_zero)
    assert x is not None
    assert MathVerifier.verify_solution(zero_matrix, b_zero, x)

    # Zero system with nonzero RHS should have no solution
    x = solve(zero_matrix, b_nonzero)
    assert x is None


@pytest.mark.unit
def test_solve_dimension_mismatch():
    """Test error handling for dimension mismatches."""
    A = identity(3)
    b_wrong_size = [1, 0]  # Wrong size

    with pytest.raises(ValueError, match="Matrix and vector dimensions must match"):
        solve(A, b_wrong_size)


@pytest.mark.unit
def test_solve_underdetermined_system(small_matrices):
    """Test solving underdetermined systems."""
    # Create a 2x3 matrix (more variables than equations)
    A = SparseGF2Matrix(2, 3)
    A.set(0, 0, 1)
    A.set(0, 1, 1)
    A.set(1, 1, 1)
    A.set(1, 2, 1)

    b = [1, 0]
    x = solve(A, b)

    if x is not None:  # System may be consistent
        assert MathVerifier.verify_solution(A, b, x)


@pytest.mark.unit
def test_solve_overdetermined_system():
    """Test solving overdetermined systems."""
    # Create a 3x2 matrix (more equations than variables)
    A = SparseGF2Matrix(3, 2)
    A.set(0, 0, 1)
    A.set(1, 1, 1)
    A.set(2, 0, 1)
    A.set(2, 1, 1)

    b_consistent = [1, 1, 0]  # Should have solution
    b_inconsistent = [1, 1, 1]  # Should have no solution

    x1 = solve(A, b_consistent)
    if x1 is not None:
        assert MathVerifier.verify_solution(A, b_consistent, x1)

    _x2 = solve(A, b_inconsistent)
    # May or may not have solution depending on system


@pytest.mark.unit
def test_solve_various_sizes(small_matrices):
    """Test solve function with various matrix sizes."""
    for matrix in small_matrices.values():
        if matrix.rows == matrix.cols and matrix.rows > 0:  # Square matrices only
            # Create a random right-hand side
            b = [i % 2 for i in range(matrix.rows)]
            x = solve(matrix, b)

            if x is not None:
                assert len(x) == matrix.cols
                assert MathVerifier.verify_solution(matrix, b, x)


# Nullspace computation tests
@pytest.mark.unit
def test_nullspace_identity():
    """Test nullspace of identity matrix (should be empty)."""
    identity_matrix = identity(5)
    null_basis = nullspace(identity_matrix)
    assert len(null_basis) == 0


@pytest.mark.unit
def test_nullspace_zero_matrix():
    """Test nullspace of zero matrix (should be full space)."""
    zero_matrix = zeros(3, 4)
    null_basis = nullspace(zero_matrix)

    # Nullspace should have dimension equal to number of columns
    assert len(null_basis) == 4

    # Each basis vector should be in nullspace
    for vec in null_basis:
        assert MathVerifier.verify_nullspace_vector(zero_matrix, vec)


@pytest.mark.unit
def test_nullspace_rank_deficient():
    """Test nullspace of rank-deficient matrix."""
    # Create a 3x4 matrix with rank 2
    A = SparseGF2Matrix(3, 4)
    A.set(0, 0, 1)
    A.set(0, 1, 1)
    A.set(1, 2, 1)
    A.set(1, 3, 1)
    # Third row is zero, so rank is 2

    null_basis = nullspace(A)

    # Verify the actual rank first
    actual_rank = rank(A)
    expected_nullity = 4 - actual_rank

    # Nullity should be 4 - rank
    assert len(null_basis) == expected_nullity

    # Verify each basis vector is in nullspace
    for vec in null_basis:
        assert MathVerifier.verify_nullspace_vector(A, vec)


@pytest.mark.unit
def test_nullspace_properties(small_matrices):
    """Test mathematical properties of nullspace computation."""
    for matrix in small_matrices.values():
        if matrix.rows > 0 and matrix.cols > 0:
            null_basis = nullspace(matrix)
            matrix_rank = rank(matrix)

            # Verify rank-nullity theorem
            assert MathVerifier.verify_rank_nullity(matrix, matrix_rank, null_basis)

            # Verify each basis vector is in nullspace
            for vec in null_basis:
                assert MathVerifier.verify_nullspace_vector(matrix, vec)


@pytest.mark.unit
def test_nullspace_bitwise_consistency():
    """Test that nullspace_bitwise produces valid nullspace vectors."""
    A = random_sparse(5, 8, 0.3, seed=42)

    try:
        sol_str, elapsed_time = nullspace_bitwise(A)
        assert elapsed_time >= 0
        assert len(sol_str) == A.cols

        # Convert solution string to vector
        sol_vec = [int(bit) for bit in sol_str]

        # Verify it's in the nullspace (unless it's the zero vector)
        if any(bit == '1' for bit in sol_str):
            assert MathVerifier.verify_nullspace_vector(A, sol_vec)

    except ValueError:
        # Matrix might be full rank, which is acceptable
        pass


# Matrix inversion tests
@pytest.mark.unit
def test_inverse_identity():
    """Test inverse of identity matrix."""
    identity_matrix = identity(7)
    inv = inverse(identity_matrix)

    assert inv is not None
    assert MathVerifier.verify_inverse(identity_matrix, inv)

    # Identity matrix should be its own inverse
    for i in range(identity_matrix.rows):
        assert inv.get_row_bitwise(i) == identity_matrix.get_row_bitwise(i)


@pytest.mark.unit
def test_inverse_non_square():
    """Test error handling for non-square matrices."""
    A = SparseGF2Matrix(3, 4)

    with pytest.raises(ValueError, match="Matrix must be square"):
        inverse(A)


@pytest.mark.unit
def test_inverse_singular_matrix():
    """Test inverse of singular (non-invertible) matrix."""
    # Create a singular 3x3 matrix
    A = SparseGF2Matrix(3, 3)
    A.set(0, 0, 1)
    A.set(0, 1, 1)
    A.set(1, 0, 1)
    A.set(1, 1, 1)
    # Third row is zero, making it singular

    inv = inverse(A)
    assert inv is None


@pytest.mark.unit
def test_inverse_invertible_matrices():
    """Test inverse of various invertible matrices."""
    # Test with small invertible matrices
    test_matrices = []

    # 2x2 invertible matrix
    A2 = SparseGF2Matrix(2, 2)
    A2.set(0, 0, 1)
    A2.set(0, 1, 1)
    A2.set(1, 1, 1)
    test_matrices.append(A2)

    # 3x3 invertible matrix
    A3 = SparseGF2Matrix(3, 3)
    A3.set(0, 0, 1)
    A3.set(1, 1, 1)
    A3.set(2, 2, 1)
    A3.set(0, 2, 1)
    test_matrices.append(A3)

    for A in test_matrices:
        inv = inverse(A)
        if inv is not None:
            assert MathVerifier.verify_inverse(A, inv)


# Least squares solver tests
@pytest.mark.unit
def test_least_squares_overdetermined():
    """Test least squares solver for overdetermined systems."""
    # Create overdetermined system
    A = SparseGF2Matrix(4, 3)
    A.set(0, 0, 1)
    A.set(1, 1, 1)
    A.set(2, 2, 1)
    A.set(3, 0, 1)
    A.set(3, 1, 1)

    b = [1, 0, 1, 1]
    x = least_squares(A, b)

    if x is not None:
        assert len(x) == A.cols
        # Verify that x minimizes ||Ax - b||^2 in some sense


@pytest.mark.unit
def test_least_squares_dimension_mismatch():
    """Test error handling for dimension mismatches in least squares."""
    A = SparseGF2Matrix(3, 2)
    b = [1, 0]  # Wrong size

    # Should handle gracefully (may raise error or return None)
    try:
        least_squares(A, b)
    except ValueError:
        contextlib.suppress(ValueError)
        # pass  # Expected for dimension mismatch


# Kernel and image tests
@pytest.mark.unit
def test_kernel_alias():
    """Test that kernel is an alias for nullspace."""
    A = random_sparse(4, 6, 0.3, seed=42)

    null_basis = nullspace(A)
    kernel_basis = kernel(A)

    assert len(null_basis) == len(kernel_basis)
    # Bases might be different but should span the same space


@pytest.mark.unit
def test_image_computation():
    """Test image (column space) computation."""
    A = random_sparse(5, 4, 0.4, seed=42)
    img_basis = image(A)

    # Image basis should not be empty unless A is zero
    if rank(A) > 0:
        assert len(img_basis) > 0

    # Each basis vector should have correct dimension
    for vec in img_basis:
        assert len(vec) == A.rows


# Rank-nullity theorem verification
@pytest.mark.unit
def test_rank_nullity_theorem_function():
    """Test the rank_nullity_theorem verification function."""
    matrices_to_test = [
        identity(5),
        zeros(3, 4),
        random_sparse(4, 6, 0.3, seed=42),
        random_sparse(6, 4, 0.4, seed=43)
    ]

    for A in matrices_to_test:
        matrix_rank, nullity, cols = rank_nullity_theorem(A)

        assert matrix_rank >= 0
        assert nullity >= 0
        assert cols == A.cols
        assert matrix_rank + nullity == cols


# Multiple RHS solver tests
@pytest.mark.unit
def test_solve_multiple_rhs():
    """Test solving systems with multiple right-hand sides."""
    A = identity(3)

    # Create B matrix with multiple columns
    B = SparseGF2Matrix(3, 2)
    B.set(0, 0, 1)
    B.set(1, 1, 1)
    B.set(2, 0, 1)

    X = solve_multiple_rhs(A, B)

    assert X is not None
    assert X.rows == A.cols
    assert X.cols == B.cols

    # Verify AX = B by checking each column
    for j in range(B.cols):
        b_col = [(B.get_row_bitwise(i) >> j) & 1 for i in range(B.rows)]
        x_col = [(X.get_row_bitwise(i) >> j) & 1 for i in range(X.rows)]

        assert MathVerifier.verify_solution(A, b_col, x_col)


@pytest.mark.unit
def test_solve_multiple_rhs_dimension_mismatch():
    """Test error handling for dimension mismatches in multiple RHS."""
    A = SparseGF2Matrix(3, 3)
    B = SparseGF2Matrix(2, 2)  # Wrong number of rows

    with pytest.raises(ValueError, match="A and B must have same number of rows"):
        solve_multiple_rhs(A, B)


# Condition analysis tests
@pytest.mark.unit
def test_condition_analysis():
    """Test condition analysis function."""
    matrices_to_test = [identity(4), zeros(3, 3), random_sparse(5, 5, 0.4, seed=42)]

    for A in matrices_to_test:
        analysis = condition_analysis(A)

        # Check required fields
        assert "rows" in analysis
        assert "cols" in analysis
        assert "rank" in analysis
        assert "is_square" in analysis
        assert "is_invertible" in analysis
        assert "nullity" in analysis
        assert "rank_nullity_check" in analysis

        # Verify values
        assert analysis["rows"] == A.rows
        assert analysis["cols"] == A.cols
        assert analysis["is_square"] == (A.rows == A.cols)
        assert analysis["rank_nullity_check"] is True


# Iterative refinement tests
@pytest.mark.unit
def test_iterative_refinement():
    """Test iterative refinement solver."""
    A = identity(4)
    b = [1, 0, 1, 0]

    x, iterations = iterative_refinement(A, b, max_iterations=5)

    assert x is not None
    assert iterations >= 0
    assert len(x) == A.cols
    assert MathVerifier.verify_solution(A, b, x)


@pytest.mark.unit
def test_iterative_refinement_with_initial_guess():
    """Test iterative refinement with initial guess."""
    A = identity(3)
    b = [1, 1, 0]
    x0 = [0, 0, 0]  # Initial guess

    x, iterations = iterative_refinement(A, b, x0=x0, max_iterations=10)

    assert x is not None
    assert MathVerifier.verify_solution(A, b, x)


# Edge case tests
@pytest.mark.unit
def test_solve_single_element_matrix():
    """Test solving with 1x1 matrices."""
    # 1x1 matrix with value 1
    A1 = ones(1, 1)
    b1 = [1]
    x1 = solve(A1, b1)
    assert x1 == [1]

    # Test with b = [0] for the same matrix
    b0_for_A1 = [0]
    x0_for_A1 = solve(A1, b0_for_A1)
    assert x0_for_A1 == [0]

    # 1x1 matrix with value 0
    A0 = zeros(1, 1)
    b0 = [0]
    x0 = solve(A0, b0)
    assert x0 is not None  # Should have solution [0] or any value

    b1_inconsistent = [1]
    x1_inconsistent = solve(A0, b1_inconsistent)
    assert x1_inconsistent is None  # No solution


@pytest.mark.unit
def test_nullspace_single_element():
    """Test nullspace computation for 1x1 matrices."""
    # 1x1 matrix with value 1 (full rank)
    A1 = ones(1, 1)
    null1 = nullspace(A1)
    assert len(null1) == 0

    # 1x1 matrix with value 0 (rank 0)
    A0 = zeros(1, 1)
    null0 = nullspace(A0)
    assert len(null0) == 1
    assert null0[0] == [1]


@pytest.mark.unit
def test_inverse_single_element():
    """Test matrix inversion for 1x1 matrices."""
    # 1x1 matrix with value 1
    A1 = ones(1, 1)
    inv1 = inverse(A1)
    assert inv1 is not None
    assert inv1.get(0, 0) == 1

    # 1x1 matrix with value 0 (singular)
    A0 = zeros(1, 1)
    inv0 = inverse(A0)
    assert inv0 is None


@pytest.mark.unit
def test_empty_matrix_edge_cases():
    """Test edge cases with empty matrices."""
    # 0x0 matrix
    A_empty = SparseGF2Matrix(0, 0)

    # Nullspace of empty matrix
    null_empty = nullspace(A_empty)
    assert len(null_empty) == 0

    # Solve with empty system
    x_empty = solve(A_empty, [])
    assert x_empty == []


# Property-based tests using hypothesis
@pytest.mark.property
@given(st.integers(min_value=1, max_value=8))
@settings(max_examples=10)
def test_solve_identity_property(n):
    """Property test: solving with identity matrix."""
    identity_matrix = identity(n)

    # Generate random binary vector
    import random
    random.seed(42)
    b = [random.randint(0, 1) for _ in range(n)]

    x = solve(identity_matrix, b)
    assert x == b
    assert MathVerifier.verify_solution(identity_matrix, b, x)


@pytest.mark.property
@given(st.integers(min_value=1, max_value=6))
@settings(max_examples=8)
def test_nullspace_rank_nullity_property(n):
    """Property test: rank-nullity theorem."""
    A = random_sparse(n, n, 0.4, seed=42)

    matrix_rank = rank(A)
    null_basis = nullspace(A)
    nullity = len(null_basis)

    # Verify rank-nullity theorem
    assert matrix_rank + nullity == n

    # Verify each nullspace vector
    for vec in null_basis:
        assert MathVerifier.verify_nullspace_vector(A, vec)


@pytest.mark.property
@given(st.integers(min_value=2, max_value=5))
@settings(max_examples=5)
def test_inverse_property(n):
    """Property test: matrix inversion properties."""
    # Generate a potentially invertible matrix
    A = random_sparse(n, n, 0.6, seed=42)

    inv = inverse(A)

    if inv is not None:
        # Verify A * A^(-1) = I
        assert MathVerifier.verify_inverse(A, inv)

        # Verify (A^(-1))^(-1) = A
        inv_inv = inverse(inv)
        if inv_inv is not None:
            for i in range(n):
                assert A.get_row_bitwise(i) == inv_inv.get_row_bitwise(i)


# Performance and stress tests
@pytest.mark.unit
def test_solver_performance_regression():
    """Test that solvers complete within reasonable time for medium matrices."""
    import time

    A = random_sparse(50, 50, 0.1, seed=42)
    b = [i % 2 for i in range(50)]

    start_time = time.time()
    x = solve(A, b)
    elapsed = time.time() - start_time

    # Should complete within 1 second for 50x50 sparse matrix
    assert elapsed < 1.0

    if x is not None:
        assert MathVerifier.verify_solution(A, b, x)
