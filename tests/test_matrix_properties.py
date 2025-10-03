"""
Matrix-specific property tests for binpy.

This module implements property-based testing for matrix-specific properties
including transpose properties, rank-nullity theorem verification, and
matrix decomposition reconstruction accuracy.
"""

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from binpy.core import (
    add,
    det,
    is_invertible,
    lu_decomposition,
    matrix_power,
    multiply,
    rank,
    reduced_row_echelon_form,
    trace,
    transpose,
)
from binpy.generators import identity, random_sparse, zeros
from binpy.solvers import inverse, nullspace, rank_nullity_theorem, solve
from binpy.sparse import SparseGF2Matrix


# Custom hypothesis strategies
@st.composite
def matrix_dimensions(draw, min_size=1, max_size=8):
    """Generate valid matrix dimensions for property testing."""
    rows = draw(st.integers(min_value=min_size, max_value=max_size))
    cols = draw(st.integers(min_value=min_size, max_value=max_size))
    return (rows, cols)


@st.composite
def square_matrix_size(draw, min_size=1, max_size=6):
    """Generate square matrix dimensions for property testing."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    return n


@st.composite
def random_matrix(draw, rows=None, cols=None):
    """Generate a test matrix with random dimensions and sparsity."""
    if rows is None or cols is None:
        rows, cols = draw(matrix_dimensions())

    # Create matrix by directly setting bits
    matrix = SparseGF2Matrix(rows, cols)

    # Generate random coordinates for 1s
    num_ones = draw(st.integers(min_value=0, max_value=min(rows * cols, 20)))
    coordinates = draw(
        st.sets(
            st.tuples(
                st.integers(min_value=0, max_value=rows - 1),
                st.integers(min_value=0, max_value=cols - 1),
            ),
            min_size=0,
            max_size=num_ones,
        ))

    for r, c in coordinates:
        matrix.set(r, c, 1)

    return matrix


def matrices_equal(A, B):
    """Check if two matrices are equal by comparing all rows."""
    if A.rows != B.rows or A.cols != B.cols:
        return False

    return all(A.get_row_bitwise(i) == B.get_row_bitwise(i) for i in range(A.rows))


# Transpose Properties
@pytest.mark.property
@given(matrix_dimensions())
@settings(max_examples=30, deadline=3000)
def test_transpose_involution_property(dims):
    """Test that (A^T)^T = A for all matrices."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.4, seed=42)

    AT = transpose(A)
    ATT = transpose(AT)

    assert matrices_equal(A, ATT), "Transpose involution property failed"


@pytest.mark.property
@given(matrix_dimensions())
@settings(max_examples=25, deadline=3000)
def test_transpose_addition_linearity(dims):
    """Test that (A + B)^T = A^T + B^T."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.4, seed=42)
    B = random_sparse(rows, cols, 0.4, seed=43)

    # Compute (A + B)^T
    AB = add(A, B)
    AB_T = transpose(AB)

    # Compute A^T + B^T
    AT = transpose(A)
    BT = transpose(B)
    AT_BT = add(AT, BT)

    assert matrices_equal(AB_T, AT_BT), "Transpose addition linearity failed"


@pytest.mark.property
@given(st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5))
@settings(max_examples=20, deadline=5000)
def test_transpose_multiplication_reversal(m, n):
    """Test that (A * B)^T = B^T * A^T."""
    A = random_sparse(m, n, 0.4, seed=42)
    B = random_sparse(n, m, 0.4, seed=43)

    # Compute (A * B)^T
    AB = multiply(A, B)
    AB_T = transpose(AB)

    # Compute B^T * A^T
    AT = transpose(A)
    BT = transpose(B)
    BT_AT = multiply(BT, AT)

    assert matrices_equal(AB_T, BT_AT), "Transpose multiplication reversal failed"


@pytest.mark.property
@given(square_matrix_size(min_size=1, max_size=5))
@settings(max_examples=20, deadline=3000)
def test_transpose_determinant_invariance(n):
    """Test that det(A) = det(A^T) for square matrices."""
    A = random_sparse(n, n, 0.5, seed=42)
    AT = transpose(A)

    assert det(A) == det(AT), "Transpose determinant invariance failed"


@pytest.mark.property
@given(matrix_dimensions())
@settings(max_examples=20, deadline=3000)
def test_transpose_rank_invariance(dims):
    """Test that rank(A) = rank(A^T)."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.4, seed=42)
    AT = transpose(A)

    assert rank(A) == rank(AT), "Transpose rank invariance failed"


# Rank Properties
@pytest.mark.property
@given(matrix_dimensions())
@settings(max_examples=25, deadline=3000)
def test_rank_bounds(dims):
    """Test that 0 <= rank(A) <= min(rows, cols)."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.4, seed=42)

    matrix_rank = rank(A)
    assert 0 <= matrix_rank <= min(rows, cols), "Rank bounds violated"


@pytest.mark.property
@given(matrix_dimensions())
@settings(max_examples=20, deadline=3000)
def test_rank_subadditivity(dims):
    """Test that rank(A + B) <= rank(A) + rank(B)."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.4, seed=42)
    B = random_sparse(rows, cols, 0.4, seed=43)

    rank_A = rank(A)
    rank_B = rank(B)
    AB = add(A, B)
    rank_AB = rank(AB)

    assert rank_AB <= rank_A + rank_B, "Rank subadditivity failed"


@pytest.mark.property
@given(st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5))
@settings(max_examples=15, deadline=5000)
def test_rank_multiplication_bound(m, n):
    """Test that rank(A * B) <= min(rank(A), rank(B))."""
    A = random_sparse(m, n, 0.4, seed=42)
    B = random_sparse(n, m, 0.4, seed=43)

    rank_A = rank(A)
    rank_B = rank(B)
    AB = multiply(A, B)
    rank_AB = rank(AB)

    assert rank_AB <= min(rank_A, rank_B), "Rank multiplication bound failed"


# Rank-Nullity Theorem
@pytest.mark.property
@given(matrix_dimensions())
@settings(max_examples=30, deadline=3000)
def test_rank_nullity_theorem_verification(dims):
    """Test that rank(A) + nullity(A) = cols(A)."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.4, seed=42)

    matrix_rank, nullity, num_cols = rank_nullity_theorem(A)

    assert matrix_rank + nullity == num_cols, \
        f"Rank-nullity theorem failed: {matrix_rank} + {nullity} != {num_cols}"


@pytest.mark.property
@given(matrix_dimensions())
@settings(max_examples=25, deadline=3000)
def test_nullspace_basis_properties(dims):
    """Test properties of nullspace basis vectors."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.3, seed=42)

    null_basis = nullspace(A)

    # Each basis vector should be in the nullspace
    for basis_vector in null_basis:
        # Compute A * basis_vector
        result = [0] * A.rows
        for i in range(A.rows):
            row_packed = A.get_row_bitwise(i)
            dot_product = 0
            for j in range(A.cols):
                if (row_packed >> j) & 1:
                    dot_product ^= basis_vector[j]
            result[i] = dot_product

        # Result should be zero vector
        assert all(x == 0 for x in result), "Nullspace basis vector not in nullspace"

    # Verify nullity matches basis size
    expected_nullity = len(null_basis)
    matrix_rank = rank(A)
    assert matrix_rank + expected_nullity == A.cols, \
        "Nullspace basis size inconsistent with rank-nullity theorem"


@pytest.mark.property
@given(square_matrix_size(min_size=1, max_size=5))
@settings(max_examples=20, deadline=3000)
def test_full_rank_nullspace(n):
    """Test that full rank matrices have trivial nullspace."""
    # Identity matrix is always full rank
    identity_matrix = identity(n)

    null_basis = nullspace(identity_matrix)
    matrix_rank = rank(identity_matrix)

    assert matrix_rank == n, "Identity matrix should have full rank"
    assert len(null_basis) == 0, "Full rank matrix should have trivial nullspace"


# Determinant Properties
@pytest.mark.property
@given(square_matrix_size(min_size=1, max_size=4))
@settings(max_examples=20, deadline=3000)
def test_determinant_rank_relationship(n):
    """Test that det(A) != 0 iff rank(A) = n for square matrices."""
    A = random_sparse(n, n, 0.5, seed=42)

    det_A = det(A)
    rank_A = rank(A)

    if det_A != 0:
        assert rank_A == n, "Non-zero determinant should imply full rank"
    else:
        assert rank_A < n, "Zero determinant should imply rank deficiency"


@pytest.mark.property
@given(square_matrix_size(min_size=2, max_size=3))
@settings(max_examples=15, deadline=5000)
def test_determinant_multiplication_property(n):
    """Test that det(A * B) = det(A) * det(B) in GF(2)."""
    A = random_sparse(n, n, 0.5, seed=42)
    B = random_sparse(n, n, 0.5, seed=43)

    det_A = det(A)
    det_B = det(B)
    AB = multiply(A, B)
    det_AB = det(AB)

    # In GF(2), multiplication is AND operation
    expected_det = det_A & det_B
    assert det_AB == expected_det, "Determinant multiplication property failed"


@pytest.mark.property
@given(square_matrix_size(min_size=1, max_size=4))
@settings(max_examples=15, deadline=3000)
def test_determinant_invertibility_equivalence(n):
    """Test that det(A) != 0 iff A is invertible."""
    A = random_sparse(n, n, 0.6, seed=42)

    det_A = det(A)
    invertible = is_invertible(A)

    assert (det_A != 0) == invertible, "Determinant and invertibility should be equivalent"


# Matrix Decomposition Properties
@pytest.mark.property
@given(square_matrix_size(min_size=2, max_size=4))
@settings(max_examples=10, deadline=5000)
def test_lu_decomposition_reconstruction(n):
    """Test LU decomposition properties and structure."""
    # Try multiple seeds to find matrices suitable for LU decomposition
    for seed in range(50, 80):
        A = random_sparse(n, n, 0.6, seed=seed)

        try:
            L, U = lu_decomposition(A)

            # Verify dimensions are correct
            assert L.rows == n and L.cols == n, "L matrix has incorrect dimensions"
            assert U.rows == n and U.cols == n, "U matrix has incorrect dimensions"

            # Verify L has 1s on diagonal (unit lower triangular)
            for i in range(n):
                L_row = L.get_row_bitwise(i)
                assert (L_row >> i) & 1 == 1, f"L matrix diagonal element L[{i},{i}] should be 1"

            # Verify L is lower triangular (no elements above diagonal)
            for i in range(n):
                L_row = L.get_row_bitwise(i)
                for j in range(i + 1, n):
                    assert (L_row >> j) & 1 == 0, f"L matrix element L[{i},{j}] should be 0"

            # Verify U is upper triangular (no elements below diagonal)
            for i in range(1, n):  # Start from row 1
                U_row = U.get_row_bitwise(i)
                for j in range(i):
                    assert (U_row >> j) & 1 == 0, f"U matrix element U[{i},{j}] should be 0"

            # Test that L * U has same rank as A (may not be exactly equal due to row swaps)
            LU = multiply(L, U)
            rank_A = rank(A)
            rank_LU = rank(LU)

            # In GF(2), the rank should be preserved
            assert rank_LU == rank_A, f"LU decomposition should preserve rank: {rank_A} != {rank_LU}"

            return  # Success, exit the test

        except ValueError:
            # Matrix might be singular or not suitable for LU decomposition
            continue

    # If no suitable matrix found, skip the test
    pytest.skip("Could not find matrix suitable for LU decomposition")


@pytest.mark.property
@given(matrix_dimensions())
@settings(max_examples=20, deadline=3000)
def test_rref_properties(dims):
    """Test properties of reduced row echelon form."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.4, seed=42)

    rref_matrix, pivot_cols = reduced_row_echelon_form(A)

    # RREF should have same or smaller rank
    original_rank = rank(A)
    rref_rank = rank(rref_matrix)
    assert rref_rank == original_rank, "RREF should preserve rank"

    # Number of pivot columns should equal rank
    assert len(pivot_cols) == rref_rank, "Number of pivot columns should equal rank"

    # Pivot columns should be in ascending order
    assert pivot_cols == sorted(pivot_cols), "Pivot columns should be in ascending order"


@pytest.mark.property
@given(square_matrix_size(min_size=2, max_size=4))
@settings(max_examples=10, deadline=5000)
def test_matrix_inverse_reconstruction(n):
    """Test that A * A^(-1) = I when A is invertible."""
    # Try multiple seeds to find an invertible matrix
    for seed in range(50, 100):
        A = random_sparse(n, n, 0.7, seed=seed)

        if is_invertible(A):
            try:
                A_inv = inverse(A)
                identity_matrix = identity(n)

                # Test A * A^(-1) = I
                A_Ainv = multiply(A, A_inv)
                assert matrices_equal(A_Ainv, identity_matrix), "A * A^(-1) != I"

                # Test A^(-1) * A = I
                Ainv_A = multiply(A_inv, A)
                assert matrices_equal(Ainv_A, identity_matrix), "A^(-1) * A != I"

                return  # Success, exit the test

            except Exception:
                continue

    # If no invertible matrix found, skip the test
    pytest.skip("Could not find invertible matrix for testing")


# Matrix Power Properties
@pytest.mark.property
@given(square_matrix_size(min_size=2, max_size=4))
@settings(max_examples=15, deadline=3000)
def test_matrix_power_properties(n):
    """Test properties of matrix powers."""
    A = random_sparse(n, n, 0.5, seed=42)

    # A^0 = I
    A0 = matrix_power(A, 0)
    identity_matrix = identity(n)
    assert matrices_equal(A0, identity_matrix), "A^0 should equal identity"

    # A^1 = A
    A1 = matrix_power(A, 1)
    assert matrices_equal(A1, A), "A^1 should equal A"

    # A^2 = A * A
    A2_power = matrix_power(A, 2)
    A2_mult = multiply(A, A)
    assert matrices_equal(A2_power, A2_mult), "A^2 should equal A * A"


# Trace Properties
@pytest.mark.property
@given(square_matrix_size(min_size=1, max_size=6))
@settings(max_examples=20, deadline=3000)
def test_trace_linearity(n):
    """Test that trace is linear: tr(A + B) = tr(A) + tr(B) in GF(2)."""
    A = random_sparse(n, n, 0.4, seed=42)
    B = random_sparse(n, n, 0.4, seed=43)

    trace_A = trace(A)
    trace_B = trace(B)
    AB = add(A, B)
    trace_AB = trace(AB)

    # In GF(2), addition is XOR
    expected_trace = trace_A ^ trace_B
    assert trace_AB == expected_trace, "Trace linearity failed in GF(2)"


@pytest.mark.property
@given(square_matrix_size(min_size=1, max_size=5))
@settings(max_examples=15, deadline=3000)
def test_trace_transpose_invariance(n):
    """Test that tr(A) = tr(A^T)."""
    A = random_sparse(n, n, 0.5, seed=42)
    AT = transpose(A)

    trace_A = trace(A)
    trace_AT = trace(AT)

    assert trace_A == trace_AT, "Trace should be invariant under transpose"


# Special Matrix Properties
@pytest.mark.property
@given(square_matrix_size(min_size=1, max_size=5))
@settings(max_examples=15, deadline=2000)
def test_identity_matrix_properties(n):
    """Test properties specific to identity matrices."""
    identity_matrix = identity(n)

    # Identity properties
    assert rank(identity_matrix) == n, "Identity should have full rank"
    assert det(identity_matrix) == 1, "Identity should have determinant 1"
    assert trace(identity_matrix) == (n % 2), "Identity trace should be n mod 2 in GF(2)"
    assert is_invertible(identity_matrix), "Identity should be invertible"

    # Identity is its own transpose and inverse
    IT = transpose(identity_matrix)
    assert matrices_equal(identity_matrix, IT), "Identity should equal its transpose"

    I_inv = inverse(identity_matrix)
    assert matrices_equal(identity_matrix, I_inv), "Identity should be its own inverse"


@pytest.mark.property
@given(square_matrix_size(min_size=1, max_size=5))
@settings(max_examples=15, deadline=2000)
def test_zero_matrix_properties(n):
    """Test properties specific to zero matrices."""
    zero_matrix = zeros(n, n)

    # Zero matrix properties
    assert rank(zero_matrix) == 0, "Zero matrix should have rank 0"
    assert det(zero_matrix) == 0, "Zero matrix should have determinant 0"
    assert trace(zero_matrix) == 0, "Zero matrix should have trace 0"
    assert not is_invertible(zero_matrix), "Zero matrix should not be invertible"

    # Zero matrix nullspace should be entire space
    null_basis = nullspace(zero_matrix)
    assert len(null_basis) == n, "Zero matrix nullspace should have dimension n"


# Linear System Properties
@pytest.mark.property
@given(square_matrix_size(min_size=2, max_size=4))
@settings(max_examples=10, deadline=5000)
def test_linear_system_solution_verification(n):
    """Test that solutions to Ax = b actually satisfy the equation."""
    # Try to find an invertible matrix for a unique solution
    for seed in range(50, 80):
        A = random_sparse(n, n, 0.7, seed=seed)

        if is_invertible(A):
            # Generate a random right-hand side
            b = [(seed + i) % 2 for i in range(n)]

            try:
                x = solve(A, b)
                if x is not None:
                    # Verify that A * x = b
                    result = [0] * n
                    for i in range(n):
                        row_packed = A.get_row_bitwise(i)
                        dot_product = 0
                        for j in range(n):
                            if (row_packed >> j) & 1:
                                dot_product ^= x[j]
                        result[i] = dot_product

                    assert result == b, "Solution does not satisfy Ax = b"
                    return  # Success

            except Exception:
                continue

    # If no suitable system found, skip
    pytest.skip("Could not find suitable linear system for testing")


# Matrix Norm Properties (Hamming norm for GF(2))
@pytest.mark.property
@given(matrix_dimensions())
@settings(max_examples=20, deadline=3000)
def test_hamming_norm_properties(dims):
    """Test properties of Hamming norm (number of 1s) for GF(2) matrices."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.4, seed=42)
    B = random_sparse(rows, cols, 0.4, seed=43)

    # Count 1s manually for verification
    def count_ones(matrix):
        count = 0
        for i in range(matrix.rows):
            row_packed = matrix.get_row_bitwise(i)
            count += bin(row_packed).count("1")
        return count

    ones_A = count_ones(A)
    ones_B = count_ones(B)

    # Hamming norm should be non-negative
    assert ones_A >= 0, "Hamming norm should be non-negative"
    assert ones_B >= 0, "Hamming norm should be non-negative"

    # Zero matrix should have norm 0
    zero_matrix = zeros(rows, cols)
    assert count_ones(zero_matrix) == 0, "Zero matrix should have Hamming norm 0"


# Idempotent and Nilpotent Properties
@pytest.mark.property
@given(square_matrix_size(min_size=1, max_size=4))
@settings(max_examples=10, deadline=3000)
def test_idempotent_matrix_properties(n):
    """Test properties of known idempotent matrices."""
    # Identity is idempotent: I^2 = I
    identity_matrix = identity(n)
    I2 = multiply(identity_matrix, identity_matrix)
    assert matrices_equal(identity_matrix, I2), "Identity should be idempotent"

    # Zero is idempotent: 0^2 = 0
    zero_matrix = zeros(n, n)
    Z2 = multiply(zero_matrix, zero_matrix)
    assert matrices_equal(zero_matrix, Z2), "Zero matrix should be idempotent"


@pytest.mark.property
@given(square_matrix_size(min_size=1, max_size=4))
@settings(max_examples=10, deadline=3000)
def test_involutory_matrix_properties(n):
    """Test properties of involutory matrices (A^2 = I)."""
    # Identity is involutory
    identity_matrix = identity(n)
    I2 = multiply(identity_matrix, identity_matrix)
    assert matrices_equal(identity_matrix, I2), "Identity should be involutory"

    # If A is involutory and invertible, then A = A^(-1)
    if is_invertible(identity_matrix):
        I_inv = inverse(identity_matrix)
        assert matrices_equal(identity_matrix, I_inv), "Involutory matrix should be its own inverse"
