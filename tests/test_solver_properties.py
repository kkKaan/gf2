"""
Solver property verification tests using property-based testing.

This module implements comprehensive property-based testing for solver correctness
including solution verification, nullspace property verification, and matrix
inversion properties using hypothesis for automated test case generation.
"""

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from gf2.core import multiply, rank
from gf2.generators import identity, random_sparse, zeros
from gf2.solvers import inverse, nullspace, solve
from gf2.sparse import SparseGF2Matrix


# Custom hypothesis strategies for solver testing
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

    # Create matrix by directly setting bits to avoid random module
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
        )
    )

    for r, c in coordinates:
        matrix.set(r, c, 1)

    return matrix


@st.composite
def binary_vector(draw, length):
    """Generate a binary vector of specified length."""
    return draw(st.lists(st.integers(min_value=0, max_value=1), min_size=length, max_size=length))


@st.composite
def solvable_system(draw):
    """Generate a matrix and compatible right-hand side vector."""
    rows, cols = draw(matrix_dimensions(min_size=2, max_size=6))
    A = draw(random_matrix(rows, cols))
    b = draw(binary_vector(rows))
    return A, b


@st.composite
def square_matrix_and_vector(draw):
    """Generate a square matrix and compatible vector."""
    n = draw(square_matrix_size(min_size=2, max_size=6))
    A = draw(random_matrix(n, n))
    b = draw(binary_vector(n))
    return A, b


def matrices_equal(A, B):
    """Check if two matrices are equal by comparing all rows."""
    if A.rows != B.rows or A.cols != B.cols:
        return False

    return all(A.get_row_bitwise(i) == B.get_row_bitwise(i) for i in range(A.rows))


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


def verify_nullspace_vector(A, x):
    """Verify that x is in the nullspace of A (Ax = 0)."""
    zero_vector = [0] * A.rows
    return verify_solution(A, zero_vector, x)


# Solution Correctness Properties
@pytest.mark.property
@given(solvable_system())
@settings(max_examples=50, deadline=5000)
def test_solution_correctness_property(system):
    """Property test: if solve returns a solution, it must satisfy Ax = b."""
    A, b = system

    x = solve(A, b)

    if x is not None:
        # If a solution exists, it must satisfy the equation
        assert verify_solution(A, b, x), "Solution does not satisfy Ax = b"
        assert len(x) == A.cols, "Solution vector has incorrect length"


@pytest.mark.property
@given(square_matrix_size(min_size=1, max_size=5))
@settings(max_examples=30, deadline=3000)
def test_identity_system_solution_property(n):
    """Property test: solving Ix = b should always give x = b."""
    identity_matrix = identity(n)

    # Generate random right-hand side
    b = [(42 + i) % 2 for i in range(n)]  # Deterministic but varied

    x = solve(identity_matrix, b)

    assert x is not None, "Identity system should always have a solution"
    assert x == b, "Solution to identity system should equal right-hand side"
    assert verify_solution(identity_matrix, b, x), "Identity solution verification failed"


@pytest.mark.property
@given(matrix_dimensions(min_size=2, max_size=6))
@settings(max_examples=25, deadline=3000)
def test_zero_system_solution_property(dims):
    """Property test: zero system Ax = b has solution iff b = 0."""
    rows, cols = dims
    zero_matrix = zeros(rows, cols)

    # Test with zero right-hand side
    b_zero = [0] * rows
    x_zero = solve(zero_matrix, b_zero)
    assert x_zero is not None, "Zero system with zero RHS should have solution"
    assert verify_solution(zero_matrix, b_zero, x_zero), "Zero system solution verification failed"

    # Test with non-zero right-hand side (if possible)
    if rows > 0:
        b_nonzero = [1] + [0] * (rows - 1)
        x_nonzero = solve(zero_matrix, b_nonzero)
        assert x_nonzero is None, "Zero system with non-zero RHS should have no solution"


@pytest.mark.property
@given(square_matrix_and_vector())
@settings(max_examples=30, deadline=5000)
def test_solution_uniqueness_for_invertible_matrices(system):
    """Property test: invertible matrices have unique solutions."""
    A, b = system

    # Check if matrix is invertible
    A_inv = inverse(A)
    if A_inv is not None:
        # Matrix is invertible, should have unique solution
        x = solve(A, b)
        assert x is not None, "Invertible system should always have a solution"
        assert verify_solution(A, b, x), "Invertible system solution verification failed"

        # Verify solution matches A^(-1) * b
        expected_solution = []
        for i in range(A.rows):
            row_packed = A_inv.get_row_bitwise(i)
            dot_product = 0
            for j in range(len(b)):
                if (row_packed >> j) & 1:
                    dot_product ^= b[j]
            expected_solution.append(dot_product)

        assert x == expected_solution, "Solution should match A^(-1) * b for invertible matrices"


# Nullspace Property Verification
@pytest.mark.property
@given(matrix_dimensions(min_size=1, max_size=6))
@settings(max_examples=40, deadline=4000)
def test_nullspace_vectors_property(dims):
    """Property test: all vectors in nullspace basis satisfy Ax = 0."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.4, seed=42)

    null_basis = nullspace(A)

    # Each basis vector should be in the nullspace
    for basis_vector in null_basis:
        assert len(basis_vector) == cols, "Nullspace vector has incorrect length"
        assert verify_nullspace_vector(A, basis_vector), f"Basis vector {basis_vector} not in nullspace"


@pytest.mark.property
@given(matrix_dimensions(min_size=1, max_size=6))
@settings(max_examples=35, deadline=4000)
def test_rank_nullity_theorem_property(dims):
    """Property test: rank(A) + nullity(A) = cols(A) for all matrices."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.4, seed=42)

    matrix_rank = rank(A)
    null_basis = nullspace(A)
    nullity = len(null_basis)

    assert matrix_rank + nullity == cols, (
        f"Rank-nullity theorem failed: rank={matrix_rank}, nullity={nullity}, cols={cols}"
    )


@pytest.mark.property
@given(square_matrix_size(min_size=1, max_size=5))
@settings(max_examples=25, deadline=3000)
def test_full_rank_nullspace_property(n):
    """Property test: full rank matrices have trivial nullspace."""
    identity_matrix = identity(n)

    null_basis = nullspace(identity_matrix)
    matrix_rank = rank(identity_matrix)

    assert matrix_rank == n, "Identity matrix should have full rank"
    assert len(null_basis) == 0, "Full rank matrix should have trivial nullspace"


@pytest.mark.property
@given(matrix_dimensions(min_size=2, max_size=5))
@settings(max_examples=20, deadline=3000)
def test_zero_matrix_nullspace_property(dims):
    """Property test: zero matrix nullspace spans entire space."""
    rows, cols = dims
    zero_matrix = zeros(rows, cols)

    null_basis = nullspace(zero_matrix)
    matrix_rank = rank(zero_matrix)

    assert matrix_rank == 0, "Zero matrix should have rank 0"
    assert len(null_basis) == cols, "Zero matrix nullspace should have dimension equal to columns"

    # Verify each basis vector is in nullspace
    for basis_vector in null_basis:
        assert verify_nullspace_vector(zero_matrix, basis_vector), (
            "Zero matrix nullspace vector verification failed"
        )


@pytest.mark.property
@given(matrix_dimensions(min_size=2, max_size=5))
@settings(max_examples=20, deadline=4000)
def test_nullspace_basis_properties(dims):
    """Property test: nullspace basis vectors should all be in the nullspace."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.3, seed=42)

    null_basis = nullspace(A)

    # Each basis vector should be in the nullspace
    for basis_vector in null_basis:
        assert len(basis_vector) == cols, "Nullspace vector has incorrect length"
        assert verify_nullspace_vector(A, basis_vector), "Basis vector not in nullspace"

    # If we have multiple basis vectors, they should not all be the zero vector
    if len(null_basis) > 1:
        non_zero_count = sum(1 for basis_vector in null_basis if any(bit for bit in basis_vector))
        assert non_zero_count > 0, "Nullspace basis should contain at least one non-zero vector"


# Matrix Inversion Properties
@pytest.mark.property
@given(square_matrix_size(min_size=2, max_size=4))
@settings(max_examples=20, deadline=6000)
def test_matrix_inverse_property(n):
    """Property test: A * A^(-1) = I when A is invertible."""
    # Try multiple seeds to find invertible matrices
    for seed in range(50, 100):
        A = random_sparse(n, n, 0.7, seed=seed)

        A_inv = inverse(A)
        if A_inv is not None:
            identity_matrix = identity(n)

            # Test A * A^(-1) = I
            A_Ainv = multiply(A, A_inv)
            assert matrices_equal(A_Ainv, identity_matrix), "A * A^(-1) != I"

            # Test A^(-1) * A = I
            Ainv_A = multiply(A_inv, A)
            assert matrices_equal(Ainv_A, identity_matrix), "A^(-1) * A != I"

            return  # Success, exit the test

    # If no invertible matrix found, skip the test
    pytest.skip("Could not generate invertible matrix for testing")


@pytest.mark.property
@given(square_matrix_size(min_size=1, max_size=4))
@settings(max_examples=15, deadline=3000)
def test_identity_inverse_property(n):
    """Property test: identity matrix is its own inverse."""
    identity_matrix = identity(n)

    I_inv = inverse(identity_matrix)
    assert I_inv is not None, "Identity matrix should be invertible"
    assert matrices_equal(identity_matrix, I_inv), "Identity matrix should be its own inverse"


@pytest.mark.property
@given(square_matrix_size(min_size=2, max_size=4))
@settings(max_examples=15, deadline=5000)
def test_inverse_involution_property(n):
    """Property test: (A^(-1))^(-1) = A when both inverses exist."""
    # Try to find an invertible matrix
    for seed in range(50, 80):
        A = random_sparse(n, n, 0.8, seed=seed)

        A_inv = inverse(A)
        if A_inv is not None:
            # Compute inverse of inverse
            A_inv_inv = inverse(A_inv)
            if A_inv_inv is not None:
                assert matrices_equal(A, A_inv_inv), "(A^(-1))^(-1) should equal A"
                return  # Success

    pytest.skip("Could not find matrix with computable double inverse")


@pytest.mark.property
@given(square_matrix_size(min_size=1, max_size=4))
@settings(max_examples=15, deadline=3000)
def test_singular_matrix_no_inverse_property(n):
    """Property test: singular matrices have no inverse."""
    # Create a known singular matrix (zero matrix)
    zero_matrix = zeros(n, n)

    zero_inv = inverse(zero_matrix)
    assert zero_inv is None, "Singular matrix should not have inverse"

    # Create rank-deficient matrix if n > 1
    if n > 1:
        rank_deficient = SparseGF2Matrix(n, n)
        # Set only first row to have some 1s, leave other rows zero
        for j in range(min(n, 3)):
            rank_deficient.set(0, j, 1)

        rd_inv = inverse(rank_deficient)
        assert rd_inv is None, "Rank-deficient matrix should not have inverse"


# Solver Consistency Properties
@pytest.mark.property
@given(square_matrix_and_vector())
@settings(max_examples=25, deadline=5000)
def test_solver_inverse_consistency_property(system):
    """Property test: solve(A, b) should match A^(-1) * b when A is invertible."""
    A, b = system

    A_inv = inverse(A)
    if A_inv is not None:
        # Both methods should give same result
        x_solve = solve(A, b)

        # Compute A^(-1) * b manually
        x_inverse = []
        for i in range(A.rows):
            row_packed = A_inv.get_row_bitwise(i)
            dot_product = 0
            for j in range(len(b)):
                if (row_packed >> j) & 1:
                    dot_product ^= b[j]
            x_inverse.append(dot_product)

        assert x_solve is not None, "Invertible system should have solution"
        assert x_solve == x_inverse, "solve() and inverse multiplication should give same result"


@pytest.mark.property
@given(matrix_dimensions(min_size=2, max_size=5))
@settings(max_examples=20, deadline=4000)
def test_homogeneous_system_nullspace_consistency_property(dims):
    """Property test: solutions to Ax = 0 should be in nullspace."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.4, seed=42)

    # Solve homogeneous system Ax = 0
    zero_rhs = [0] * rows
    x_homogeneous = solve(A, zero_rhs)

    if x_homogeneous is not None:
        # Solution should be in nullspace
        assert verify_nullspace_vector(A, x_homogeneous), "Homogeneous solution should be in nullspace"

    # All nullspace vectors should solve homogeneous system
    null_basis = nullspace(A)
    for null_vector in null_basis:
        assert verify_solution(A, zero_rhs, null_vector), "Nullspace vector should solve homogeneous system"


# Edge Cases and Boundary Conditions
@pytest.mark.property
@given(st.integers(min_value=1, max_value=3))
@settings(max_examples=10, deadline=2000)
def test_single_element_solver_properties(n):
    """Property test: 1x1 matrix solver properties."""
    from gf2.generators import ones

    # Test 1x1 matrix with value 1 (use ones generator to avoid set method issues)
    A_one = ones(1, 1)

    # Test solving with different RHS values
    x_1 = solve(A_one, [1])
    x_0 = solve(A_one, [0])

    assert x_1 == [1], "1x1 system [1]*x = [1] should have solution x = [1]"
    assert x_0 == [0], "1x1 system [1]*x = [0] should have solution x = [0]"

    # Test nullspace
    null_basis_one = nullspace(A_one)
    assert len(null_basis_one) == 0, "Full rank 1x1 matrix should have empty nullspace"

    # Test inverse
    A_one_inv = inverse(A_one)
    assert A_one_inv is not None, "1x1 matrix [1] should be invertible"
    assert A_one_inv.get(0, 0) == 1, "Inverse of 1x1 matrix [1] should be [1]"

    # Test 1x1 matrix with value 0
    A_zero = zeros(1, 1)

    x_zero_consistent = solve(A_zero, [0])
    x_zero_inconsistent = solve(A_zero, [1])

    assert x_zero_consistent is not None, "1x1 zero system with zero RHS should have solution"
    assert x_zero_inconsistent is None, "1x1 zero system with non-zero RHS should have no solution"

    # Test nullspace of zero matrix
    null_basis_zero = nullspace(A_zero)
    assert len(null_basis_zero) == 1, "1x1 zero matrix should have 1-dimensional nullspace"
    assert null_basis_zero[0] == [1], "Nullspace of 1x1 zero matrix should be {[1]}"


@pytest.mark.property
@given(st.integers(min_value=2, max_value=4))
@settings(max_examples=10, deadline=3000)
def test_empty_nullspace_properties(n):
    """Property test: matrices with empty nullspace are full rank."""
    identity_matrix = identity(n)

    null_basis = nullspace(identity_matrix)
    matrix_rank = rank(identity_matrix)

    assert len(null_basis) == 0, "Identity matrix should have empty nullspace"
    assert matrix_rank == n, "Identity matrix should have full rank"

    # Empty nullspace implies only trivial solution to homogeneous system
    zero_rhs = [0] * n
    x_homogeneous = solve(identity_matrix, zero_rhs)

    assert x_homogeneous == zero_rhs, "Full rank homogeneous system should have only trivial solution"


# Performance and Stress Properties
@pytest.mark.property
@given(square_matrix_size(min_size=3, max_size=5))
@settings(max_examples=10, deadline=8000)
def test_solver_performance_consistency_property(n):
    """Property test: solver should be consistent across multiple calls."""
    A = random_sparse(n, n, 0.6, seed=42)
    b = [(42 + i) % 2 for i in range(n)]

    # Solve multiple times
    solutions = []
    for _ in range(3):
        x = solve(A, b)
        solutions.append(x)

    # All solutions should be identical (deterministic solver)
    if solutions[0] is not None:
        for sol in solutions[1:]:
            assert sol == solutions[0], "Solver should be deterministic"
            assert verify_solution(A, b, sol), "All solutions should be valid"


@pytest.mark.property
@given(matrix_dimensions(min_size=3, max_size=6))
@settings(max_examples=15, deadline=5000)
def test_nullspace_completeness_property(dims):
    """Property test: nullspace basis should span the entire nullspace."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.3, seed=42)

    null_basis = nullspace(A)
    matrix_rank = rank(A)
    expected_nullity = cols - matrix_rank

    assert len(null_basis) == expected_nullity, (
        f"Nullspace basis size {len(null_basis)} should equal expected nullity {expected_nullity}"
    )

    # If nullspace is non-trivial, verify linear combinations are also in nullspace
    if len(null_basis) >= 2:
        # Test that sum of first two basis vectors is in nullspace
        combined_vector = [(null_basis[0][i] ^ null_basis[1][i]) for i in range(cols)]
        assert verify_nullspace_vector(A, combined_vector), (
            "Linear combination of nullspace vectors should be in nullspace"
        )
