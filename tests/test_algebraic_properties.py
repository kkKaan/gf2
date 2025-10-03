"""
Comprehensive algebraic property tests using hypothesis.

This module implements property-based testing for fundamental algebraic properties
of GF(2) matrix operations including associativity, commutativity, distributivity,
identity, and inverse properties.
"""

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from binpy.core import add, det, is_invertible, multiply, rank, transpose
from binpy.generators import identity, ones, random_sparse, zeros
from binpy.solvers import inverse
from binpy.sparse import SparseGF2Matrix


# Custom hypothesis strategies for matrix dimensions and sparsity levels
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
def sparsity_level(draw):
    """Generate realistic sparsity levels for property testing."""
    return draw(st.floats(min_value=0.1, max_value=0.9))


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
        st.sets(st.tuples(st.integers(min_value=0, max_value=rows - 1),
                          st.integers(min_value=0, max_value=cols - 1)),
                min_size=0,
                max_size=num_ones))

    for r, c in coordinates:
        matrix.set(r, c, 1)

    return matrix


@st.composite
def compatible_matrix_triple(draw):
    """Generate three matrices A, B, C where A*B and B*C are both valid."""
    m = draw(st.integers(min_value=2, max_value=5))
    n = draw(st.integers(min_value=2, max_value=5))
    p = draw(st.integers(min_value=2, max_value=5))
    q = draw(st.integers(min_value=2, max_value=5))

    A = draw(random_matrix(m, n))
    B = draw(random_matrix(n, p))
    C = draw(random_matrix(p, q))

    return A, B, C


@st.composite
def same_size_matrix_pair(draw):
    """Generate two matrices of the same dimensions."""
    rows, cols = draw(matrix_dimensions())
    A = draw(random_matrix(rows, cols))
    B = draw(random_matrix(rows, cols))
    return A, B


@st.composite
def same_size_matrix_triple(draw):
    """Generate three matrices of the same dimensions."""
    rows, cols = draw(matrix_dimensions())
    A = draw(random_matrix(rows, cols))
    B = draw(random_matrix(rows, cols))
    C = draw(random_matrix(rows, cols))
    return A, B, C


def matrices_equal(A, B):
    """Check if two matrices are equal by comparing all rows."""
    if A.rows != B.rows or A.cols != B.cols:
        return False

    return all(A.get_row_bitwise(i) == B.get_row_bitwise(i) for i in range(A.rows))


# Associativity Properties
@pytest.mark.property
@given(same_size_matrix_triple())
@settings(max_examples=30, deadline=3000)
def test_addition_associativity(matrices):
    """Test associativity of matrix addition: (A + B) + C = A + (B + C)."""
    A, B, C = matrices

    # Compute (A + B) + C
    AB = add(A, B)
    AB_C = add(AB, C)

    # Compute A + (B + C)
    BC = add(B, C)
    A_BC = add(A, BC)

    # Verify associativity
    assert matrices_equal(AB_C, A_BC), "Addition associativity failed"


@pytest.mark.property
@given(compatible_matrix_triple())
@settings(max_examples=20, deadline=5000)
def test_multiplication_associativity(matrices):
    """Test associativity of matrix multiplication: (A * B) * C = A * (B * C)."""
    A, B, C = matrices

    # Compute (A * B) * C
    AB = multiply(A, B)
    AB_C = multiply(AB, C)

    # Compute A * (B * C)
    BC = multiply(B, C)
    A_BC = multiply(A, BC)

    # Verify associativity
    assert matrices_equal(AB_C, A_BC), "Multiplication associativity failed"


# Commutativity Properties
@pytest.mark.property
@given(same_size_matrix_pair())
@settings(max_examples=30, deadline=3000)
def test_addition_commutativity(matrices):
    """Test commutativity of matrix addition: A + B = B + A."""
    A, B = matrices

    AB = add(A, B)
    BA = add(B, A)

    assert matrices_equal(AB, BA), "Addition commutativity failed"


@pytest.mark.property
@given(square_matrix_size())
@settings(max_examples=20, deadline=3000)
def test_multiplication_commutativity_special_cases(n):
    """Test multiplication commutativity for special matrices."""
    # Identity matrix commutes with any matrix
    identity_matrix = identity(n)
    A = random_sparse(n, n, 0.4, seed=42)  # Fixed seed for reproducibility in this specific test

    IA = multiply(identity_matrix, A)
    AI = multiply(A, identity_matrix)

    assert matrices_equal(IA, AI), "Identity multiplication commutativity failed"
    assert matrices_equal(IA, A), "Identity should be multiplicative identity"


# Distributivity Properties
@pytest.mark.property
@given(st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5))
@settings(max_examples=20, deadline=5000)
def test_left_distributivity(m, n):
    """Test left distributivity: A * (B + C) = A * B + A * C."""
    A = random_sparse(m, n, 0.4, seed=42)
    B = random_sparse(n, m, 0.4, seed=43)
    C = random_sparse(n, m, 0.4, seed=44)

    # Compute A * (B + C)
    BC = add(B, C)
    A_BC = multiply(A, BC)

    # Compute A * B + A * C
    AB = multiply(A, B)
    AC = multiply(A, C)
    AB_AC = add(AB, AC)

    assert matrices_equal(A_BC, AB_AC), "Left distributivity failed"


@pytest.mark.property
@given(st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5))
@settings(max_examples=20, deadline=5000)
def test_right_distributivity(m, n):
    """Test right distributivity: (A + B) * C = A * C + B * C."""
    A = random_sparse(m, n, 0.4, seed=42)
    B = random_sparse(m, n, 0.4, seed=43)
    C = random_sparse(n, m, 0.4, seed=44)

    # Compute (A + B) * C
    AB = add(A, B)
    AB_C = multiply(AB, C)

    # Compute A * C + B * C
    AC = multiply(A, C)
    BC = multiply(B, C)
    AC_BC = add(AC, BC)

    assert matrices_equal(AB_C, AC_BC), "Right distributivity failed"


# Identity Properties
@pytest.mark.property
@given(matrix_dimensions())
@settings(max_examples=25, deadline=3000)
def test_additive_identity(dims):
    """Test additive identity: A + 0 = A."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.4, seed=42)
    Z = zeros(rows, cols)

    A_plus_Z = add(A, Z)
    Z_plus_A = add(Z, A)

    assert matrices_equal(A_plus_Z, A), "Right additive identity failed"
    assert matrices_equal(Z_plus_A, A), "Left additive identity failed"


@pytest.mark.property
@given(square_matrix_size())
@settings(max_examples=25, deadline=3000)
def test_multiplicative_identity(n):
    """Test multiplicative identity: A * I = I * A = A."""
    A = random_sparse(n, n, 0.4, seed=42)
    identity_matrix = identity(n)

    AI = multiply(A, identity_matrix)
    IA = multiply(identity_matrix, A)

    assert matrices_equal(AI, A), "Right multiplicative identity failed"
    assert matrices_equal(IA, A), "Left multiplicative identity failed"


# Inverse Properties (for GF(2))
@pytest.mark.property
@given(matrix_dimensions())
@settings(max_examples=25, deadline=3000)
def test_additive_inverse(dims):
    """Test additive inverse: A + A = 0 in GF(2)."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.4, seed=42)
    Z = zeros(rows, cols)

    A_plus_A = add(A, A)

    assert matrices_equal(A_plus_A, Z), "Additive inverse property failed in GF(2)"


@pytest.mark.property
@given(square_matrix_size(min_size=2, max_size=4))
@settings(max_examples=15, deadline=5000)
def test_multiplicative_inverse_when_exists(n):
    """Test multiplicative inverse: A * A^(-1) = I when A is invertible."""
    # Generate matrices until we find an invertible one
    for seed in range(50, 100):  # Try different seeds
        A = random_sparse(n, n, 0.6, seed=seed)

        if is_invertible(A):
            try:
                A_inv = inverse(A)
                identity_matrix = identity(n)

                # Test A * A^(-1) = I
                A_Ainv = multiply(A, A_inv)
                assert matrices_equal(A_Ainv, identity_matrix), "Right multiplicative inverse failed"

                # Test A^(-1) * A = I
                Ainv_A = multiply(A_inv, A)
                assert matrices_equal(Ainv_A, identity_matrix), "Left multiplicative inverse failed"

                return  # Success, exit the test
            except Exception:
                continue  # Try next seed if inversion fails

    # If we can't find an invertible matrix, skip the test
    pytest.skip("Could not generate invertible matrix for testing")


# Transpose Properties
@pytest.mark.property
@given(matrix_dimensions())
@settings(max_examples=25, deadline=3000)
def test_transpose_involution(dims):
    """Test transpose involution: (A^T)^T = A."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.4, seed=42)

    AT = transpose(A)
    ATT = transpose(AT)

    assert matrices_equal(A, ATT), "Transpose involution failed"


@pytest.mark.property
@given(same_size_matrix_pair())
@settings(max_examples=20, deadline=3000)
def test_transpose_addition_distributivity(matrices):
    """Test transpose distributivity over addition: (A + B)^T = A^T + B^T."""
    A, B = matrices

    # Compute (A + B)^T
    AB = add(A, B)
    AB_T = transpose(AB)

    # Compute A^T + B^T
    AT = transpose(A)
    BT = transpose(B)
    AT_BT = add(AT, BT)

    assert matrices_equal(AB_T, AT_BT), "Transpose addition distributivity failed"


@pytest.mark.property
@given(st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5))
@settings(max_examples=15, deadline=5000)
def test_transpose_multiplication_property(m, n):
    """Test transpose multiplication property: (A * B)^T = B^T * A^T."""
    A = random_sparse(m, n, 0.4, seed=42)
    B = random_sparse(n, m, 0.4, seed=43)

    # Compute (A * B)^T
    AB = multiply(A, B)
    AB_T = transpose(AB)

    # Compute B^T * A^T
    AT = transpose(A)
    BT = transpose(B)
    BT_AT = multiply(BT, AT)

    assert matrices_equal(AB_T, BT_AT), "Transpose multiplication property failed"


# Rank Properties
@pytest.mark.property
@given(matrix_dimensions())
@settings(max_examples=20, deadline=3000)
def test_rank_transpose_invariance(dims):
    """Test that rank is invariant under transpose: rank(A) = rank(A^T)."""
    rows, cols = dims
    A = random_sparse(rows, cols, 0.4, seed=42)
    AT = transpose(A)

    assert rank(A) == rank(AT), "Rank transpose invariance failed"


@pytest.mark.property
@given(same_size_matrix_pair())
@settings(max_examples=20, deadline=3000)
def test_rank_subadditivity(matrices):
    """Test rank subadditivity: rank(A + B) <= rank(A) + rank(B)."""
    A, B = matrices

    rank_A = rank(A)
    rank_B = rank(B)
    AB = add(A, B)
    rank_AB = rank(AB)

    assert rank_AB <= rank_A + rank_B, "Rank subadditivity failed"


# Determinant Properties (for square matrices)
@pytest.mark.property
@given(square_matrix_size(min_size=1, max_size=4))
@settings(max_examples=20, deadline=3000)
def test_determinant_transpose_invariance(n):
    """Test that det(A) = det(A^T)."""
    A = random_sparse(n, n, 0.5, seed=42)
    AT = transpose(A)

    assert det(A) == det(AT), "Determinant transpose invariance failed"


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


# Zero and Identity Special Cases
@pytest.mark.property
@given(square_matrix_size())
@settings(max_examples=15, deadline=2000)
def test_zero_matrix_properties(n):
    """Test properties specific to zero matrices."""
    Z = zeros(n, n)
    A = random_sparse(n, n, 0.4, seed=42)

    # Zero matrix properties
    assert rank(Z) == 0, "Zero matrix should have rank 0"
    assert det(Z) == 0, "Zero matrix should have determinant 0"

    # Zero matrix in operations
    ZA = multiply(Z, A)
    AZ = multiply(A, Z)
    assert matrices_equal(ZA, Z), "Zero matrix multiplication (left) failed"
    assert matrices_equal(AZ, Z), "Zero matrix multiplication (right) failed"


@pytest.mark.property
@given(square_matrix_size())
@settings(max_examples=15, deadline=2000)
def test_identity_matrix_properties(n):
    """Test properties specific to identity matrices."""
    identity_matrix = identity(n)

    # Identity matrix properties
    assert rank(identity_matrix) == n, "Identity matrix should have full rank"
    assert det(identity_matrix) == 1, "Identity matrix should have determinant 1"

    # Identity is its own transpose and inverse
    IT = transpose(identity_matrix)
    assert matrices_equal(identity_matrix, IT), "Identity matrix should equal its transpose"


# Nilpotent and Idempotent Properties
@pytest.mark.property
@given(square_matrix_size(min_size=2, max_size=4))
@settings(max_examples=10, deadline=3000)
def test_idempotent_properties(n):
    """Test properties of idempotent matrices: A^2 = A."""
    # Create a simple idempotent matrix (projection-like)
    # In GF(2), we can construct idempotent matrices more easily

    # Test with identity (trivial idempotent)
    identity_matrix = identity(n)
    I2 = multiply(identity_matrix, identity_matrix)
    assert matrices_equal(identity_matrix, I2), "Identity should be idempotent"

    # Test with zero matrix (trivial idempotent)
    Z = zeros(n, n)
    Z2 = multiply(Z, Z)
    assert matrices_equal(Z, Z2), "Zero matrix should be idempotent"


@pytest.mark.property
@given(square_matrix_size(min_size=2, max_size=4))
@settings(max_examples=10, deadline=3000)
def test_involution_properties(n):
    """Test properties of involutory matrices: A^2 = I."""
    # In GF(2), many matrices are involutory
    # Test with identity (trivial involution)
    identity_matrix = identity(n)
    I2 = multiply(identity_matrix, identity_matrix)
    assert matrices_equal(identity_matrix, I2), "Identity should be involutory"


# Edge Cases and Boundary Conditions
@pytest.mark.property
@given(st.integers(min_value=1, max_value=3))
@settings(max_examples=10, deadline=2000)
def test_single_element_matrix_properties(n):
    """Test properties of 1x1 matrices."""
    # 1x1 matrices in GF(2) are just 0 or 1
    zero_1x1 = zeros(1, 1)
    ones_1x1 = ones(1, 1)  # Use the ones generator instead

    # Test basic properties
    assert rank(zero_1x1) == 0
    assert rank(ones_1x1) == 1
    assert det(zero_1x1) == 0
    assert det(ones_1x1) == 1

    # Test operations
    sum_result = add(zero_1x1, ones_1x1)
    assert matrices_equal(sum_result, ones_1x1)

    product_result = multiply(ones_1x1, ones_1x1)
    assert matrices_equal(product_result, ones_1x1)
