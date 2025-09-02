"""
Comprehensive core operations tests for binpy.

This module provides exhaustive testing of core GF(2) matrix operations:
- Addition (XOR) with mathematical property verification
- Matrix multiplication with various matrix types and sizes
- Transpose operations with property verification
- Edge cases including empty matrices, single elements, and maximum sizes
- Mathematical properties: associativity, commutativity, identity laws
"""

import pytest
from hypothesis import given, strategies as st

from binpy.core import add, det, multiply, rank, trace, transpose
from binpy.generators import identity, random_sparse, zeros
from binpy.sparse import SparseGF2Matrix


class TestAdditionOperation:
    """Comprehensive tests for GF(2) matrix addition (XOR)."""

    @pytest.mark.unit
    def test_addition_basic_functionality(self, small_matrices):
        """Test basic addition functionality with known matrices."""
        # Identity + Identity = Zero
        identity_matrix = small_matrices["identity_3x3"]
        Z = small_matrices["zero_3x3"]
        result = add(identity_matrix, identity_matrix)

        # Verify result is zero matrix
        for i in range(3):
            assert result.get_row_bitwise(i) == Z.get_row_bitwise(i)

        # Zero + Matrix = Matrix (use compatible dimensions)
        A = small_matrices["sparse_5x5_low"]
        Z_5x5 = zeros(5, 5)  # Create compatible zero matrix
        result = add(Z_5x5, A)
        for i in range(A.rows):
            assert result.get_row_bitwise(i) == A.get_row_bitwise(i)

    @pytest.mark.unit
    def test_addition_commutativity(self, matrix_pairs):
        """Test that A + B = B + A for all matrix pairs."""
        for pair_name, (A, B) in matrix_pairs.items():
            if A.rows == B.rows and A.cols == B.cols:
                result_AB = add(A, B)
                result_BA = add(B, A)

                # Verify commutativity
                for i in range(A.rows):
                    assert result_AB.get_row_bitwise(i) == result_BA.get_row_bitwise(i), (
                        f"Commutativity failed for {pair_name}"
                    )

    @pytest.mark.unit
    def test_addition_associativity(self, small_matrices):
        """Test that (A + B) + C = A + (B + C)."""
        A = small_matrices["sparse_5x5_low"]
        B = small_matrices["sparse_5x5_med"]
        C = small_matrices["sparse_5x5_high"]

        # Compute (A + B) + C
        AB = add(A, B)
        result_left = add(AB, C)

        # Compute A + (B + C)
        BC = add(B, C)
        result_right = add(A, BC)

        # Verify associativity
        for i in range(A.rows):
            assert result_left.get_row_bitwise(i) == result_right.get_row_bitwise(i)

    @pytest.mark.unit
    def test_addition_identity_property(self, small_matrices):
        """Test that A + 0 = A for all matrices."""
        matrices_to_test = ["identity_3x3", "sparse_5x5_low", "sparse_5x5_med", "ones_2x2"]

        for matrix_name in matrices_to_test:
            A = small_matrices[matrix_name]
            Z = zeros(A.rows, A.cols)
            result = add(A, Z)

            # Verify A + 0 = A
            for i in range(A.rows):
                assert result.get_row_bitwise(i) == A.get_row_bitwise(i), (
                    f"Identity property failed for {matrix_name}"
                )

    @pytest.mark.unit
    def test_addition_self_inverse(self, small_matrices):
        """Test that A + A = 0 for all matrices (self-inverse property in GF(2))."""
        matrices_to_test = ["identity_3x3", "sparse_5x5_low", "ones_2x2", "sparse_4x6"]

        for matrix_name in matrices_to_test:
            A = small_matrices[matrix_name]
            result = add(A, A)

            # Verify A + A = 0
            for i in range(A.rows):
                assert result.get_row_bitwise(i) == 0, f"Self-inverse property failed for {matrix_name}"

    @pytest.mark.unit
    def test_addition_dimension_mismatch_errors(self, small_matrices):
        """Test that addition raises appropriate errors for dimension mismatches."""
        A = small_matrices["sparse_4x6"]  # 4x6
        B = small_matrices["sparse_6x4"]  # 6x4
        C = small_matrices["identity_3x3"]  # 3x3

        # Test row mismatch
        with pytest.raises(ValueError, match="Matrix dimensions must match"):
            add(A, B)

        # Test column mismatch
        with pytest.raises(ValueError, match="Matrix dimensions must match"):
            add(A, C)

    @pytest.mark.unit
    def test_addition_edge_cases(self, edge_case_matrices):
        """Test addition with edge case matrices."""
        # Single element matrices
        single_zero = edge_case_matrices["single_element_1x1_zero"]
        single_one = edge_case_matrices["single_element_1x1_one"]

        # 1 + 1 = 0 in GF(2)
        result = add(single_one, single_one)
        assert result.get_bit(0, 0) == 0

        # 1 + 0 = 1
        result = add(single_one, single_zero)
        assert result.get_bit(0, 0) == 1

        # Single row/column matrices
        single_row = edge_case_matrices["single_row_1x10"]
        single_col = edge_case_matrices["single_col_10x1"]

        # Test self-addition
        result = add(single_row, single_row)
        assert result.get_row_bitwise(0) == 0

        result = add(single_col, single_col)
        for i in range(10):
            assert result.get_bit(i, 0) == 0

    @pytest.mark.property
    @given(st.integers(min_value=1, max_value=10), st.floats(min_value=0.1, max_value=0.9))
    def test_addition_properties_random(self, n, density):
        """Property-based test for addition with random matrices."""
        # Generate random matrices
        A = random_sparse(n, n, density, seed=42)
        B = random_sparse(n, n, density, seed=43)
        C = random_sparse(n, n, density, seed=44)

        # Test commutativity: A + B = B + A
        AB = add(A, B)
        BA = add(B, A)
        for i in range(n):
            assert AB.get_row_bitwise(i) == BA.get_row_bitwise(i)

        # Test associativity: (A + B) + C = A + (B + C)
        AB_C = add(add(A, B), C)
        A_BC = add(A, add(B, C))
        for i in range(n):
            assert AB_C.get_row_bitwise(i) == A_BC.get_row_bitwise(i)

        # Test self-inverse: A + A = 0
        AA = add(A, A)
        for i in range(n):
            assert AA.get_row_bitwise(i) == 0


class TestMultiplicationOperation:
    """Comprehensive tests for GF(2) matrix multiplication."""

    @pytest.mark.unit
    def test_multiplication_basic_functionality(self, small_matrices):
        """Test basic multiplication functionality."""
        # Identity * Matrix = Matrix
        identity_matrix = small_matrices["identity_3x3"]
        A = random_sparse(3, 3, 0.5, seed=100)
        result = multiply(identity_matrix, A)

        for i in range(3):
            assert result.get_row_bitwise(i) == A.get_row_bitwise(i)

        # Matrix * Identity = Matrix
        result = multiply(A, identity_matrix)
        for i in range(3):
            assert result.get_row_bitwise(i) == A.get_row_bitwise(i)

    @pytest.mark.unit
    def test_multiplication_zero_matrix(self, small_matrices):
        """Test multiplication with zero matrices."""
        A = small_matrices["sparse_5x5_med"]
        # Z = small_matrices["zero_3x3"]
        Z_compatible = zeros(5, 5)

        # A * 0 = 0
        result = multiply(A, Z_compatible)
        for i in range(A.rows):
            assert result.get_row_bitwise(i) == 0

        # 0 * A = 0
        result = multiply(Z_compatible, A)
        for i in range(A.rows):
            assert result.get_row_bitwise(i) == 0

    @pytest.mark.unit
    def test_multiplication_associativity(self, small_matrices):
        """Test that (A * B) * C = A * (B * C)."""
        # Use smaller matrices for efficiency
        A = random_sparse(3, 4, 0.4, seed=200)
        B = random_sparse(4, 3, 0.5, seed=201)
        C = random_sparse(3, 2, 0.6, seed=202)

        # Compute (A * B) * C
        AB = multiply(A, B)
        result_left = multiply(AB, C)

        # Compute A * (B * C)
        BC = multiply(B, C)
        result_right = multiply(A, BC)

        # Verify associativity
        for i in range(A.rows):
            assert result_left.get_row_bitwise(i) == result_right.get_row_bitwise(i)

    @pytest.mark.unit
    def test_multiplication_distributivity(self, small_matrices):
        """Test that A * (B + C) = A * B + A * C."""
        A = random_sparse(3, 4, 0.4, seed=300)
        B = random_sparse(4, 3, 0.5, seed=301)
        C = random_sparse(4, 3, 0.6, seed=302)

        # Compute A * (B + C)
        BC_sum = add(B, C)
        result_left = multiply(A, BC_sum)

        # Compute A * B + A * C
        AB = multiply(A, B)
        AC = multiply(A, C)
        result_right = add(AB, AC)

        # Verify distributivity
        for i in range(A.rows):
            assert result_left.get_row_bitwise(i) == result_right.get_row_bitwise(i)

    @pytest.mark.unit
    def test_multiplication_dimension_compatibility(self, matrix_pairs):
        """Test multiplication with compatible and incompatible dimensions."""
        # Test compatible dimensions
        A, B = matrix_pairs["multiplicable_3x4_4x5"]
        result = multiply(A, B)
        assert result.rows == A.rows
        assert result.cols == B.cols

        # Test incompatible dimensions
        A, B = matrix_pairs["same_size_3x3"]
        C = random_sparse(5, 3, 0.4, seed=400)  # Incompatible

        with pytest.raises(ValueError, match="Inner dimensions must match"):
            multiply(A, C)

    @pytest.mark.unit
    def test_multiplication_edge_cases(self, edge_case_matrices):
        """Test multiplication with edge case matrices."""
        # Single element matrices
        single_zero = edge_case_matrices["single_element_1x1_zero"]
        single_one = edge_case_matrices["single_element_1x1_one"]

        # 1 * 1 = 1
        result = multiply(single_one, single_one)
        assert result.get_bit(0, 0) == 1

        # 1 * 0 = 0
        result = multiply(single_one, single_zero)
        assert result.get_bit(0, 0) == 0

        # Single row * single column
        single_row = edge_case_matrices["single_row_1x10"]
        single_col = edge_case_matrices["single_col_10x1"]

        result = multiply(single_row, single_col)
        assert result.rows == 1
        assert result.cols == 1

    @pytest.mark.unit
    def test_multiplication_powers(self, small_matrices):
        """Test matrix powers through repeated multiplication."""
        A = small_matrices["identity_3x3"]

        # I^2 = I
        I_squared = multiply(A, A)
        for i in range(3):
            assert I_squared.get_row_bitwise(i) == A.get_row_bitwise(i)

        # Test with a non-identity matrix
        B = random_sparse(3, 3, 0.4, seed=500)
        B_squared = multiply(B, B)
        B_cubed = multiply(B_squared, B)

        # Verify dimensions are preserved
        assert B_squared.rows == B.rows
        assert B_squared.cols == B.cols
        assert B_cubed.rows == B.rows
        assert B_cubed.cols == B.cols

    @pytest.mark.property
    @given(st.integers(min_value=2, max_value=8))
    def test_multiplication_properties_random(self, n):
        """Property-based test for multiplication with random matrices."""
        # Generate random matrices with compatible dimensions
        A = random_sparse(n, n, 0.3, seed=600)
        B = random_sparse(n, n, 0.3, seed=601)
        C = random_sparse(n, n, 0.3, seed=602)
        identity_matrix = identity(n)

        # Test identity property: A * I = I * A = A
        AI = multiply(A, identity_matrix)
        IA = multiply(identity_matrix, A)
        for i in range(n):
            assert AI.get_row_bitwise(i) == A.get_row_bitwise(i)
            assert IA.get_row_bitwise(i) == A.get_row_bitwise(i)

        # Test associativity: (A * B) * C = A * (B * C)
        AB_C = multiply(multiply(A, B), C)
        A_BC = multiply(A, multiply(B, C))
        for i in range(n):
            assert AB_C.get_row_bitwise(i) == A_BC.get_row_bitwise(i)


class TestTransposeOperation:
    """Comprehensive tests for matrix transpose operation."""

    @pytest.mark.unit
    def test_transpose_basic_functionality(self, small_matrices):
        """Test basic transpose functionality."""
        # Transpose of identity is identity
        identity_matrix = small_matrices["identity_3x3"]
        IT = transpose(identity_matrix)

        for i in range(3):
            assert IT.get_row_bitwise(i) == identity_matrix.get_row_bitwise(i)

        # Test rectangular matrix transpose
        A = small_matrices["sparse_4x6"]
        AT = transpose(A)

        assert AT.rows == A.cols
        assert AT.cols == A.rows

        # Verify transpose property: A[i,j] = A^T[j,i]
        for i in range(A.rows):
            for j in range(A.cols):
                assert A.get_bit(i, j) == AT.get_bit(j, i)

    @pytest.mark.unit
    def test_transpose_involution(self, small_matrices):
        """Test that (A^T)^T = A (transpose involution property)."""
        matrices_to_test = ["identity_3x3", "sparse_5x5_low", "sparse_4x6", "ones_2x2"]

        for matrix_name in matrices_to_test:
            A = small_matrices[matrix_name]
            ATT = transpose(transpose(A))

            # Verify (A^T)^T = A
            assert ATT.rows == A.rows
            assert ATT.cols == A.cols

            for i in range(A.rows):
                assert ATT.get_row_bitwise(i) == A.get_row_bitwise(i), (
                    f"Transpose involution failed for {matrix_name}"
                )

    @pytest.mark.unit
    def test_transpose_addition_property(self, matrix_pairs):
        """Test that (A + B)^T = A^T + B^T."""
        for pair_name, (A, B) in matrix_pairs.items():
            if A.rows == B.rows and A.cols == B.cols:
                # Compute (A + B)^T
                AB_sum = add(A, B)
                AB_sum_T = transpose(AB_sum)

                # Compute A^T + B^T
                AT = transpose(A)
                BT = transpose(B)
                AT_BT_sum = add(AT, BT)

                # Verify (A + B)^T = A^T + B^T
                for i in range(AB_sum_T.rows):
                    assert AB_sum_T.get_row_bitwise(i) == AT_BT_sum.get_row_bitwise(i), (
                        f"Transpose addition property failed for {pair_name}"
                    )

    @pytest.mark.unit
    def test_transpose_multiplication_property(self, matrix_pairs):
        """Test that (A * B)^T = B^T * A^T."""
        # Use compatible multiplication pairs
        compatible_pairs = ["multiplicable_3x4_4x5", "multiplicable_5x3_3x7"]

        for pair_name in compatible_pairs:
            if pair_name in matrix_pairs:
                A, B = matrix_pairs[pair_name]

                # Compute (A * B)^T
                AB = multiply(A, B)
                AB_T = transpose(AB)

                # Compute B^T * A^T
                AT = transpose(A)
                BT = transpose(B)
                BT_AT = multiply(BT, AT)

                # Verify (A * B)^T = B^T * A^T
                for i in range(AB_T.rows):
                    assert AB_T.get_row_bitwise(i) == BT_AT.get_row_bitwise(i), (
                        f"Transpose multiplication property failed for {pair_name}"
                    )

    @pytest.mark.unit
    def test_transpose_edge_cases(self, edge_case_matrices):
        """Test transpose with edge case matrices."""
        # Single element matrices
        single_zero = edge_case_matrices["single_element_1x1_zero"]
        single_one = edge_case_matrices["single_element_1x1_one"]

        # Transpose of 1x1 matrix is itself
        assert transpose(single_zero).get_bit(0, 0) == 0
        assert transpose(single_one).get_bit(0, 0) == 1

        # Single row becomes single column
        single_row = edge_case_matrices["single_row_1x10"]
        single_row_T = transpose(single_row)

        assert single_row_T.rows == 10
        assert single_row_T.cols == 1

        # Verify elements are preserved
        for j in range(10):
            assert single_row.get_bit(0, j) == single_row_T.get_bit(j, 0)

        # Single column becomes single row
        single_col = edge_case_matrices["single_col_10x1"]
        single_col_T = transpose(single_col)

        assert single_col_T.rows == 1
        assert single_col_T.cols == 10

        for i in range(10):
            assert single_col.get_bit(i, 0) == single_col_T.get_bit(0, i)

    @pytest.mark.unit
    def test_transpose_symmetric_matrices(self, small_matrices):
        """Test transpose of symmetric matrices."""
        # Create a symmetric matrix
        n = 4
        symmetric = SparseGF2Matrix(n, n)

        # Set symmetric pattern
        positions = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
        for r, c in positions:
            symmetric.set_bit(r, c)

        # Transpose should equal original
        symmetric_T = transpose(symmetric)

        for i in range(n):
            assert symmetric.get_row_bitwise(i) == symmetric_T.get_row_bitwise(i)

    @pytest.mark.property
    @given(st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=10))
    def test_transpose_properties_random(self, rows, cols):
        """Property-based test for transpose with random matrices."""
        A = random_sparse(rows, cols, 0.3, seed=700)

        # Test involution: (A^T)^T = A
        ATT = transpose(transpose(A))
        assert ATT.rows == A.rows
        assert ATT.cols == A.cols

        for i in range(rows):
            assert ATT.get_row_bitwise(i) == A.get_row_bitwise(i)

        # Test dimension swap
        AT = transpose(A)
        assert AT.rows == cols
        assert AT.cols == rows


class TestCoreOperationsIntegration:
    """Integration tests combining multiple core operations."""

    @pytest.mark.unit
    def test_operations_combination(self, small_matrices):
        """Test combinations of core operations."""
        A = small_matrices["sparse_5x5_med"]
        B = small_matrices["sparse_5x5_high"]

        # Test (A + B)^T = A^T + B^T
        AB_sum = add(A, B)
        AB_sum_T = transpose(AB_sum)

        AT = transpose(A)
        BT = transpose(B)
        AT_BT_sum = add(AT, BT)

        for i in range(AB_sum_T.rows):
            assert AB_sum_T.get_row_bitwise(i) == AT_BT_sum.get_row_bitwise(i)

    @pytest.mark.unit
    def test_rank_preservation_properties(self, small_matrices):
        """Test rank-related properties of operations."""
        A = small_matrices["identity_5x5"]

        # rank(A) = rank(A^T)
        AT = transpose(A)
        assert rank(A) == rank(AT)

        # rank(A + A) = 0 in GF(2)
        AA = add(A, A)
        assert rank(AA) == 0

    @pytest.mark.unit
    def test_determinant_properties(self, small_matrices):
        """Test determinant properties with operations."""
        A = small_matrices["identity_3x3"]

        # det(A^T) = det(A)
        AT = transpose(A)
        assert det(A) == det(AT)

        # det(I) = 1
        assert det(A) == 1

    @pytest.mark.unit
    def test_trace_properties(self, small_matrices):
        """Test trace properties with operations."""
        A = small_matrices["identity_5x5"]
        B = random_sparse(5, 5, 0.3, seed=800)

        # tr(A + B) = tr(A) + tr(B) in GF(2)
        AB_sum = add(A, B)
        assert trace(AB_sum) == (trace(A) + trace(B)) % 2

        # tr(A) = tr(A^T)
        AT = transpose(A)
        assert trace(A) == trace(AT)

    @pytest.mark.performance
    def test_operations_performance_scaling(self, performance_context):
        """Test performance scaling of operations with increasing matrix size."""
        sizes = [5, 10, 15]  # Use smaller sizes to avoid overflow

        for size in sizes:
            A = random_sparse(size, size, 0.2, seed=900 + size)
            B = random_sparse(size, size, 0.2, seed=901 + size)

            # Measure addition performance
            with performance_context(f"addition_{size}x{size}") as _perf:
                _result = add(A, B)

            # Measure multiplication performance
            with performance_context(f"multiplication_{size}x{size}") as _perf:
                _result = multiply(A, B)

            # Measure transpose performance
            with performance_context(f"transpose_{size}x{size}") as _perf:
                _result = transpose(A)

    @pytest.mark.stress
    def test_operations_with_extreme_matrices(self, edge_case_matrices):
        """Stress test operations with extreme matrices."""
        very_sparse = edge_case_matrices["very_sparse"]
        very_dense = edge_case_matrices["very_dense"]

        # Test operations don't crash with extreme sparsity
        result = add(very_sparse, very_sparse)
        assert result.get_row_bitwise(0) == 0  # Should be zero

        result = transpose(very_sparse)
        assert result.rows == very_sparse.cols

        # Test with very dense matrices
        result = transpose(very_dense)
        assert result.rows == very_dense.cols

        # Test mixed sparsity operations
        if very_sparse.rows == very_dense.rows and very_sparse.cols == very_dense.cols:
            result = add(very_sparse, very_dense)
            # Should not crash and preserve dimensions
            assert result.rows == very_sparse.rows
            assert result.cols == very_sparse.cols
