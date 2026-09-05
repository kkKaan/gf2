"""
Invariants that must hold for a released build.

Two kinds of check live here, both aimed at failures the rest of the suite is
structurally unable to see:

1. **Cross-path agreement.** Several operations pick an implementation at run
   time - from matrix density, from dimension thresholds, from storage format.
   A per-function test exercises whichever branch its input happens to select,
   so a divergence between branches passes unnoticed. These tests drive the
   same input down both paths and compare.

2. **Representation independence.** The packed representation is defined as
   little-endian and is reached through raw bytes. On a host where that
   assumption does not hold the library returns wrong answers rather than
   failing, so the conversions are tested against explicitly byte-ordered
   arrays instead of trusting the host.
"""

import random

import numpy as np
import pytest

import gf2
from gf2.core import (
    _M4RM_MIN_CELLS,
    _PACKED_ELIM_MIN_DIM,
    _U64,
    _int_from_words,
    _pack_rows_u64,
    _packed_u64,
    _rank_packed,
    lu_decomposition,
    multiply,
    rank,
    transpose,
)
from gf2.sparse import SparseGF2Matrix


def dense_ref_multiply(a, b):
    """GF(2) product via NumPy. uint8 wraps mod 256, which preserves mod 2."""
    return ((np.array(a, dtype=np.uint8) @ np.array(b, dtype=np.uint8)) & 1).tolist()


def random_matrix(rows, cols, density, seed):
    rng = random.Random(seed)
    return [[1 if rng.random() < density else 0 for _ in range(cols)] for _ in range(rows)]


# ---------------------------------------------------------------------------
# byte order
# ---------------------------------------------------------------------------
class TestByteOrderIndependence:
    """The packed form is little-endian by definition, not by host accident.

    Every int/array conversion pairs an explicit ``"little"`` on the Python
    side with a NumPy array. If that array were native-order, then on a
    big-endian host a single set bit in word 0 would read back as 2**56 and
    ``unpackbits`` would return the wrong byte's bits - silently, with no
    exception. These tests feed explicitly byte-ordered arrays through the
    conversions, so they check the layout logic on any host.
    """

    @pytest.mark.unit
    def test_packed_dtype_is_explicitly_little_endian(self):
        M = SparseGF2Matrix(4, 200, random_matrix(4, 200, 0.5, 1))
        assert _U64.str[0] == "<", "the packed dtype must not be native-order"
        assert M.packed_u64().dtype == _U64
        assert gf2.DenseGF2Matrix(3, 100).data.dtype == _U64

    @pytest.mark.unit
    @pytest.mark.parametrize("order", ["<u8", ">u8"])
    def test_int_from_words_ignores_array_byte_order(self, order):
        for value, words in (
            (1, [1, 0]),
            (2**64, [0, 1]),
            (2**64 + 1, [1, 1]),
            (2**63, [1 << 63, 0]),
        ):
            got = _int_from_words(np.array(words, dtype=order))
            assert got == value, f"{order}: expected {value}, got {got}"

    @pytest.mark.unit
    @pytest.mark.parametrize("order", ["<u8", ">u8"])
    def test_rank_matches_across_array_byte_order(self, order):
        n = _PACKED_ELIM_MIN_DIM + 16
        rows = [int.from_bytes(random.Random(i).randbytes(n // 8), "little") for i in range(n)]
        packed = _pack_rows_u64(rows, n)
        assert _rank_packed(packed.astype(order), n) == _rank_packed(packed, n)

    @pytest.mark.unit
    def test_round_trip_through_packed_form(self):
        data = random_matrix(30, 150, 0.4, seed=7)
        A = SparseGF2Matrix(30, 150, data)
        assert A.to_dense() == data
        for i, row in enumerate(A.get_all_rows_bitwise()):
            assert _int_from_words(_packed_u64(A)[i]) == row


# ---------------------------------------------------------------------------
# implementation thresholds
# ---------------------------------------------------------------------------
class TestImplementationPathsAgree:
    """Both sides of every run-time implementation switch must agree.

    ``multiply`` chooses between a blocked NumPy kernel and a big-integer
    row-XOR loop; ``rank`` switches to vectorised elimination past a dimension
    threshold. A test that happens to sit on one side of a threshold cannot see
    a bug on the other.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("density", [0.01, 0.2, 0.5, 0.9, 1.0])
    def test_multiply_both_kernels_match_numpy(self, density):
        # _M4RM_MIN_CELLS is a cell count, so sizes are chosen to land either
        # side of it while keeping the density branch in play too.
        for n in (8, 64, 200):
            a = random_matrix(n, n, density, seed=n)
            b = random_matrix(n, n, density, seed=n + 1)
            got = multiply(SparseGF2Matrix(n, n, a), SparseGF2Matrix(n, n, b)).to_dense()
            assert got == dense_ref_multiply(a, b), f"n={n} density={density}"

    @pytest.mark.unit
    def test_multiply_straddles_the_blocking_threshold(self):
        # Sizes whose cell count sits immediately either side of the switch.
        n_small = max(2, int(_M4RM_MIN_CELLS**0.5) - 1)
        n_large = int(_M4RM_MIN_CELLS**0.5) + 1
        for n in (n_small, n_large):
            a = random_matrix(n, n, 0.5, seed=n * 3)
            b = random_matrix(n, n, 0.5, seed=n * 5)
            got = multiply(SparseGF2Matrix(n, n, a), SparseGF2Matrix(n, n, b)).to_dense()
            assert got == dense_ref_multiply(a, b), f"n={n}"

    @pytest.mark.unit
    def test_rank_straddles_the_vectorisation_threshold(self):
        for n in (_PACKED_ELIM_MIN_DIM - 1, _PACKED_ELIM_MIN_DIM, _PACKED_ELIM_MIN_DIM + 1):
            data = random_matrix(n, n, 0.5, seed=n)
            A = SparseGF2Matrix(n, n, data)
            rows = A.get_all_rows_bitwise()
            assert rank(A) == _rank_packed(_pack_rows_u64(rows, n), n), f"n={n}"

    @pytest.mark.unit
    @pytest.mark.parametrize("hint", ["auto", "csr", "csr_compact", "bitpacked"])
    def test_every_operation_is_storage_format_independent(self, hint):
        """The same matrix in any storage format must give the same answers.

        Format is normally chosen from density by a threshold that sits at
        exactly 0.5, so a random half-dense matrix lands on either side of it
        run to run. Results must not depend on which side it landed.
        """
        data = random_matrix(24, 24, 0.5, seed=11)
        reference = SparseGF2Matrix(24, 24, data, "bitpacked")
        candidate = SparseGF2Matrix(24, 24, data, hint)

        assert candidate.to_dense() == reference.to_dense()
        assert candidate.get_all_rows_bitwise() == reference.get_all_rows_bitwise()
        assert rank(candidate) == rank(reference)
        assert gf2.det(candidate) == gf2.det(reference)
        assert gf2.trace(candidate) == gf2.trace(reference)
        assert transpose(candidate).to_dense() == transpose(reference).to_dense()
        assert multiply(candidate, candidate).to_dense() == multiply(reference, reference).to_dense()
        assert gf2.nullspace(candidate) == gf2.nullspace(reference)

    @pytest.mark.unit
    def test_nullspace_entry_points_agree(self):
        """nullspace, nullspace_bitwise and nullspace_fast must all be valid.

        They legitimately pick different free variables, so the check is
        membership in the null space, not vector equality.
        """
        data = random_matrix(40, 41, 0.5, seed=13)
        A = SparseGF2Matrix(40, 41, data)

        def in_nullspace(bits):
            x = [int(c) for c in bits]
            assert any(x), "the trivial vector is not a witness"
            return all(sum(r & v for r, v in zip(row, x, strict=True)) % 2 == 0 for row in data)

        assert in_nullspace(gf2.nullspace_fast(data)[0])
        assert in_nullspace(gf2.nullspace_bitwise(A)[0])
        basis = gf2.nullspace(A)
        assert len(basis) == 41 - rank(A)
        for vector in basis:
            assert in_nullspace("".join(str(b) for b in vector))

    @pytest.mark.unit
    def test_solve_multiple_rhs_matches_solving_each_column(self):
        """One batched elimination must agree with solving column by column.

        The contract is all-or-nothing: a result is returned only when *every*
        column is consistent, so the comparison is against `all(...)` of the
        per-column solves rather than against each one individually.
        """
        rng = random.Random(17)
        batched_solved = per_column_solved = 0

        for _ in range(60):
            n, k = rng.randint(1, 7), rng.randint(1, 4)
            a = [[rng.getrandbits(1) for _ in range(n)] for _ in range(n)]
            b = [[rng.getrandbits(1) for _ in range(k)] for _ in range(n)]
            A, B = SparseGF2Matrix(n, n, a), SparseGF2Matrix(n, k, b)

            X = gf2.solve_multiple_rhs(A, B)
            columns = [gf2.solve(A, [b[i][j] for i in range(n)]) for j in range(k)]
            every_column_solvable = all(c is not None for c in columns)

            assert (X is not None) == every_column_solvable
            if X is not None:
                assert multiply(A, X).to_dense() == b
                batched_solved += 1
            if every_column_solvable:
                per_column_solved += 1

        # Guard against a run where nothing was solvable and the test was vacuous.
        assert batched_solved > 0 and per_column_solved > 0


# ---------------------------------------------------------------------------
# algebraic invariants
# ---------------------------------------------------------------------------
class TestAlgebraicInvariants:
    @pytest.mark.unit
    def test_inverse_round_trip(self):
        rng = random.Random(23)
        checked = 0
        for _ in range(60):
            n = rng.randint(1, 8)
            data = [[rng.getrandbits(1) for _ in range(n)] for _ in range(n)]
            A = SparseGF2Matrix(n, n, data)
            inv = gf2.inverse(A)
            assert (inv is None) == (not gf2.is_invertible(A))
            if inv is not None:
                identity = gf2.identity(n).to_dense()
                assert multiply(A, inv).to_dense() == identity
                assert multiply(inv, A).to_dense() == identity
                checked += 1
        assert checked > 0, "no invertible matrix was drawn; the test proved nothing"

    @pytest.mark.unit
    def test_matrix_power_matches_repeated_multiplication(self):
        data = random_matrix(9, 9, 0.5, seed=29)
        A = SparseGF2Matrix(9, 9, data)
        expected = gf2.identity(9)
        for k in range(6):
            assert gf2.matrix_power(A, k).to_dense() == expected.to_dense(), f"k={k}"
            expected = multiply(expected, A)

    @pytest.mark.unit
    def test_lu_reconstructs_the_permuted_matrix(self):
        """A[perm] == L @ U, for every random square matrix.

        The previous implementation swapped rows of U without recording the
        permutation, so it rebuilt A in only about half of random cases.
        """
        rng = random.Random(31)
        for _ in range(80):
            n = rng.randint(1, 7)
            data = [[rng.getrandbits(1) for _ in range(n)] for _ in range(n)]
            L, U, perm = lu_decomposition(SparseGF2Matrix(n, n, data))

            assert sorted(perm) == list(range(n)), "perm must be a permutation"
            assert multiply(L, U).to_dense() == [data[i] for i in perm]

            lower, upper = L.to_dense(), U.to_dense()
            assert all(lower[i][i] == 1 for i in range(n)), "L diagonal must be unit"
            assert all(lower[i][j] == 0 for i in range(n) for j in range(n) if j > i)
            assert all(upper[i][j] == 0 for i in range(n) for j in range(n) if j < i)

    @pytest.mark.unit
    def test_transpose_is_an_involution_on_non_square_matrices(self):
        data = random_matrix(17, 43, 0.35, seed=37)
        A = SparseGF2Matrix(17, 43, data)
        assert transpose(A).rows == 43 and transpose(A).cols == 17
        assert transpose(transpose(A)).to_dense() == data
        assert transpose(A).to_dense() == np.array(data).T.tolist()

    @pytest.mark.unit
    def test_image_and_kernel_dimensions_satisfy_rank_nullity(self):
        for seed, (rows, cols) in enumerate([(12, 18), (18, 12), (15, 15)]):
            data = random_matrix(rows, cols, 0.4, seed=seed + 41)
            A = SparseGF2Matrix(rows, cols, data)
            r = rank(A)
            assert len(gf2.image(A)) == r, "image basis size must equal the rank"
            assert len(gf2.kernel(A)) == cols - r, "rank-nullity must hold"
            assert gf2.rank_nullity_theorem(A) == (r, cols - r, cols)

    @pytest.mark.unit
    def test_add_is_its_own_inverse(self):
        a = random_matrix(11, 13, 0.5, seed=43)
        b = random_matrix(11, 13, 0.5, seed=47)
        A, B = SparseGF2Matrix(11, 13, a), SparseGF2Matrix(11, 13, b)
        assert gf2.add(gf2.add(A, B), B).to_dense() == a
        assert gf2.add(A, B).to_dense() == gf2.add(B, A).to_dense()


# ---------------------------------------------------------------------------
# exported-but-thinly-covered surface
# ---------------------------------------------------------------------------
class TestNormAndConditionNumber:
    """These are exported, so their documented behaviour is a promise.

    Neither is a norm or a condition number in the usual analytic sense - GF(2)
    has no ordering and no positive-definite form - so the tests pin what the
    functions actually return rather than implying more.
    """

    @pytest.mark.unit
    def test_hamming_norm_counts_set_bits(self):
        A = SparseGF2Matrix(3, 3, [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        assert gf2.matrix_norm(A, "hamming") == 6.0
        assert gf2.matrix_norm(SparseGF2Matrix(4, 4), "hamming") == 0.0

    @pytest.mark.unit
    def test_rank_norm_equals_rank(self):
        A = SparseGF2Matrix(3, 3, [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        assert gf2.matrix_norm(A, "rank") == float(rank(A))

    @pytest.mark.unit
    def test_spectral_norm_is_the_rank_of_the_gram_matrix(self):
        A = SparseGF2Matrix(3, 3, [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        assert gf2.matrix_norm(A, "spectral") == float(rank(multiply(transpose(A), A)))

    @pytest.mark.unit
    def test_unknown_norm_type_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown norm type"):
            gf2.matrix_norm(gf2.identity(2), "frobenius")

    @pytest.mark.unit
    def test_condition_number_is_one_or_infinite(self):
        assert gf2.condition_number(gf2.identity(4)) == 1.0
        assert gf2.condition_number(SparseGF2Matrix(2, 2, [[1, 1], [1, 1]])) == float("inf")
        with pytest.raises(ValueError):
            gf2.condition_number(SparseGF2Matrix(2, 3, [[1, 0, 1], [0, 1, 0]]))


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------
class TestInputValidation:
    @pytest.mark.unit
    def test_dimension_mismatches_raise(self):
        A = SparseGF2Matrix(2, 3, [[1, 0, 1], [0, 1, 0]])
        B = SparseGF2Matrix(3, 2, [[1, 0], [0, 1], [1, 1]])
        with pytest.raises(ValueError, match="dimensions must match"):
            gf2.add(A, B)
        with pytest.raises(ValueError, match="Inner dimensions"):
            multiply(A, A)
        with pytest.raises(ValueError, match="must be square"):
            gf2.det(A)
        with pytest.raises(ValueError, match="must be square"):
            gf2.trace(A)
        with pytest.raises(ValueError, match="must be square"):
            lu_decomposition(A)
        with pytest.raises(ValueError):
            gf2.solve(A, [1, 0, 1])  # len(b) != A.rows

    @pytest.mark.unit
    def test_rank_on_packed_rows_requires_a_column_count(self):
        with pytest.raises(ValueError, match="n_cols required"):
            rank([0b101, 0b011])

    @pytest.mark.unit
    def test_is_invertible_is_false_for_non_square(self):
        assert gf2.is_invertible(SparseGF2Matrix(2, 3, [[1, 0, 1], [0, 1, 0]])) is False


# ---------------------------------------------------------------------------
# the other exported matrix type
# ---------------------------------------------------------------------------
class TestDenseMatrixIsAFirstClassInput:
    """DenseGF2Matrix is exported and every core signature accepts it.

    Nothing exercised it through the core operations, so the fallback branches
    that handle a matrix without the sparse type's cached row accessors were
    never run. Each operation is checked against the sparse result on the same
    data rather than against a hand-written expectation.
    """

    @staticmethod
    def _pair(n=12, density=0.5, seed=3):
        data = random_matrix(n, n, density, seed)
        return gf2.DenseGF2Matrix(n, n, data), SparseGF2Matrix(n, n, data), data

    @pytest.mark.unit
    def test_scalar_results_match_the_sparse_type(self):
        dense, sparse, _ = self._pair()
        assert rank(dense) == rank(sparse)
        assert gf2.det(dense) == gf2.det(sparse)
        assert gf2.trace(dense) == gf2.trace(sparse)
        assert gf2.is_invertible(dense) == gf2.is_invertible(sparse)
        assert gf2.matrix_norm(dense, "hamming") == gf2.matrix_norm(sparse, "hamming")

    @pytest.mark.unit
    def test_matrix_results_match_the_sparse_type(self):
        dense, sparse, data = self._pair()
        assert gf2.transpose(dense).to_dense() == gf2.transpose(sparse).to_dense()
        assert gf2.add(dense, dense).to_dense() == gf2.add(sparse, sparse).to_dense()
        assert gf2.multiply(dense, dense).to_dense() == gf2.multiply(sparse, sparse).to_dense()
        assert gf2.reduced_row_echelon_form(dense)[1] == gf2.reduced_row_echelon_form(sparse)[1]
        assert gf2.characteristic_polynomial(dense) == gf2.characteristic_polynomial(sparse)

    @pytest.mark.unit
    def test_solver_results_match_the_sparse_type(self):
        dense, sparse, data = self._pair()
        n = dense.rows
        assert gf2.nullspace(dense) == gf2.nullspace(sparse)
        assert gf2.solve(dense, [0] * n) == gf2.solve(sparse, [0] * n)
        inverted_dense, inverted_sparse = gf2.inverse(dense), gf2.inverse(sparse)
        assert (inverted_dense is None) == (inverted_sparse is None)
        if inverted_dense is not None:
            assert inverted_dense.to_dense() == inverted_sparse.to_dense()

    @pytest.mark.unit
    def test_bit_access_round_trips_across_word_boundaries(self):
        # cols > 64 so more than one 64-bit word per row is in play
        data = random_matrix(5, 200, 0.3, seed=53)
        M = gf2.DenseGF2Matrix(5, 200, data)
        for i in range(5):
            for j in range(200):
                assert M.get_bit(i, j) == data[i][j], f"({i},{j})"
        assert M.get_row_bitwise(0) == SparseGF2Matrix(5, 200, data).get_row_bitwise(0)

    @pytest.mark.unit
    def test_memory_usage_counts_set_bits_exactly(self):
        data = random_matrix(6, 130, 0.25, seed=59)
        stats = gf2.DenseGF2Matrix(6, 130, data).memory_usage()
        assert stats.nnz == sum(sum(row) for row in data)
        assert stats.memory_bytes > 0


# ---------------------------------------------------------------------------
# exported helpers with no test of their own
# ---------------------------------------------------------------------------
class TestRemainingExports:
    @pytest.mark.unit
    def test_nullspace_bitwise_rejects_a_full_rank_system(self):
        """A full-rank square system has only the trivial null space."""
        with pytest.raises(ValueError, match="No free variable"):
            gf2.nullspace_bitwise(gf2.identity(6))

    @pytest.mark.unit
    def test_benchmark_solver_reports_ordered_statistics(self):
        A = gf2.identity(8)
        stats = gf2.benchmark_solver(A, [1] * 8, num_trials=5)
        assert stats["trials"] == 5
        assert stats["min_time"] <= stats["median_time"] <= stats["max_time"]
        assert stats["min_time"] >= 0
        assert stats["total_time"] >= stats["max_time"]

    @pytest.mark.unit
    def test_iterative_refinement_returns_an_exact_solution(self):
        data = random_matrix(10, 10, 0.5, seed=61)
        A = SparseGF2Matrix(10, 10, data)
        b = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
        x, iterations = gf2.iterative_refinement(A, b, max_iterations=10)
        if x is not None and gf2.is_invertible(A):
            # GF(2) arithmetic is exact, so one corrective solve suffices and the
            # next pass only confirms a zero residual. What matters is that it
            # converges rather than exhausting max_iterations.
            assert iterations < 10, "did not converge; it ran out of iterations"
            assert iterations <= 1, f"exact arithmetic should settle at once, took {iterations}"
            assert [sum(r & v for r, v in zip(row, x, strict=True)) % 2 for row in data] == b

    @pytest.mark.unit
    def test_sparse_stats_is_exported_and_describes_storage(self):
        M = SparseGF2Matrix(50, 50, random_matrix(50, 50, 0.02, seed=67))
        stats = M.memory_usage()
        assert isinstance(stats, gf2.SparseStats)
        assert stats.nnz == M.nnz
        assert 0 <= stats.density <= 1
        assert stats.compression_ratio > 1, "a 2%-dense matrix should beat byte-per-cell"


# ---------------------------------------------------------------------------
# structured results
# ---------------------------------------------------------------------------
class TestStructuredResults:
    """Operations returning several values return named, unpackable results.

    Each is a NamedTuple, so positional unpacking keeps working exactly as a
    plain tuple would; the names exist so a reader does not have to remember
    what the third element of an LU result is.
    """

    @pytest.mark.unit
    def test_results_unpack_positionally_like_plain_tuples(self):
        A = SparseGF2Matrix(4, 4, random_matrix(4, 4, 0.5, seed=71))

        L, U, perm = gf2.lu_decomposition(A)
        again = gf2.lu_decomposition(A)
        assert (L, U, perm) == tuple(again)

        reduced, pivots = gf2.reduced_row_echelon_form(A)
        assert pivots == gf2.reduced_row_echelon_form(A).pivot_columns

        bits, seconds = gf2.nullspace_fast([[1, 0, 1], [0, 1, 1]])
        assert isinstance(bits, str) and isinstance(seconds, float)

        r, nullity, cols = gf2.rank_nullity_theorem(A)
        assert r + nullity == cols

    @pytest.mark.unit
    def test_result_fields_are_named(self):
        A = SparseGF2Matrix(5, 7, random_matrix(5, 7, 0.4, seed=73))

        assert gf2.lu_decomposition(SparseGF2Matrix(3, 3))._fields == ("L", "U", "perm")
        assert gf2.reduced_row_echelon_form(A)._fields == ("matrix", "pivot_columns")
        assert gf2.nullspace_fast([[1, 1]])._fields == ("bits", "seconds")

        rn = gf2.rank_nullity_theorem(A)
        assert rn._fields == ("rank", "nullity", "columns")
        assert rn.rank == rank(A) and rn.columns == 7

    @pytest.mark.unit
    def test_rref_returns_only_the_non_zero_rows(self):
        # Two identical rows: rank 1, so one reduced row and one pivot.
        A = SparseGF2Matrix(2, 3, [[1, 0, 1], [1, 0, 1]])
        reduced = gf2.reduced_row_echelon_form(A)
        assert reduced.matrix.rows == rank(A) == 1
        assert len(reduced.pivot_columns) == reduced.matrix.rows

    @pytest.mark.unit
    def test_permutation_matrix_gives_the_p_a_equals_l_u_form(self):
        """P @ A == L @ U, the form that composes with the other operations."""
        rng = random.Random(79)
        for _ in range(50):
            n = rng.randint(1, 7)
            data = [[rng.getrandbits(1) for _ in range(n)] for _ in range(n)]
            A = SparseGF2Matrix(n, n, data)
            result = gf2.lu_decomposition(A)
            P = result.permutation_matrix()

            assert multiply(P, A).to_dense() == multiply(result.L, result.U).to_dense()
            assert multiply(result.L, result.U).to_dense() == [data[i] for i in result.perm]

            # P must actually be a permutation: one 1 per row and per column.
            rows_of_p = P.to_dense()
            assert all(sum(row) == 1 for row in rows_of_p)
            assert all(sum(col) == 1 for col in zip(*rows_of_p, strict=True))

    @pytest.mark.unit
    def test_result_types_are_exported(self):
        for name in ("LUDecomposition", "RowEchelonForm", "NullspaceVector", "RankNullity"):
            assert name in gf2.__all__, f"{name} is returned by the API but not exported"
            assert hasattr(gf2, name)


# ---------------------------------------------------------------------------
# equality
# ---------------------------------------------------------------------------
class TestMatrixEquality:
    """``A == B`` compares contents, not identity.

    Without ``__eq__`` the comparison fell through to identity and returned
    False for two matrices holding the same bits - which is what a caller
    writing ``assert result == expected`` would hit first.
    """

    @pytest.mark.unit
    def test_equal_contents_compare_equal_across_storage_formats(self):
        data = random_matrix(6, 70, 0.4, seed=83)
        csr = SparseGF2Matrix(6, 70, data, "csr")
        packed = SparseGF2Matrix(6, 70, data, "bitpacked")
        dense = gf2.DenseGF2Matrix(6, 70, data)

        assert csr == packed, "storage format must not affect equality"
        assert csr == dense, "the two matrix types must compare by content"
        assert dense == packed

    @pytest.mark.unit
    def test_differing_contents_or_shape_compare_unequal(self):
        A = SparseGF2Matrix(2, 3, [[1, 0, 1], [0, 1, 1]])
        assert SparseGF2Matrix(2, 3, [[1, 0, 1], [0, 1, 0]]) != A
        assert SparseGF2Matrix(3, 3, [[1, 0, 1], [0, 1, 1], [0, 0, 0]]) != A
        assert SparseGF2Matrix(2, 4, [[1, 0, 1, 0], [0, 1, 1, 0]]) != A

    @pytest.mark.unit
    def test_comparison_with_an_unrelated_object_is_false_not_an_error(self):
        A = SparseGF2Matrix(2, 2, [[1, 0], [0, 1]])
        assert A != "not a matrix"
        assert A is not None
        assert (A == 42) is False

    @pytest.mark.unit
    def test_matrices_are_unhashable_because_they_are_mutable(self):
        """set_bit mutates in place, so a stable hash cannot exist."""
        A = SparseGF2Matrix(2, 2, [[1, 0], [0, 1]])
        with pytest.raises(TypeError):
            hash(A)

    @pytest.mark.unit
    def test_mutation_changes_equality(self):
        A = SparseGF2Matrix(3, 3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        B = SparseGF2Matrix(3, 3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert A == B
        B.set_bit(0, 2)
        assert A != B
        B.clear_bit(0, 2)
        assert A == B

    @pytest.mark.unit
    def test_operations_can_be_compared_directly(self):
        """The point of __eq__: assertions read the way people write them."""
        A = SparseGF2Matrix(4, 4, random_matrix(4, 4, 0.5, seed=89))
        assert transpose(transpose(A)) == A
        assert gf2.add(gf2.add(A, A), A) == A
        assert gf2.matrix_power(A, 1) == A
        assert multiply(gf2.identity(4), A) == A
