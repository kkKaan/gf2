"""
Regression tests for the defects found in the performance/correctness review.

Each test names the specific bug it pins down. These are the behaviours that
were wrong *silently* - the suite passed while they were broken - so they are
asserted directly rather than through a property that might drift.
"""

import random
from itertools import permutations

import numpy as np
import pytest

from gf2.core import (
    _PACKED_ELIM_MIN_DIM,
    _pack_rows_u64,
    _rank_bitwise,
    _rank_packed,
    _rows_of,
    add,
    characteristic_polynomial,
    det,
    gaussian_elimination_inplace,
    matrix_power,
    minimal_polynomial,
    multiply,
    rank,
    transpose,
)
from gf2.generators import (
    hypergraph_product,
    ldpc_matrix,
    ones,
    random_regular,
    random_sparse,
    surface_code_matrix,
)
from gf2.solvers import iterative_refinement, nullspace, nullspace_fast, solve, solve_multiple_rhs
from gf2.sparse import SparseGF2Matrix, create_sparse_matrix


# --------------------------------------------------------------------------
# storage: mutation used to corrupt or silently drop writes
# --------------------------------------------------------------------------
class TestStorageMutation:
    @pytest.mark.unit
    def test_set_bit_on_empty_format_is_not_dropped(self):
        """set_bit went through _from_dense, which has no branch for "empty"."""
        M = create_sparse_matrix(3, 3)
        assert M.format == "empty"
        M.set_bit(0, 0)
        assert M.get_bit(0, 0) == 1
        assert M.to_dense() == [[1, 0, 0], [0, 0, 0], [0, 0, 0]]

    @pytest.mark.unit
    def test_set_bit_invalidates_the_packed_row_cache(self):
        """The CSR rebuild left _packed_rows_cache stale, so every algorithm
        (which reads rows through it) saw the pre-write matrix."""
        M = SparseGF2Matrix(3, 3, ([0], [1]))
        M.get_all_rows_bitwise()  # warm the cache
        M.set_bit(0, 0)

        assert M.get_bit(0, 0) == 1
        assert M.get_row_bitwise(0) == 0b11
        assert M.to_dense()[0] == [1, 1, 0]
        assert rank(M) == 1

    @pytest.mark.unit
    def test_clear_bit_and_set_zero(self):
        M = SparseGF2Matrix(2, 2, ([0, 1], [0, 1]))
        M.clear_bit(0, 0)
        assert M.get_bit(0, 0) == 0
        M.set(1, 1, 0)
        assert M.get_bit(1, 1) == 0
        assert M.to_dense() == [[0, 0], [0, 0]]

    @pytest.mark.unit
    def test_duplicate_coordinates_counted_once(self):
        """Duplicates were stored twice, so nnz and every density-driven
        decision over-reported the number of entries."""
        M = SparseGF2Matrix(2, 2, ([0, 0], [1, 1]))
        assert M.nnz == 1
        assert M.to_dense() == [[0, 1], [0, 0]]

    @pytest.mark.unit
    def test_packed_rows_masked_to_column_count(self):
        """Bits past `cols` used to survive in the cache, making
        get_row_bitwise and get_bit disagree and inflating nnz."""
        M = SparseGF2Matrix(2, 3)
        M.set_from_packed_rows([0b1111, 0])
        assert M.get_row_bitwise(0) == 0b111
        assert M.nnz == 3
        assert M.to_dense()[0] == [1, 1, 1]

    @pytest.mark.unit
    def test_set_from_packed_rows_rejects_wrong_length(self):
        M = SparseGF2Matrix(2, 3)
        with pytest.raises(ValueError):
            M.set_from_packed_rows([0, 0, 0])

    @pytest.mark.unit
    def test_zero_dimension_does_not_divide_by_zero(self):
        assert SparseGF2Matrix(0, 3, ([], [])).rows == 0
        assert SparseGF2Matrix(3, 0, ([], [])).cols == 0

    @pytest.mark.unit
    def test_coordinates_are_bounds_checked(self):
        with pytest.raises(IndexError):
            SparseGF2Matrix(2, 2, ([5], [0]))

    @pytest.mark.unit
    def test_csr_rows_match_bitpacked_rows(self):
        """CSR row access had no cache and was rebuilt per call; the two
        formats must still agree bit for bit."""
        rng = random.Random(4)
        data = [[1 if rng.random() < 0.3 else 0 for _ in range(70)] for _ in range(20)]
        csr = SparseGF2Matrix(20, 70, data, "csr")
        packed = SparseGF2Matrix(20, 70, data, "bitpacked")
        assert csr.get_all_rows_bitwise() == packed.get_all_rows_bitwise()
        assert csr.to_dense() == packed.to_dense() == [[v & 1 for v in r] for r in data]


# --------------------------------------------------------------------------
# solvers
# --------------------------------------------------------------------------
class TestSolverRegressions:
    @pytest.mark.unit
    def test_nullspace_fast_honours_include_packing_time(self):
        """The flag was accepted and then ignored: timing always started after
        packing, so every reported figure silently excluded it."""
        matrix = [[1 if (i + j) % 3 else 0 for j in range(200)] for i in range(150)]
        _, with_packing = nullspace_fast(matrix, include_packing_time=True)
        _, without_packing = nullspace_fast(matrix, include_packing_time=False)
        assert with_packing > without_packing

    @pytest.mark.unit
    def test_iterative_refinement_does_not_mutate_a_numpy_rhs(self):
        """`b[:]` on an ndarray is a view, so the residual was XOR-ed straight
        into the caller's array."""
        from gf2.generators import identity

        A = identity(3)
        b = np.array([1, 0, 1], dtype=int)
        before = b.copy()
        iterative_refinement(A, b)
        np.testing.assert_array_equal(b, before)

    @pytest.mark.unit
    def test_solve_matches_brute_force(self):
        rng = random.Random(7)
        for _ in range(150):
            m, n = rng.randint(1, 5), rng.randint(1, 5)
            data = [[rng.getrandbits(1) for _ in range(n)] for _ in range(m)]
            A = SparseGF2Matrix(m, n, data)
            b = [rng.getrandbits(1) for _ in range(m)]

            exists = any(
                all(sum(data[i][j] & ((c >> j) & 1) for j in range(n)) % 2 == b[i] for i in range(m))
                for c in range(1 << n)
            )
            x = solve(A, b)
            assert (x is not None) == exists
            if x is not None:
                assert all(sum(data[i][j] & x[j] for j in range(n)) % 2 == b[i] for i in range(m))

    @pytest.mark.unit
    def test_nullspace_basis_is_correct_when_rank_deficient(self):
        """The O(free * pivots * cols) back-substitution only showed up with
        many free columns; check the answers as well as the count."""
        rng = random.Random(11)
        base = [[rng.getrandbits(1) for _ in range(40)] for _ in range(20)]
        data = base + base  # rank <= 20 out of 40 rows
        A = SparseGF2Matrix(40, 40, data)

        basis = nullspace(A)
        assert len(basis) == 40 - rank(A)
        for vector in basis:
            assert any(vector)
            for row in data:
                assert sum(r & v for r, v in zip(row, vector, strict=True)) % 2 == 0

    @pytest.mark.unit
    def test_solve_multiple_rhs_matches_column_by_column(self):
        rng = random.Random(13)
        n = 12
        a_data = [[rng.getrandbits(1) for _ in range(n)] for _ in range(n)]
        A = SparseGF2Matrix(n, n, a_data)
        B = SparseGF2Matrix(n, 5, [[rng.getrandbits(1) for _ in range(5)] for _ in range(n)])

        X = solve_multiple_rhs(A, B)
        if X is not None:
            assert multiply(A, X).to_dense() == B.to_dense()


# --------------------------------------------------------------------------
# core arithmetic
# --------------------------------------------------------------------------
class TestCoreRegressions:
    @pytest.mark.unit
    def test_multiply_matches_reference_across_densities(self):
        """Covers both the blocked (dense) and row-XOR (sparse) branches."""
        rng = random.Random(17)
        for n, density in ((7, 0.5), (64, 0.5), (150, 0.02), (150, 0.9)):
            a = [[1 if rng.random() < density else 0 for _ in range(n)] for _ in range(n)]
            b = [[1 if rng.random() < density else 0 for _ in range(n)] for _ in range(n)]
            got = multiply(SparseGF2Matrix(n, n, a), SparseGF2Matrix(n, n, b)).to_dense()
            expected = ((np.array(a, dtype=np.uint8) @ np.array(b, dtype=np.uint8)) & 1).tolist()
            assert got == expected, f"n={n} density={density}"

    @pytest.mark.unit
    def test_multiply_non_square(self):
        a = [[1, 0, 1], [0, 1, 1]]
        b = [[1, 1], [0, 1], [1, 0]]
        got = multiply(SparseGF2Matrix(2, 3, a), SparseGF2Matrix(3, 2, b)).to_dense()
        expected = ((np.array(a, dtype=np.uint8) @ np.array(b, dtype=np.uint8)) & 1).tolist()
        assert got == expected

    @pytest.mark.unit
    def test_transpose_round_trip(self):
        rng = random.Random(19)
        data = [[rng.getrandbits(1) for _ in range(37)] for _ in range(23)]
        A = SparseGF2Matrix(23, 37, data)
        assert transpose(A).to_dense() == np.array(data).T.tolist()
        assert transpose(transpose(A)).to_dense() == [[v & 1 for v in r] for r in data]

    @pytest.mark.unit
    def test_packed_and_bigint_rank_agree(self):
        """rank() switches implementation at _PACKED_ELIM_MIN_DIM; both sides
        of the switch must give the same answer."""
        rng = random.Random(23)
        n = _PACKED_ELIM_MIN_DIM + 40
        for density in (0.02, 0.5, 0.95):
            rows = [sum(1 << j for j in range(n) if rng.random() < density) for _ in range(n)]
            assert _rank_packed(_pack_rows_u64(rows, n), n) == _rank_bitwise_forced(rows, n)

    @pytest.mark.unit
    def test_det_is_one_exactly_when_full_rank(self):
        rng = random.Random(29)
        for _ in range(60):
            n = rng.randint(1, 6)
            data = [[rng.getrandbits(1) for _ in range(n)] for _ in range(n)]
            A = SparseGF2Matrix(n, n, data)
            assert det(A) == (1 if rank(A) == n else 0)


def _rank_bitwise_forced(rows, n_cols):
    """_rank_bitwise with the vectorised shortcut disabled."""
    from gf2 import core

    saved = core._PACKED_ELIM_MIN_DIM
    core._PACKED_ELIM_MIN_DIM = 10**9
    try:
        return _rank_bitwise(rows, n_cols)
    finally:
        core._PACKED_ELIM_MIN_DIM = saved


# --------------------------------------------------------------------------
# polynomials: both were mathematically wrong / unusable
# --------------------------------------------------------------------------
def _charpoly_reference(data, n):
    """det(xI + A) over GF(2)[x] by Leibniz expansion. Polynomials are bitmasks."""
    entries = [
        [(0b10 ^ (data[i][i] & 1)) if i == j else (data[i][j] & 1) for j in range(n)] for i in range(n)
    ]

    def pmul(a, b):
        out = 0
        while b:
            low = b & -b
            out ^= a << (low.bit_length() - 1)
            b ^= low
        return out

    total = 0
    for perm in permutations(range(n)):
        product = 1
        for i, j in enumerate(perm):
            product = pmul(product, entries[i][j])
            if product == 0:
                break
        total ^= product
    return [(total >> i) & 1 for i in range(n + 1)]


class TestPolynomials:
    @pytest.mark.unit
    def test_characteristic_polynomial_matches_exact_determinant(self):
        """The old version returned a stub - leading 1, trace, det, and zeros
        in between - which is wrong for every n >= 3."""
        rng = random.Random(31)
        for _ in range(80):
            n = rng.randint(1, 5)
            data = [[rng.getrandbits(1) for _ in range(n)] for _ in range(n)]
            A = SparseGF2Matrix(n, n, data)
            assert characteristic_polynomial(A) == _charpoly_reference(data, n)

    @pytest.mark.unit
    def test_characteristic_polynomial_of_a_companion_matrix(self):
        """Companion matrix of x^3 + x + 1; the stub returned x^3 + 1."""
        A = SparseGF2Matrix(3, 3, [[0, 1, 0], [0, 0, 1], [1, 1, 0]])
        assert characteristic_polynomial(A) == [1, 1, 0, 1]

    @pytest.mark.unit
    def test_minimal_polynomial_is_monic_annihilating_and_minimal(self):
        rng = random.Random(37)
        for _ in range(40):
            n = rng.randint(1, 5)
            data = [[rng.getrandbits(1) for _ in range(n)] for _ in range(n)]
            A = SparseGF2Matrix(n, n, data)
            mp = minimal_polynomial(A)

            assert mp[-1] == 1, "not monic"
            assert _annihilates(A, mp, n), "does not annihilate A"

            degree = len(mp) - 1
            for lower in range(degree):
                for mask in range(1 << lower):
                    candidate = [(mask >> i) & 1 for i in range(lower)] + [1]
                    assert not _annihilates(A, candidate, n), f"degree {lower} also works"

    @pytest.mark.unit
    def test_minimal_polynomial_is_polynomial_time(self):
        """The old enumeration was O(2^n); n=32 was unreachable."""
        rng = random.Random(41)
        n = 32
        A = SparseGF2Matrix(n, n, [[rng.getrandbits(1) for _ in range(n)] for _ in range(n)])
        mp = minimal_polynomial(A)
        assert mp[-1] == 1
        assert len(mp) - 1 <= n
        assert _annihilates(A, mp, n)


def _annihilates(A, coeffs, n):
    identity = SparseGF2Matrix(n, n, [[1 if r == c else 0 for c in range(n)] for r in range(n)])
    acc = SparseGF2Matrix(n, n)
    for i, c in enumerate(coeffs):
        if c:
            acc = add(acc, identity if i == 0 else matrix_power(A, i))
    return not any(_rows_of(acc))


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------
class TestGeneratorRegressions:
    @pytest.mark.unit
    def test_seeded_generators_do_not_touch_the_global_rng(self):
        """They called random.seed(), resetting the caller's random stream."""
        random.seed(12345)
        expected = [random.random() for _ in range(3)]

        random.seed(12345)
        random_sparse(10, 10, 0.3, seed=99)
        random_regular(4, 4, 2, seed=99)
        ldpc_matrix(4, 8, 4, method="random", seed=99)
        assert [random.random() for _ in range(3)] == expected

    @pytest.mark.unit
    def test_random_regular_has_exact_weights(self):
        """A plain shuffle could pair a row with the same column twice, which
        under set semantics quietly produced an under-weight row."""
        for seed in range(15):
            H = random_regular(12, 18, 3, seed=seed)
            rows = H.get_all_rows_bitwise()
            assert all(row.bit_count() == 3 for row in rows)

            col_weights = [0] * 18
            for row in rows:
                remaining = row
                while remaining:
                    low = remaining & -remaining
                    col_weights[low.bit_length() - 1] += 1
                    remaining ^= low
            assert all(w == 2 for w in col_weights)

    @pytest.mark.unit
    def test_ones_is_all_ones(self):
        M = ones(3, 5)
        assert M.to_dense() == [[1] * 5] * 3
        assert M.nnz == 15

    @pytest.mark.unit
    def test_random_sparse_density_and_bounds(self):
        M = random_sparse(50, 40, 0.25, seed=3)
        assert M.nnz == int(50 * 40 * 0.25)
        assert M.rows == 50 and M.cols == 40
        with pytest.raises(ValueError):
            random_sparse(5, 5, 1.5)

    @pytest.mark.unit
    def test_random_sparse_is_reproducible(self):
        a = random_sparse(30, 30, 0.2, seed=8).get_all_rows_bitwise()
        b = random_sparse(30, 30, 0.2, seed=8).get_all_rows_bitwise()
        assert a == b

    @pytest.mark.unit
    def test_ldpc_progressive_is_regular_and_fast(self):
        H = ldpc_matrix(50, 100, 4, method="progressive")
        rows = H.get_all_rows_bitwise()
        assert all(row.bit_count() == 4 for row in rows)

    @pytest.mark.unit
    def test_hypergraph_product_z_block_is_not_empty(self):
        """H_z used to come back all zero, which made every commutation test
        pass vacuously."""
        from gf2.generators import hamming_matrix

        H_x, H_z = hypergraph_product(hamming_matrix(3), hamming_matrix(2))
        assert H_z.nnz > 0
        assert H_x.nnz > 0
        assert not any(multiply(H_x, transpose(H_z)).get_all_rows_bitwise())

    @pytest.mark.unit
    def test_surface_code_is_a_real_css_code(self):
        for distance in (3, 5):
            H_x, H_z = surface_code_matrix(distance)
            n = distance**2 + (distance - 1) ** 2
            assert H_x.cols == H_z.cols == n
            assert not any(multiply(H_x, transpose(H_z)).get_all_rows_bitwise())
            assert n - rank(H_x) - rank(H_z) == 1


# --------------------------------------------------------------------------
# elimination helper
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_gaussian_elimination_produces_true_rref():
    rng = random.Random(43)
    n = 24
    rows = [sum(1 << j for j in range(n) if rng.random() < 0.4) for _ in range(n)]
    reduced, pivot_cols = gaussian_elimination_inplace(rows[:], n)

    assert len(reduced) == len(pivot_cols)
    for i, pivot in enumerate(pivot_cols):
        assert (reduced[i] >> pivot) & 1, "pivot bit not set"
        for k, other in enumerate(pivot_cols):
            if k != i:
                assert not (reduced[i] >> other) & 1, "not fully reduced"
        assert reduced[i] & ((1 << pivot) - 1) == 0, "not in echelon form"
