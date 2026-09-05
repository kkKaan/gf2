"""
Core GF(2) Linear Algebra Operations
===================================

Comprehensive suite of matrix operations over GF(2) with optimized algorithms
for sparse and dense representations. All operations use bitwise arithmetic
for maximum performance.

Operations:
- Basic arithmetic: add, multiply, transpose
- Properties: rank, determinant, trace
- Decompositions: LU, QR (modified for GF(2))
- System solving: Ax=b, matrix inversion
- Specialized: nullspace, kernel, image
"""

import numpy as np

from .sparse import DenseGF2Matrix, SparseGF2Matrix, _int_from_words, _set_bit_positions

# Block width for the Method of Four Russians. 8 keeps the 2**k lookup table
# at 256 rows, which stays in L1 for every matrix width we care about, and
# measured fastest across n = 256..2048 (k=6 and k=10 were both slower).
_M4RM_BLOCK = 8

# Below this many result bits the NumPy round-trip costs more than it saves,
# so the big-integer path wins.
_M4RM_MIN_CELLS = 1 << 14

# Elimination crossover. Below this the big-integer loop wins (no NumPy call
# overhead per column); above it, selecting the rows to XOR with a vectorised
# flatnonzero wins and the gap widens with n. Measured on this codebase:
# n=128 big-int 2.3x faster, n=384 a tie, n=1536 NumPy 1.8x faster.
_PACKED_ELIM_MIN_DIM = 384


def _popcount_parity(x: int) -> int:
    """Return parity of number of set bits in x."""
    return x.bit_count() & 1


def _rows_of(A: SparseGF2Matrix | DenseGF2Matrix) -> list[int]:
    """Packed rows of ``A``, reusing the matrix's own cache when it has one.

    Calling ``get_row_bitwise`` in a loop re-materialises a CSR row from its
    column indices on every call; on a 300x300 matrix that made a full row
    sweep 238x more expensive than reading the cached form once.
    """
    getter = getattr(A, "get_all_rows_bitwise", None)
    if getter is not None:
        return getter()
    return [A.get_row_bitwise(i) for i in range(A.rows)]


def _packed_u64(A: SparseGF2Matrix | DenseGF2Matrix) -> np.ndarray:
    """``A`` as a (rows, words) little-endian uint64 array."""
    accessor = getattr(A, "packed_u64", None)
    if accessor is not None:
        return accessor()
    words = max((A.cols + 63) // 64, 1)
    out = np.zeros((A.rows, words), dtype=np.uint64)
    nbytes = words * 8
    for i, value in enumerate(_rows_of(A)):
        out[i] = np.frombuffer(int(value).to_bytes(nbytes, "little"), dtype=np.uint64)
    return out


def _m4rm(A_bits: np.ndarray, B_packed: np.ndarray, k: int = _M4RM_BLOCK) -> np.ndarray:
    """GF(2) product via the Method of Four Russians.

    ``A_bits`` is (m, n) uint8 0/1, ``B_packed`` is (n, w) packed uint64 rows;
    the result is (m, w) packed uint64 rows.

    Splitting A's columns into blocks of ``k`` lets one table of 2**k
    precombined B-rows serve every row of A, so the XOR count drops from
    O(m*n/2) to O(m*n/k + n/k * 2**k). Every step is a whole-array NumPy op,
    which is what makes it beat the scalar loop by ~900x at n=256.
    """
    m, n = A_bits.shape
    w = B_packed.shape[1]
    n_blocks = (n + k - 1) // k

    padded = np.zeros((m, n_blocks * k), dtype=np.uint8)
    padded[:, :n] = A_bits
    weights = 1 << np.arange(k, dtype=np.uint16)
    codes = (padded.reshape(m, n_blocks, k).astype(np.uint16) * weights).sum(axis=2).astype(np.uint8)

    out = np.zeros((m, w), dtype=np.uint64)
    block_rows = np.zeros((k, w), dtype=np.uint64)
    table = np.zeros((1 << k, w), dtype=np.uint64)

    for block in range(n_blocks):
        base = block * k
        high = min(k, n - base)
        block_rows[:] = 0
        block_rows[:high] = B_packed[base : base + high]

        # Build the table by doubling: k vector ops instead of 2**k scalar ones.
        table[0] = 0
        size = 1
        for i in range(k):
            table[size : 2 * size] = table[:size] ^ block_rows[i]
            size *= 2

        out ^= table[codes[:, block]]

    return out


def add(
    A: SparseGF2Matrix | DenseGF2Matrix, B: SparseGF2Matrix | DenseGF2Matrix
) -> SparseGF2Matrix | DenseGF2Matrix:
    """
    Add two GF(2) matrices: C = A + B (XOR).

    Args:
        A, B: Input matrices (must have same dimensions)

    Returns:
        Sum matrix in optimal format
    """
    if A.rows != B.rows or A.cols != B.cols:
        raise ValueError("Matrix dimensions must match")

    # Use bitwise XOR for addition in GF(2)
    result = SparseGF2Matrix(A.rows, A.cols)

    # Convert both to packed rows for efficient XOR
    packed_rows = [ra ^ rb for ra, rb in zip(_rows_of(A), _rows_of(B), strict=True)]

    result.set_from_packed_rows(packed_rows)
    return result


def multiply(
    A: SparseGF2Matrix | DenseGF2Matrix, B: SparseGF2Matrix | DenseGF2Matrix
) -> SparseGF2Matrix | DenseGF2Matrix:
    """
    Multiply two GF(2) matrices: C = A * B.

    Uses optimized bitwise operations for matrix multiplication over GF(2).
    """
    if A.cols != B.rows:
        raise ValueError("Inner dimensions must match")

    result = SparseGF2Matrix(A.rows, B.cols)
    if A.rows == 0 or B.cols == 0:
        return result

    a_rows = _rows_of(A)
    b_rows = _rows_of(B)

    # Row i of A*B is the XOR of the B-rows selected by the set bits of A's
    # row i. That replaces the old formulation, which materialised every
    # column of B and then took A.rows * B.cols separate parity dot products.
    nnz_a = sum(row.bit_count() for row in a_rows)
    cells = A.rows * B.cols

    if cells >= _M4RM_MIN_CELLS and nnz_a * 8 > A.rows * A.cols:
        # Dense enough that blocking pays: one shared table per column block.
        a_bits = np.unpackbits(_packed_u64(A).view(np.uint8), axis=1, bitorder="little")[:, : A.cols]
        packed = _m4rm(a_bits, _packed_u64(B))
        result_rows = [_int_from_words(packed[i]) for i in range(A.rows)]
    else:
        # Sparse A: cost is proportional to nnz(A), which beats blocking.
        result_rows = []
        for row_a in a_rows:
            acc = 0
            remaining = row_a
            while remaining:
                low = remaining & -remaining
                acc ^= b_rows[low.bit_length() - 1]
                remaining ^= low
            result_rows.append(acc)

    result.set_from_packed_rows(result_rows)
    return result


def transpose(A: SparseGF2Matrix | DenseGF2Matrix) -> SparseGF2Matrix | DenseGF2Matrix:
    """
    Transpose a GF(2) matrix: A^T.
    """
    # Accumulate each source row straight into the destination rows. Walking
    # only the set bits (via ``value & -value``) costs one pass per non-zero
    # instead of the one-shift-per-column loop this used to run, which was
    # quadratic in the row width.
    transposed = [0] * A.cols
    for i, row_packed in enumerate(_rows_of(A)):
        bit_i = 1 << i
        for j in _set_bit_positions(row_packed):
            transposed[j] |= bit_i

    result = SparseGF2Matrix(A.cols, A.rows)
    result.set_from_packed_rows(transposed)
    return result


def rank(A: SparseGF2Matrix | DenseGF2Matrix | list[int], n_cols: int | None = None) -> int:
    """
    Compute rank of GF(2) matrix using optimized Gaussian elimination.

    Args:
        A: Matrix object or list of packed row integers (fast path)
        n_cols: Number of columns (required if A is list[int])

    Returns:
        Rank of the matrix
    """
    # OPTIMIZATION: Fast path for direct packed input
    if isinstance(A, list):
        if n_cols is None:
            raise ValueError("n_cols required when A is list[int]")
        return _rank_bitwise(A, n_cols)

    # A bit-packed matrix already holds the uint64 buffer the vectorised path
    # wants; going via big integers would pack it a second time for nothing.
    if A.cols >= _PACKED_ELIM_MIN_DIM and A.rows >= _PACKED_ELIM_MIN_DIM:
        return _rank_packed(_packed_u64(A), A.cols)

    return _rank_bitwise(_rows_of(A), A.cols)


def _rank_packed(packed: np.ndarray, n_cols: int) -> int:
    """Rank via vectorised elimination over packed uint64 rows.

    One ``flatnonzero`` finds every row that needs the pivot XOR-ed in, so the
    per-column Python work is constant instead of proportional to the number
    of rows. That is what makes it overtake the big-integer loop past
    ``_PACKED_ELIM_MIN_DIM``.
    """
    A = packed.copy()  # elimination is destructive
    n_rows = A.shape[0]
    rank_count = 0
    for col in range(n_cols):
        if rank_count == n_rows:
            break
        word = col >> 6
        mask = np.uint64(1) << np.uint64(col & 63)

        hits = np.flatnonzero(A[rank_count:, word] & mask)
        if hits.size == 0:
            continue

        pivot_row = rank_count + int(hits[0])
        if pivot_row != rank_count:
            A[[rank_count, pivot_row]] = A[[pivot_row, rank_count]]

        below = rank_count + 1 + np.flatnonzero(A[rank_count + 1 :, word] & mask)
        if below.size:
            A[below] ^= A[rank_count]

        rank_count += 1

    return rank_count


def _pack_rows_u64(rows: list[int], n_cols: int) -> np.ndarray:
    """Big-integer rows -> a (rows, words) uint64 array."""
    words = max((n_cols + 63) // 64, 1)
    nbytes = words * 8
    out = np.empty((len(rows), words), dtype=np.uint64)
    for i, value in enumerate(rows):
        out[i] = np.frombuffer(int(value).to_bytes(nbytes, "little"), dtype=np.uint64)
    return out


def _rank_bitwise(rows: list[int], n_cols: int) -> int:
    """Internal rank computation using bitwise operations."""
    if n_cols >= _PACKED_ELIM_MIN_DIM and len(rows) >= _PACKED_ELIM_MIN_DIM:
        return _rank_packed(_pack_rows_u64(rows, n_cols), n_cols)

    # Copy for in-place elimination
    A = rows[:]
    n_rows = len(A)
    rank_count = 0
    bit = 1

    for _col in range(n_cols):
        if rank_count == n_rows:
            break

        # Test with ``A[i] & bit``. ``(A[i] >> col) & 1`` allocates a shifted
        # copy of the whole row on every probe; masking does not.
        pivot_row = None
        for i in range(rank_count, n_rows):
            if A[i] & bit:
                pivot_row = i
                break

        if pivot_row is not None:
            if pivot_row != rank_count:
                A[rank_count], A[pivot_row] = A[pivot_row], A[rank_count]

            pivot = A[rank_count]
            # Forward elimination only: enough for rank, half the XORs of RREF.
            for i in range(rank_count + 1, n_rows):
                if A[i] & bit:
                    A[i] ^= pivot

            rank_count += 1

        bit <<= 1

    return rank_count


def det(A: SparseGF2Matrix | DenseGF2Matrix) -> int:
    """
    Compute determinant of square GF(2) matrix.

    Returns:
        0 or 1 (determinant in GF(2))
    """
    if A.rows != A.cols:
        raise ValueError("Matrix must be square")

    return _det_bitwise(_rows_of(A), A.cols)


def _det_bitwise(rows: list[int], n: int) -> int:
    """Internal determinant computation over GF(2).

    Returns 1 iff the matrix is full rank (all pivots found), else 0.
    """
    # det over GF(2) is 1 exactly when the matrix is full rank, and rank only
    # needs forward elimination -- eliminating above the pivot as well doubled
    # the XOR count for an answer that never used the upper triangle.
    return 1 if _rank_bitwise(rows, n) == n else 0


def trace(A: SparseGF2Matrix | DenseGF2Matrix) -> int:
    """
    Compute trace (sum of diagonal elements) in GF(2).
    """
    if A.rows != A.cols:
        raise ValueError("Matrix must be square")

    tr = 0
    for i in range(A.rows):
        row_packed = A.get_row_bitwise(i)
        if (row_packed >> i) & 1:
            tr ^= 1  # XOR in GF(2)

    return tr


def is_invertible(A: SparseGF2Matrix | DenseGF2Matrix) -> bool:
    """
    Check if matrix is invertible over GF(2).
    """
    if A.rows != A.cols:
        return False

    return rank(A) == A.rows


def gaussian_elimination_inplace(
    rows: list[int], n_cols: int, full_reduction: bool = True
) -> tuple[list[int], list[int]]:
    """
    Perform Gaussian elimination in-place and return pivot columns.

    Args:
        rows: List of packed row integers
        n_cols: Number of columns
        full_reduction: If True, compute RREF (eliminate above and below).
                       If False, only forward elimination (eliminate below only).

    Returns:
        (reduced_rows, pivot_columns)
    """
    pivot_cols: list[int] = []
    rank_count = 0
    n_rows = len(rows)
    bit = 1

    for col in range(n_cols):
        if rank_count == n_rows:
            break

        pivot_row = None
        for i in range(rank_count, n_rows):
            if rows[i] & bit:
                pivot_row = i
                break

        if pivot_row is not None:
            if pivot_row != rank_count:
                rows[rank_count], rows[pivot_row] = rows[pivot_row], rows[rank_count]

            pivot_cols.append(col)
            pivot = rows[rank_count]

            if full_reduction:
                # Full RREF: needed for nullspace and solve back-substitution.
                for i in range(n_rows):
                    if i != rank_count and rows[i] & bit:
                        rows[i] ^= pivot
            else:
                # Forward elimination only: sufficient for rank.
                for i in range(rank_count + 1, n_rows):
                    if rows[i] & bit:
                        rows[i] ^= pivot

            rank_count += 1

        bit <<= 1

    return rows[:rank_count], pivot_cols


def reduced_row_echelon_form(A: SparseGF2Matrix | DenseGF2Matrix) -> tuple[SparseGF2Matrix, list[int]]:
    """
    Compute reduced row echelon form (RREF) of matrix.

    Returns:
        (rref_matrix, pivot_columns)
    """
    # Copy: gaussian_elimination_inplace mutates the list it is handed.
    rows = _rows_of(A)[:]

    # Perform elimination
    rref_rows, pivot_cols = gaussian_elimination_inplace(rows, A.cols)

    # Create result matrix
    result = SparseGF2Matrix(len(rref_rows), A.cols)
    result.set_from_packed_rows(rref_rows)

    return result, pivot_cols


def lu_decomposition(A: SparseGF2Matrix | DenseGF2Matrix) -> tuple[SparseGF2Matrix, SparseGF2Matrix]:
    """
    LU decomposition over GF(2) using Gaussian elimination.

    Returns PLU decomposition where P is implicit (no pivoting for simplicity).
    For GF(2), we perform elimination without pivoting when possible.

    Returns:
        (L, U) where L * U approximates A (may need permutation for exact equality)
    """
    if A.rows != A.cols:
        raise ValueError("Matrix must be square for LU decomposition")

    n = A.rows

    # Initialize L as identity, U as copy of A
    L_rows = [(1 << i) for i in range(n)]  # Identity matrix
    U_rows = _rows_of(A)[:]

    # Gaussian elimination without pivoting (for GF(2) simplicity)
    for k in range(n):
        # Check if pivot exists
        if not ((U_rows[k] >> k) & 1):
            # Try to find a row below with non-zero element in column k
            pivot_found = False
            for i in range(k + 1, n):
                if (U_rows[i] >> k) & 1:
                    # Swap rows in U only
                    U_rows[k], U_rows[i] = U_rows[i], U_rows[k]
                    pivot_found = True
                    break

            if not pivot_found:
                # Column k is all zeros below diagonal, continue
                continue

        # Eliminate below diagonal
        for i in range(k + 1, n):
            if (U_rows[i] >> k) & 1:
                # Record the elimination in L
                L_rows[i] |= 1 << k  # Set L[i,k] = 1
                # Eliminate in U
                U_rows[i] ^= U_rows[k]

    # Create result matrices
    L = SparseGF2Matrix(n, n)
    L.set_from_packed_rows(L_rows)

    U = SparseGF2Matrix(n, n)
    U.set_from_packed_rows(U_rows)

    return L, U


def matrix_power(A: SparseGF2Matrix | DenseGF2Matrix, k: int) -> SparseGF2Matrix | DenseGF2Matrix:
    """
    Compute A^k using fast exponentiation.
    """
    if A.rows != A.cols:
        raise ValueError("Matrix must be square")

    if k == 0:
        # Return identity matrix
        n = A.rows
        identity_rows = [(1 << i) for i in range(n)]
        identity_result = SparseGF2Matrix(n, n)
        identity_result.set_from_packed_rows(identity_rows)
        return identity_result

    if k == 1:
        return A

    # Fast exponentiation
    result: SparseGF2Matrix | DenseGF2Matrix = matrix_power(A, k // 2)
    result = multiply(result, result)

    if k % 2 == 1:
        result = multiply(result, A)

    return result


def _hessenberg_gf2(rows: list[int], n: int) -> list[int]:
    """Upper Hessenberg form of ``rows`` by similarity transforms over GF(2).

    Similarity preserves the characteristic polynomial, and the char poly of a
    Hessenberg matrix follows from an O(n^2) recurrence, so this is the usual
    O(n^3) route. Faddeev-LeVerrier is unusable here: it divides by k, and
    every even k is zero in characteristic 2.
    """
    H = rows[:]

    for m in range(1, n - 1):
        pivot = None
        probe = 1 << (m - 1)
        for i in range(m, n):
            if H[i] & probe:
                pivot = i
                break
        if pivot is None:
            continue

        if pivot != m:
            H[m], H[pivot] = H[pivot], H[m]
            # Matching column swap keeps the transform a similarity.
            bit_m, bit_p = 1 << m, 1 << pivot
            for r in range(n):
                has_m = H[r] & bit_m
                has_p = H[r] & bit_p
                if bool(has_m) != bool(has_p):
                    H[r] ^= bit_m | bit_p

        bit_m = 1 << m
        for i in range(m + 1, n):
            if H[i] & probe:
                # Re-read H[m]: the column op below writes into column m of
                # every row, H[m] included, so a hoisted copy goes stale.
                H[i] ^= H[m]  # row op ...
                bit_i = 1 << i
                for r in range(n):  # ... and its inverse column op
                    if H[r] & bit_i:
                        H[r] ^= bit_m

    return H


def _poly_mul_shift_add(poly: list[int], constant: int) -> list[int]:
    """(x + constant) * poly over GF(2)."""
    out = [0] * (len(poly) + 1)
    for i, c in enumerate(poly):
        if c:
            out[i + 1] ^= 1
            if constant:
                out[i] ^= 1
    return out


def characteristic_polynomial(A: SparseGF2Matrix | DenseGF2Matrix) -> list[int]:
    """
    Compute characteristic polynomial coefficients over GF(2).

    Returns coefficients of det(xI - A) as list [c0, c1, ..., cn]
    where the polynomial is c0 + c1*x + ... + cn*x^n (so cn == 1).

    Note:
        Previous releases returned a stub: leading coefficient 1, c[n-1] =
        trace, c[0] = det and *every other coefficient zero*. That is wrong for
        any n >= 3 -- e.g. the companion matrix of x^3 + x + 1 came back as
        x^3 + 1 -- and it was wrong silently.
    """
    if A.rows != A.cols:
        raise ValueError("Matrix must be square")

    n = A.rows
    if n == 0:
        return [1]

    H = _hessenberg_gf2(_rows_of(A)[:], n)

    # Cohen, "A Course in Computational Algebraic Number Theory", 2.2.9.
    # Signs vanish in characteristic 2, so every minus is a plus.
    polys: list[list[int]] = [[1]]
    for m in range(1, n + 1):
        acc = _poly_mul_shift_add(polys[m - 1], (H[m - 1] >> (m - 1)) & 1)

        subdiag = 1
        for i in range(1, m):
            subdiag &= (H[m - i] >> (m - i - 1)) & 1
            if not subdiag:
                break
            if (H[m - i - 1] >> (m - 1)) & 1:
                term = polys[m - i - 1]
                for idx, c in enumerate(term):
                    if c:
                        acc[idx] ^= 1

        polys.append(acc)

    return polys[n]


def minimal_polynomial(A: SparseGF2Matrix | DenseGF2Matrix) -> list[int]:
    """
    Compute the minimal polynomial of A over GF(2).

    The minimal polynomial is the monic polynomial of lowest degree that
    annihilates the matrix.

    Finds the first linear dependence among I, A, A^2, ... by flattening each
    power into one wide bit vector, carrying the coefficient that produced it
    alongside, and eliminating. The first power that reduces to zero hands back
    its own annihilating combination.

    Note:
        The previous implementation enumerated all 2^d coefficient vectors for
        each degree d, i.e. O(2^n) polynomial evaluations each costing a chain
        of matrix multiplies. It was unusable beyond about n = 10; this is
        O(n^4 / 64) word operations.
    """
    if A.rows != A.cols:
        raise ValueError("Matrix must be square")

    n = A.rows
    if n == 0:
        return [1]

    def flatten(rows: list[int]) -> int:
        packed = 0
        for i, row in enumerate(rows):
            packed |= row << (i * n)
        return packed

    identity_rows = [1 << i for i in range(n)]
    power_rows = identity_rows
    a_rows = _rows_of(A)

    # pivot bit -> (reduced vector, coefficient mask) of an independent power
    pivots: dict[int, tuple[int, int]] = {}

    for degree in range(n + 1):
        vector = flatten(power_rows)
        coeffs = 1 << degree

        for bit, (basis_vec, basis_coeffs) in pivots.items():
            if vector & bit:
                vector ^= basis_vec
                coeffs ^= basis_coeffs

        if vector == 0:
            # coeffs now names a combination of A^0..A^degree equal to zero.
            return [(coeffs >> i) & 1 for i in range(degree + 1)]

        pivots[vector & -vector] = (vector, coeffs)

        # Advance to the next power: one row-XOR product, no wrapper objects.
        power_rows = [_xor_selected(row, a_rows) for row in power_rows]

    # Cayley-Hamilton guarantees a dependence by degree n, so this is unreachable.
    return characteristic_polynomial(A)


def _xor_selected(selector: int, source_rows: list[int]) -> int:
    """XOR together the rows of ``source_rows`` picked out by ``selector``."""
    acc = 0
    remaining = selector
    while remaining:
        low = remaining & -remaining
        acc ^= source_rows[low.bit_length() - 1]
        remaining ^= low
    return acc


def matrix_norm(A: SparseGF2Matrix | DenseGF2Matrix, norm_type: str = "hamming") -> float:
    """
    Compute matrix norm over GF(2).

    Args:
        norm_type: "hamming" (number of 1s), "rank", or "spectral"
    """
    if norm_type == "hamming":
        # Count total number of 1s
        return float(sum(row.bit_count() for row in _rows_of(A)))

    elif norm_type == "rank":
        return float(rank(A))

    elif norm_type == "spectral":
        # For GF(2), spectral norm is more complex
        # Simplified: return sqrt of largest eigenvalue of A^T * A
        AT = transpose(A)
        ATA = multiply(AT, A)
        # For now, return rank as approximation
        return float(rank(ATA))

    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def condition_number(A: SparseGF2Matrix | DenseGF2Matrix) -> float:
    """
    Compute condition number over GF(2).

    For binary matrices, this is typically defined as
    the ratio of largest to smallest non-zero singular values.
    """
    if A.rows != A.cols:
        raise ValueError("Matrix must be square")

    if not is_invertible(A):
        return float("inf")

    # For GF(2), condition number is often just 1 for invertible matrices
    # or inf for singular matrices
    return 1.0
