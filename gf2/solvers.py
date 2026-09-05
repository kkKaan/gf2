"""
Linear System Solvers for GF(2)
===============================

Optimized solvers for linear systems over GF(2) including the enhanced
algorithms from Simon's algorithm postprocessing. All solvers use bitwise
operations for maximum performance.

Solvers:
- solve(A, b): Solve Ax = b
- nullspace(A): Find null space basis
- nullspace_fast(matrix): Fast nullspace for list of lists
- inverse(A): Matrix inversion
- least_squares(A, b): Overdetermined systems
- kernel(A): Kernel computation
- image(A): Image/column space
"""

import time
from typing import NamedTuple

import numpy as np

from .core import _rows_of, gaussian_elimination_inplace
from .sparse import DenseGF2Matrix, SparseGF2Matrix


def solve(A: SparseGF2Matrix | DenseGF2Matrix, b: list[int] | np.ndarray) -> list[int] | None:
    """
    Solve linear system Ax = b over GF(2).

    Args:
        A: Coefficient matrix
        b: Right-hand side vector

    Returns:
        Solution vector x, or None if no solution exists
    """
    if A.rows != len(b):
        raise ValueError("Matrix and vector dimensions must match")

    # Augment: b[i] rides along as bit A.cols of row i.
    rhs_bit = 1 << A.cols
    rows = [row | (rhs_bit if int(b[i]) & 1 else 0) for i, row in enumerate(_rows_of(A))]

    # Eliminate over the coefficient columns ONLY. Letting the elimination
    # pivot on the augmented column would put A.cols into pivot_cols and index
    # past the end of the solution vector during back-substitution.
    all_rows, pivot_cols = gaussian_elimination_inplace(rows, A.cols)
    rref_rows = all_rows

    # Any surviving row that is zero across the coefficients but 1 in the
    # augment states 0 == 1: no solution.
    coeff_mask = rhs_bit - 1
    for row in rows:
        if not (row & coeff_mask) and (row & rhs_bit):
            return None

    # gaussian_elimination_inplace ran a full reduction, so pivot row i is
    # e[pivot_col] plus free columns only: x[pivot_col] is simply its rhs bit.
    # Free variables stay 0, which is a valid particular solution.
    solution = [0] * A.cols
    for i, pivot_col in enumerate(pivot_cols):
        solution[pivot_col] = 1 if rref_rows[i] & rhs_bit else 0

    return solution


def nullspace(A: SparseGF2Matrix | DenseGF2Matrix) -> list[list[int]]:
    """
    Find basis for null space of A using optimized GF(2) operations.
    This is the enhanced algorithm from Simon's algorithm postprocessing.

    Returns:
        List of basis vectors for null(A)
    """
    # Copy: gaussian_elimination_inplace mutates the list it is given.
    rows = _rows_of(A)[:]

    # Gaussian elimination to find pivot columns
    rref_rows, pivot_cols = gaussian_elimination_inplace(rows, A.cols)

    # Find free columns
    all_cols = set(range(A.cols))
    free_cols = sorted(all_cols - set(pivot_cols))

    if not free_cols:
        # Null space is trivial
        return []

    # Generate basis vectors
    basis = []

    # In RREF each pivot row reads x[p_i] = sum over free columns f of R[i,f]*x[f].
    # With exactly one free variable set to 1, that sum collapses to the single
    # bit R[i, free_col] -- so the inner scan over every column to the right of
    # the pivot was doing O(cols) work per pivot to recover one stored bit,
    # making basis extraction O(free * pivots * cols) instead of O(free * pivots).
    for free_col in free_cols:
        basis_vector = [0] * A.cols
        basis_vector[free_col] = 1

        for i, pivot_col in enumerate(pivot_cols):
            basis_vector[pivot_col] = (rref_rows[i] >> free_col) & 1

        basis.append(basis_vector)

    return basis


class NullspaceVector(NamedTuple):
    """One null-space vector plus how long finding it took.

    Unpacks as ``(bits, seconds)``. The second element being a duration is
    surprising enough on its own that naming it is worth the class.

    Attributes:
        bits: The vector as a binary string, least significant column first, so
            ``bits[j]`` is the value of variable j.
        seconds: Wall-clock duration of this single call, from
            ``time.perf_counter``. A convenience for interactive use only --
            one untimed-warmup sample is not a benchmark. Use
            ``benchmarks/harness.py`` for measurement.
    """

    bits: str
    seconds: float


def nullspace_bitwise(A: SparseGF2Matrix | DenseGF2Matrix) -> NullspaceVector:
    """
    Optimized nullspace computation returning single solution as bit string.
    This is the original algorithm from Simon's algorithm postprocessing.

    Returns:
        (solution_string, computation_time)
    """
    start_time = time.perf_counter()

    rows = _rows_of(A)[:]  # copy: elimination mutates the list

    n = A.cols
    A_echelon, pivot_cols = _gaussian_elimination_GF2_bitwise(rows, n)
    sol_int = _nullspace_solution_bitwise(A_echelon, pivot_cols, n)

    # Unpack solution
    sol_bits = _unpack_vector(sol_int, n)
    sol_str = "".join(str(b) for b in sol_bits)

    elapsed_time = time.perf_counter() - start_time
    return NullspaceVector(sol_str, elapsed_time)


def nullspace_fast(matrix: list[list[int]], include_packing_time: bool = True) -> NullspaceVector:
    """
    FASTEST nullspace computation - bypasses all matrix wrapper overhead.
    Works directly with list of lists input.

    This is the zero-overhead version for maximum performance.
    Use this when you have raw matrix data and need maximum speed.

    Args:
        matrix: List of lists representing binary matrix (each row is a list of 0/1)
        include_packing_time: If True (the default) the returned time covers
            packing the input as well as the solve. Set False to time only the
            elimination, e.g. when the caller already holds packed rows.

    Returns:
        (solution_string, computation_time)

    Note:
        The returned time is a convenience for interactive use. It is a single
        untimed-warmup sample of ``time.perf_counter``; do not build a
        benchmark on it -- use ``benchmarks/harness.py``, which warms up,
        repeats, and reports the minimum.
    """
    n = len(matrix[0])

    # The flag used to be accepted and then ignored: timing always started
    # after packing, so every reported figure silently excluded it.
    start_time = time.perf_counter()
    rows = [_pack_vector(row) for row in matrix]
    if not include_packing_time:
        start_time = time.perf_counter()

    # Gaussian elimination
    A_echelon, pivot_cols = _gaussian_elimination_GF2_bitwise(rows, n)

    # Solve for nullspace vector
    sol_int = _nullspace_solution_bitwise(A_echelon, pivot_cols, n)

    # Unpack solution
    sol_bits = _unpack_vector(sol_int, n)
    sol_str = "".join(str(b) for b in sol_bits)

    elapsed_time = time.perf_counter() - start_time
    return NullspaceVector(sol_str, elapsed_time)


def _pack_vector(vec):
    """Pack list of bits into integer (bit i = vec[i])."""
    # One int() over a reversed binary string beats a shift-and-OR loop, which
    # reallocates the accumulating integer on every element.
    if not vec:
        return 0
    return int("".join("1" if b & 1 else "0" for b in reversed(vec)), 2)


def _unpack_vector(x, n):
    """Unpack integer into list of n bits."""
    if n <= 0:
        return []
    bits = format(x & ((1 << n) - 1), f"0{n}b")
    return [int(c) for c in reversed(bits)]


def _gaussian_elimination_GF2_bitwise(rows, n):
    """Forward elimination to row echelon form, tracking pivot columns."""
    A = rows[:]
    n_rows = len(A)
    pivot_cols = []
    r = 0
    bit = 1

    for col in range(n):
        if r == n_rows:
            break

        # ``A[i] & bit`` rather than ``(A[i] >> col) & 1``: the shift builds a
        # full-width copy of the row for every probe, the mask does not.
        pivot_row = None
        for i in range(r, n_rows):
            if A[i] & bit:
                pivot_row = i
                break

        if pivot_row is not None:
            A[r], A[pivot_row] = A[pivot_row], A[r]
            pivot_cols.append(col)

            pivot = A[r]
            for i in range(r + 1, n_rows):
                if A[i] & bit:
                    A[i] ^= pivot
            r += 1

        bit <<= 1

    return A, pivot_cols


def _nullspace_solution_bitwise(rows, pivot_cols, n):
    """One non-trivial nullspace vector, as a packed integer.

    Back-substitution is carried out on the packed vector directly: row i is
    zero in every column left of its pivot and x has no bit set at the pivot
    yet, so the whole "sum of already-known variables" term is the parity of
    ``row & x`` -- a single masked popcount. The previous loop walked every
    column right of each pivot, which made this O(pivots * cols) Python steps
    with a full-width shift inside each one.
    """
    free_cols = sorted(set(range(n)) - set(pivot_cols))

    if not free_cols:
        raise ValueError("No free variable found; the system appears to be full rank.")

    sol_int = 1 << free_cols[0]

    for i in reversed(range(len(pivot_cols))):
        if (rows[i] & sol_int).bit_count() & 1:
            sol_int |= 1 << pivot_cols[i]

    return sol_int


def inverse(A: SparseGF2Matrix | DenseGF2Matrix) -> SparseGF2Matrix | None:
    """
    Compute matrix inverse over GF(2) using Gauss-Jordan elimination.

    Returns:
        Inverse matrix, or None if matrix is not invertible
    """
    if A.rows != A.cols:
        raise ValueError("Matrix must be square")

    n = A.rows

    # Create augmented matrix [A | I]
    augmented_rows = [row | (1 << (n + i)) for i, row in enumerate(_rows_of(A))]

    # Gauss-Jordan on the augmented matrix, probing with a mask rather than a
    # shift so each test does not allocate a shifted copy of the whole row.
    bit = 1
    for col in range(n):
        pivot_row = None
        for i in range(col, n):
            if augmented_rows[i] & bit:
                pivot_row = i
                break

        if pivot_row is None:
            return None  # Singular matrix

        if pivot_row != col:
            augmented_rows[col], augmented_rows[pivot_row] = augmented_rows[pivot_row], augmented_rows[col]

        pivot = augmented_rows[col]
        for i in range(n):
            if i != col and augmented_rows[i] & bit:
                augmented_rows[i] ^= pivot

        bit <<= 1

    # Extract inverse from right side of augmented matrix
    mask = (1 << n) - 1
    inverse_rows = [(row >> n) & mask for row in augmented_rows]

    # Create result matrix
    result = SparseGF2Matrix(n, n)
    result.set_from_packed_rows(inverse_rows)

    return result


def least_squares(A: SparseGF2Matrix | DenseGF2Matrix, b: list[int] | np.ndarray) -> list[int] | None:
    """
    Solve the GF(2) normal equations A^T A x = A^T b.

    Warning:
        This is NOT a least-squares solver. Least squares needs an inner
        product with a positive-definite norm; over GF(2) the bilinear form
        x -> x^T x is degenerate (every even-weight vector is self-orthogonal),
        so a solution of the normal equations does not minimise anything, and
        the equations may be inconsistent even when a nearest vector exists.
        Minimising Hamming distance to the column space is the nearest-codeword
        problem, which is NP-hard.

        Kept because the normal equations are still useful for consistent
        overdetermined systems: when Ax = b has a solution, so does this.
        Returns None when the normal equations are inconsistent.
    """
    from .core import multiply, transpose

    # Compute A^T
    AT = transpose(A)

    # Compute A^T * A
    ATA = multiply(AT, A)

    # Compute A^T * b as a parity of overlaps rather than a shift per column.
    b_packed = _pack_vector([int(v) & 1 for v in b])
    ATb = [(row & b_packed).bit_count() & 1 for row in _rows_of(AT)]

    # Solve (A^T A) x = A^T b
    return solve(ATA, ATb)


def kernel(A: SparseGF2Matrix | DenseGF2Matrix) -> list[list[int]]:
    """
    Compute kernel (null space) of matrix A.
    Alias for nullspace function.
    """
    return nullspace(A)


def image(A: SparseGF2Matrix | DenseGF2Matrix) -> list[list[int]]:
    """
    Compute image (column space) of matrix A.

    Returns:
        Basis for column space of A
    """
    # Transpose to work with rows instead of columns
    from .core import transpose

    AT = transpose(A)

    # Find row echelon form
    rows = _rows_of(AT)[:]
    rref_rows, pivot_cols = gaussian_elimination_inplace(rows, AT.cols)

    # Convert back to column vectors
    return [_unpack_vector(row, AT.cols) for row in rref_rows if row]


class RankNullity(NamedTuple):
    """Result of :func:`rank_nullity_theorem`, where ``rank + nullity == columns``.

    Unpacks as ``(rank, nullity, columns)``. Three bare integers in a row is
    exactly the shape a caller mis-orders, so they are named.
    """

    rank: int
    nullity: int
    columns: int


def rank_nullity_theorem(A: SparseGF2Matrix | DenseGF2Matrix) -> RankNullity:
    """
    Verify rank-nullity theorem: rank(A) + nullity(A) = cols(A).

    Returns:
        A :class:`RankNullity` of ``(rank, nullity, columns)``.
    """
    from .core import rank

    matrix_rank = rank(A)
    null_basis = nullspace(A)
    nullity = len(null_basis)

    return RankNullity(matrix_rank, nullity, A.cols)


def solve_multiple_rhs(
    A: SparseGF2Matrix | DenseGF2Matrix, B: SparseGF2Matrix | DenseGF2Matrix
) -> SparseGF2Matrix | None:
    """
    Solve AX = B for matrix X (multiple right-hand sides).

    Args:
        A: Coefficient matrix
        B: Multiple right-hand side vectors (as columns)

    Returns:
        Solution matrix X, or None if no solution exists
    """
    if A.rows != B.rows:
        raise ValueError("A and B must have same number of rows")

    # Augment A with ALL of B's columns at once and eliminate a single time.
    # Calling solve() per column repeated the whole O(n^3) elimination B.cols
    # times over an unchanged coefficient matrix.
    a_rows = _rows_of(A)
    b_rows = _rows_of(B)
    rows = [row | (b_rows[i] << A.cols) for i, row in enumerate(a_rows)]

    coeff_mask = (1 << A.cols) - 1
    rref_rows, pivot_cols = gaussian_elimination_inplace(rows, A.cols)

    # A row that is zero across the coefficients but non-zero in some augmented
    # column means that column's system is inconsistent.
    for row in rows:
        if not (row & coeff_mask) and (row >> A.cols):
            return None

    # Free variables are 0, so each pivot variable equals its augmented row.
    result_rows = [0] * A.cols
    for i, pivot_col in enumerate(pivot_cols):
        result_rows[pivot_col] = rref_rows[i] >> A.cols

    result = SparseGF2Matrix(A.cols, B.cols)
    result.set_from_packed_rows(result_rows)

    return result


def condition_analysis(A: SparseGF2Matrix | DenseGF2Matrix) -> dict:
    """
    Analyze condition properties of matrix over GF(2).

    Returns:
        Dictionary with analysis results
    """
    from .core import det, is_invertible, rank

    analysis = {
        "rows": A.rows,
        "cols": A.cols,
        "rank": rank(A),
        "is_square": A.rows == A.cols,
        "is_invertible": False,
        "determinant": None,
        "nullity": 0,
        "condition_number": float("inf"),
    }

    if analysis["is_square"]:
        analysis["is_invertible"] = is_invertible(A)
        analysis["determinant"] = det(A)

        if analysis["is_invertible"]:
            analysis["condition_number"] = 1.0

    # Compute nullity
    null_basis = nullspace(A)
    analysis["nullity"] = len(null_basis)

    # Verify rank-nullity theorem
    rank_val = analysis["rank"]
    nullity_val = analysis["nullity"]
    if rank_val is not None and nullity_val is not None:
        analysis["rank_nullity_check"] = rank_val + nullity_val == A.cols
    else:
        analysis["rank_nullity_check"] = False

    return analysis


def iterative_refinement(
    A: SparseGF2Matrix | DenseGF2Matrix,
    b: list[int] | np.ndarray,
    x0: list[int] | None = None,
    max_iterations: int = 10,
) -> tuple[list[int] | None, int]:
    """
    Iterative refinement for solving Ax = b over GF(2).

    Args:
        A: Coefficient matrix
        b: Right-hand side
        x0: Initial guess (if None, use zero vector)
        max_iterations: Maximum number of iterations

    Returns:
        (solution, iterations_used)
    """
    x = [0] * A.cols if x0 is None else list(x0)
    a_rows = _rows_of(A)
    x_packed_width = A.cols

    for iteration in range(max_iterations):
        # list(b) rather than b[:] -- slicing a NumPy array yields a *view*, so
        # the old code XOR-ed the residual straight into the caller's array.
        residual = [int(v) & 1 for v in b]

        x_packed = _pack_vector(x[:x_packed_width])
        for i, row_packed in enumerate(a_rows):
            # A row dotted with x is the parity of their overlap: one masked
            # popcount instead of a shift per column.
            residual[i] ^= (row_packed & x_packed).bit_count() & 1

        # Check convergence
        if all(r == 0 for r in residual):
            return x, iteration

        # Solve for correction: A * delta = residual
        delta = solve(A, residual)
        if delta is None:
            break

        # Update solution: x = x + delta (XOR in GF(2))
        for j in range(A.cols):
            x[j] ^= delta[j]

    return x, max_iterations


def benchmark_solver(
    A: SparseGF2Matrix | DenseGF2Matrix, b: list[int] | np.ndarray, num_trials: int = 100
) -> dict:
    """
    Benchmark solver performance.

    Returns:
        Performance statistics
    """
    solve(A, b)  # warm up: the first call pays for row-cache construction

    times = []
    for _ in range(num_trials):
        start_time = time.perf_counter()
        solve(A, b)
        times.append(time.perf_counter() - start_time)

    # min_time is the headline figure: for deterministic CPU work the minimum
    # is the sample least contaminated by scheduler noise. mean/max are kept
    # so a noisy run is visible rather than averaged away.
    return {
        "min_time": float(np.min(times)),
        "median_time": float(np.median(times)),
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "max_time": float(np.max(times)),
        "total_time": float(np.sum(times)),
        "trials": num_trials,
    }
