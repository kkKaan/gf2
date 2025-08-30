"""
Linear System Solvers for GF(2)
===============================

Optimized solvers for linear systems over GF(2) including the enhanced
algorithms from Simon's algorithm postprocessing. All solvers use bitwise
operations for maximum performance.

Solvers:
- solve(A, b): Solve Ax = b
- nullspace(A): Find null space basis
- inverse(A): Matrix inversion
- least_squares(A, b): Overdetermined systems
- kernel(A): Kernel computation
- image(A): Image/column space
"""

import time

import numpy as np

from .core import gaussian_elimination_inplace
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

    # Convert to packed representation
    rows = []
    for i in range(A.rows):
        row_packed = A.get_row_bitwise(i)
        # Append b[i] as the least significant bit beyond the matrix columns
        augmented_row = row_packed | (b[i] << A.cols)
        rows.append(augmented_row)

    # Perform Gaussian elimination on augmented matrix
    rref_rows, pivot_cols = gaussian_elimination_inplace(rows, A.cols + 1)

    # Check for inconsistency
    for row in rref_rows:
        # If row is [0 0 ... 0 | 1], system is inconsistent
        matrix_part = row & ((1 << A.cols) - 1)
        augment_part = (row >> A.cols) & 1

        if matrix_part == 0 and augment_part == 1:
            return None  # No solution

    # Back substitution
    solution = [0] * A.cols

    # Set free variables to 0, solve for pivot variables
    for i in reversed(range(len(pivot_cols))):
        pivot_col = pivot_cols[i]
        row = rref_rows[i]

        # Extract equation: x[pivot_col] + sum of free variables = rhs
        rhs = (row >> A.cols) & 1

        # Sum contributions from variables to the right of pivot
        sum_free = 0
        for j in range(pivot_col + 1, A.cols):
            if (row >> j) & 1:
                sum_free ^= solution[j]

        solution[pivot_col] = sum_free ^ rhs

    return solution


def nullspace(A: SparseGF2Matrix | DenseGF2Matrix) -> list[list[int]]:
    """
    Find basis for null space of A using optimized GF(2) operations.
    This is the enhanced algorithm from Simon's algorithm postprocessing.

    Returns:
        List of basis vectors for null(A)
    """
    # Convert to packed representation
    rows = [A.get_row_bitwise(i) for i in range(A.rows)]

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

    for free_col in free_cols:
        # Create basis vector with this free variable set to 1
        basis_vector = [0] * A.cols
        basis_vector[free_col] = 1

        # Back substitution to find values of pivot variables
        for i in reversed(range(len(pivot_cols))):
            pivot_col = pivot_cols[i]
            row = rref_rows[i]

            # Sum contributions from free variables
            sum_free = 0
            for j in range(pivot_col + 1, A.cols):
                if (row >> j) & 1:
                    sum_free ^= basis_vector[j]

            basis_vector[pivot_col] = sum_free

        basis.append(basis_vector)

    return basis


def nullspace_bitwise(A: SparseGF2Matrix | DenseGF2Matrix) -> tuple[str, float]:
    """
    Optimized nullspace computation returning single solution as bit string.
    This is the original algorithm from Simon's algorithm postprocessing.

    Returns:
        (solution_string, computation_time)
    """
    start_time = time.time()

    # Convert to list of lists for compatibility with original algorithm
    matrix_list = []
    for i in range(A.rows):
        row_packed = A.get_row_bitwise(i)
        row = [(row_packed >> j) & 1 for j in range(A.cols)]
        matrix_list.append(row)

    # Use original optimized algorithm
    n = A.cols
    rows = [_pack_vector(row) for row in matrix_list]

    A_echelon, pivot_cols = _gaussian_elimination_GF2_bitwise(rows, n)
    sol_int = _nullspace_solution_bitwise(A_echelon, pivot_cols, n)

    # Unpack solution
    sol_bits = _unpack_vector(sol_int, n)
    sol_str = "".join(str(b) for b in sol_bits)

    elapsed_time = time.time() - start_time
    return sol_str, elapsed_time


def _pack_vector(vec):
    """Pack list of bits into integer."""
    out = 0
    for i, bit in enumerate(vec):
        out |= (bit & 1) << i
    return out


def _unpack_vector(x, n):
    """Unpack integer into list of n bits."""
    return [(x >> i) & 1 for i in range(n)]


def _gaussian_elimination_GF2_bitwise(rows, n):
    """Original optimized Gaussian elimination."""
    A = rows[:]
    pivot_cols = []
    r = 0

    for col in range(n):
        pivot_row = None
        for i in range(r, len(A)):
            if (A[i] >> col) & 1:
                pivot_row = i
                break
        if pivot_row is None:
            continue

        A[r], A[pivot_row] = A[pivot_row], A[r]
        pivot_cols.append(col)

        for i in range(r + 1, len(A)):
            if (A[i] >> col) & 1:
                A[i] ^= A[r]
        r += 1
        if r == len(A):
            break

    return A, pivot_cols


def _nullspace_solution_bitwise(rows, pivot_cols, n):
    """Original optimized nullspace solution."""
    all_cols = set(range(n))
    free_cols = sorted(all_cols - set(pivot_cols))

    if not free_cols:
        raise ValueError("No free variable found; the system appears to be full rank.")

    x = [0] * n
    x[free_cols[0]] = 1

    num_pivots = len(pivot_cols)
    for i in reversed(range(num_pivots)):
        p = pivot_cols[i]
        sum_free = 0
        for j in range(p + 1, n):
            if (rows[i] >> j) & 1:
                sum_free ^= x[j]
        x[p] = sum_free

    sol_int = _pack_vector(x)
    if sol_int == 0:
        sol_int = 1 << free_cols[0]

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
    augmented_rows = []
    for i in range(n):
        row_a = A.get_row_bitwise(i)
        row_identity = 1 << i  # Identity matrix row
        # Combine: A on left, I on right
        augmented = row_a | (row_identity << n)
        augmented_rows.append(augmented)

    # Gaussian elimination on augmented matrix
    for col in range(n):
        # Find pivot
        pivot_row = None
        for i in range(col, n):
            if (augmented_rows[i] >> col) & 1:
                pivot_row = i
                break

        if pivot_row is None:
            return None  # Singular matrix

        # Swap rows
        if pivot_row != col:
            augmented_rows[col], augmented_rows[pivot_row] = augmented_rows[pivot_row], augmented_rows[col]

        # Eliminate
        for i in range(n):
            if i != col and (augmented_rows[i] >> col) & 1:
                augmented_rows[i] ^= augmented_rows[col]

    # Extract inverse from right side of augmented matrix
    inverse_rows = []
    for i in range(n):
        # Extract right half (columns n to 2n-1)
        right_half = (augmented_rows[i] >> n) & ((1 << n) - 1)
        inverse_rows.append(right_half)

    # Create result matrix
    result = SparseGF2Matrix(n, n)
    result.set_from_packed_rows(inverse_rows)

    return result


def least_squares(A: SparseGF2Matrix | DenseGF2Matrix, b: list[int] | np.ndarray) -> list[int] | None:
    """
    Solve overdetermined system in least squares sense over GF(2).

    For GF(2), this reduces to solving A^T A x = A^T b.
    """
    from .core import multiply, transpose

    # Compute A^T
    AT = transpose(A)

    # Compute A^T * A
    ATA = multiply(AT, A)

    # Compute A^T * b
    ATb = []
    for i in range(AT.rows):
        row_packed = AT.get_row_bitwise(i)
        dot_product = 0
        for j in range(len(b)):
            if (row_packed >> j) & 1:
                dot_product ^= b[j]
        ATb.append(dot_product)

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
    rows = [AT.get_row_bitwise(i) for i in range(AT.rows)]
    rref_rows, pivot_cols = gaussian_elimination_inplace(rows, AT.cols)

    # Convert back to column vectors
    basis = []
    for row_packed in rref_rows:
        if row_packed != 0:  # Non-zero row
            col_vector = [(row_packed >> i) & 1 for i in range(AT.cols)]
            basis.append(col_vector)

    return basis


def rank_nullity_theorem(A: SparseGF2Matrix | DenseGF2Matrix) -> tuple[int, int, int]:
    """
    Verify rank-nullity theorem: rank(A) + nullity(A) = cols(A).

    Returns:
        (rank, nullity, columns)
    """
    from .core import rank

    matrix_rank = rank(A)
    null_basis = nullspace(A)
    nullity = len(null_basis)

    return matrix_rank, nullity, A.cols


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

    # Solve for each column of B
    solution_columns = []

    for j in range(B.cols):
        # Extract column j from B
        b_col = []
        for i in range(B.rows):
            row_packed = B.get_row_bitwise(i)
            b_col.append((row_packed >> j) & 1)

        # Solve Ax = b_col
        x = solve(A, b_col)
        if x is None:
            return None  # No solution exists

        solution_columns.append(x)

    # Transpose to get result matrix
    result_rows = []
    for i in range(A.cols):
        row_packed = 0
        for j in range(B.cols):
            if solution_columns[j][i]:
                row_packed |= 1 << j
        result_rows.append(row_packed)

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
    analysis["rank_nullity_check"] = analysis["rank"] + analysis["nullity"] == A.cols

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
    x = [0] * A.cols if x0 is None else x0[:]

    for iteration in range(max_iterations):
        # Compute residual r = b - Ax
        residual = b[:]

        for i in range(A.rows):
            row_packed = A.get_row_bitwise(i)
            dot_product = 0
            for j in range(A.cols):
                if (row_packed >> j) & 1:
                    dot_product ^= x[j]
            residual[i] ^= dot_product

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
    times = []

    for _ in range(num_trials):
        start_time = time.time()
        solve(A, b)
        elapsed = time.time() - start_time
        times.append(elapsed)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "total_time": np.sum(times),
        "trials": num_trials,
    }
