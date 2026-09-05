"""
Matrix Generators for Common GF(2) Structures
=============================================

Generators for structured matrices commonly used in coding theory,
quantum error correction, and cryptographic applications.

Generators:
- LDPC codes: Random, regular, structured
- Quantum codes: surface codes, CSS codes, hypergraph products, bicycle codes
- Classical codes: Hamming, repetition
- Structured: circulant, Toeplitz
- Random: Various sparsity patterns
"""

import random

from .sparse import SparseGF2Matrix, create_sparse_matrix


def _rng(seed: int | None) -> random.Random:
    """A private generator.

    ``random.seed(seed)`` reseeds the process-wide generator, so calling any
    seeded gf2 generator used to silently reset the caller's own random
    stream. A private ``random.Random`` keeps reproducibility without the
    side effect.
    """
    return random.Random(seed) if seed is not None else random.Random()


def identity(n: int) -> SparseGF2Matrix:
    """Create n×n identity matrix."""
    coordinates = [(i, i) for i in range(n)]
    return create_sparse_matrix(n, n, coordinates=coordinates)


def zeros(rows: int, cols: int) -> SparseGF2Matrix:
    """Create zero matrix."""
    return SparseGF2Matrix(rows, cols)


def ones(rows: int, cols: int) -> SparseGF2Matrix:
    """Create all-ones matrix."""
    # Build the packed rows directly. Materialising rows*cols coordinate
    # tuples costs ~60 bytes each, i.e. 6 GB for a 10000x10000 matrix.
    result = SparseGF2Matrix(rows, cols)
    if rows and cols:
        result.set_from_packed_rows([(1 << cols) - 1] * rows)
    return result


def random_sparse(rows: int, cols: int, density: float, seed: int | None = None) -> SparseGF2Matrix:
    """
    Generate random sparse binary matrix.

    Args:
        rows, cols: Matrix dimensions
        density: Fraction of entries that are 1 (0.0 to 1.0)
        seed: Random seed for reproducibility
    """
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be in [0, 1]")
    if rows <= 0 or cols <= 0:
        return SparseGF2Matrix(max(rows, 0), max(cols, 0))

    rng = _rng(seed)
    num_ones = int(rows * cols * density)

    # Sampling flat indices out of a range object never materialises the
    # cell list. The old path built every (i, j) tuple first, which is
    # O(rows*cols) memory before a single bit is chosen.
    flat = rng.sample(range(rows * cols), num_ones)
    packed = [0] * rows
    for index in flat:
        packed[index // cols] |= 1 << (index % cols)

    result = SparseGF2Matrix(rows, cols)
    result.set_from_packed_rows(packed)
    return result


def random_regular(
    rows: int, cols: int, row_weight: int, col_weight: int | None = None, seed: int | None = None
) -> SparseGF2Matrix:
    """
    Generate random regular binary matrix (constant row/column weights).

    Args:
        rows, cols: Matrix dimensions
        row_weight: Number of 1s per row
        col_weight: Number of 1s per column (if None, computed automatically)
        seed: Random seed
    """
    rng = _rng(seed)

    if col_weight is None:
        # Ensure matrix is consistent: rows * row_weight = cols * col_weight
        total_ones = rows * row_weight
        if total_ones % cols != 0:
            raise ValueError("Cannot create regular matrix with given parameters")
        col_weight = total_ones // cols

    # Verify consistency
    if rows * row_weight != cols * col_weight:
        raise ValueError("Row and column weights inconsistent")

    if row_weight > cols or (col_weight or 0) > rows:
        raise ValueError("weight exceeds the available number of positions")

    # Configuration-model pairing, retried until no row draws the same column
    # twice. A plain shuffle can pair a row with a repeated column; because
    # coordinates carry set semantics that silently produced a row of weight
    # below `row_weight`, so the matrix was not actually regular.
    edge_list = [i for i in range(rows) for _ in range(row_weight)]
    col_slots = [j for j in range(cols) for _ in range(col_weight)]

    for _attempt in range(200):
        rng.shuffle(col_slots)
        per_row: list[set[int]] = [set() for _ in range(rows)]
        clash = False
        for i, j in zip(edge_list, col_slots, strict=True):
            if j in per_row[i]:
                clash = True
                break
            per_row[i].add(j)
        if not clash:
            packed = [0] * rows
            for i, cols_set in enumerate(per_row):
                for j in cols_set:
                    packed[i] |= 1 << j
            result = SparseGF2Matrix(rows, cols)
            result.set_from_packed_rows(packed)
            return result

    raise RuntimeError("could not build a regular matrix with these weights after 200 attempts")


def circulant(first_row: list[int]) -> SparseGF2Matrix:
    """
    Create circulant matrix from first row.

    Args:
        first_row: First row of the circulant matrix
    """
    n = len(first_row)
    coordinates = []

    coordinates = [(i, j) for i in range(n) for j in range(n) if first_row[(j - i) % n] == 1]

    return create_sparse_matrix(n, n, coordinates=coordinates)


def circulant_random(n: int, weight: int, seed: int | None = None) -> SparseGF2Matrix:
    """Create random circulant matrix with given weight."""
    rng = _rng(seed)

    first_row = [0] * n
    positions = rng.sample(range(n), weight)
    for pos in positions:
        first_row[pos] = 1

    return circulant(first_row)


def toeplitz(first_row: list[int], first_col: list[int]) -> SparseGF2Matrix:
    """
    Create Toeplitz matrix.

    Args:
        first_row: First row
        first_col: First column (first_col[0] should equal first_row[0])
    """
    rows = len(first_col)
    cols = len(first_row)
    coordinates = []

    for i in range(rows):
        for j in range(cols):
            # Toeplitz: A[i,j] depends only on (i-j)
            if i - j >= 0:
                # Use first column
                if i - j < len(first_col) and first_col[i - j] == 1:
                    coordinates.append((i, j))
            else:
                # Use first row
                if j - i < len(first_row) and first_row[j - i] == 1:
                    coordinates.append((i, j))

    return create_sparse_matrix(rows, cols, coordinates=coordinates)


def hamming_matrix(r: int) -> SparseGF2Matrix:
    """
    Create parity check matrix for Hamming code.

    Args:
        r: Number of parity bits

    Returns:
        Hamming parity check matrix H (r × (2^r - 1))
    """
    n = (1 << r) - 1  # 2^r - 1
    coordinates = []

    coordinates = [
        (row, col - 1)
        for col in range(1, n + 1)
        for row in range(r)
        if (col >> row) & 1  # Check if bit 'row' is set in 'col'
    ]

    return create_sparse_matrix(r, n, coordinates=coordinates)


def ldpc_matrix(
    m: int,
    n: int,
    row_weight: int,
    col_weight: int | None = None,
    method: str = "random",
    seed: int | None = None,
) -> SparseGF2Matrix:
    """
    Generate LDPC (Low-Density Parity-Check) code matrix.

    Args:
        m: Number of parity checks (rows)
        n: Code length (columns)
        row_weight: Weight of each parity check
        col_weight: Weight of each variable (auto-computed if None)
        method: Generation method ("random", "structured", "progressive")
        seed: Random seed
    """
    if col_weight is None:
        total_ones = m * row_weight
        if total_ones % n != 0:
            raise ValueError("Cannot create regular LDPC with given parameters")
        col_weight = total_ones // n

    if method == "random":
        return random_regular(m, n, row_weight, col_weight, seed)

    elif method == "structured":
        return _ldpc_structured(m, n, row_weight, col_weight)

    elif method == "progressive":
        return _ldpc_progressive_edge_growth(m, n, row_weight, col_weight)

    else:
        raise ValueError(f"Unknown LDPC generation method: {method}")


def _ldpc_structured(m: int, n: int, row_weight: int, col_weight: int) -> SparseGF2Matrix:
    """Generate structured LDPC matrix using circulant blocks."""
    # Simplified structured LDPC using circulant submatrices
    coordinates = []

    if row_weight <= 0 or row_weight > m or row_weight > n:
        raise ValueError("row_weight must be in 1..min(m, n) for the structured method")

    block_size = n // row_weight
    period = m // row_weight

    for i in range(m):
        offset = i % period

        for j in range(row_weight):
            base_col = j * block_size
            col = (base_col + offset) % n
            coordinates.append((i, col))

    return create_sparse_matrix(m, n, coordinates=coordinates)


def _ldpc_progressive_edge_growth(m: int, n: int, row_weight: int, col_weight: int) -> SparseGF2Matrix:
    """Generate an LDPC matrix with the Progressive Edge Growth algorithm.

    For each new edge of a variable node, PEG runs a breadth-first search over
    the current Tanner graph and attaches to a check node that is as far from
    that variable node as possible -- an unreachable one if any remain -- so
    the shortest cycle it closes is as long as possible. Degree is the
    tie-break.

    Note:
        The previous implementation kept the edges in a flat list and, inside
        its innermost loop, re-scanned that whole list both to get a check
        node's degree and to test adjacency. That made edge insertion O(E) and
        the build O(n * col_weight * m * E), i.e. ~10^10 operations for a
        (500, 1000) code. Degrees and neighbour sets are now maintained
        incrementally, and the search is a real BFS.
    """
    if m <= 0 or n <= 0:
        return SparseGF2Matrix(max(m, 0), max(n, 0))
    if col_weight > m:
        raise ValueError("col_weight cannot exceed the number of check nodes")

    check_neighbors: list[set[int]] = [set() for _ in range(m)]  # check -> variables
    var_neighbors: list[set[int]] = [set() for _ in range(n)]  # variable -> checks
    check_degree = [0] * m

    for j in range(n):
        for edge in range(col_weight):
            candidates: list[int]
            if edge == 0:
                candidates = list(range(m))
            else:
                # BFS from variable j alternating variable -> check -> variable.
                reached_checks: set[int] = set()
                frontier = {j}
                seen_vars = {j}
                last_level: set[int] = set()
                while frontier:
                    level_checks = set()
                    for v in frontier:
                        level_checks |= var_neighbors[v] - reached_checks
                    if not level_checks:
                        break
                    reached_checks |= level_checks
                    last_level = level_checks
                    if len(reached_checks) == m:
                        break
                    next_vars = set()
                    for c in level_checks:
                        next_vars |= check_neighbors[c] - seen_vars
                    seen_vars |= next_vars
                    frontier = next_vars

                unreached = [c for c in range(m) if c not in reached_checks]
                # Prefer a check node the BFS never reached (no cycle at all);
                # otherwise the deepest level found (longest cycle).
                candidates = unreached if unreached else sorted(last_level)

            usable = [c for c in candidates if j not in check_neighbors[c] and check_degree[c] < row_weight]
            if not usable:
                usable = [c for c in range(m) if j not in check_neighbors[c]]
            if not usable:
                break

            best = min(usable, key=lambda c: (check_degree[c], c))
            check_neighbors[best].add(j)
            var_neighbors[j].add(best)
            check_degree[best] += 1

    packed = [0] * m
    for i, variables in enumerate(check_neighbors):
        for j in variables:
            packed[i] |= 1 << j

    result = SparseGF2Matrix(m, n)
    result.set_from_packed_rows(packed)
    return result


def repetition_matrix(n: int) -> SparseGF2Matrix:
    """Parity check matrix of the length-``n`` repetition code ((n-1) x n)."""
    if n < 2:
        raise ValueError("repetition code needs n >= 2")
    packed = [(0b11 << i) for i in range(n - 1)]
    H = SparseGF2Matrix(n - 1, n)
    H.set_from_packed_rows(packed)
    return H


def surface_code_matrix(distance: int, boundary: str = "open") -> tuple[SparseGF2Matrix, SparseGF2Matrix]:
    """
    Generate planar surface code parity check matrices.

    Built as the hypergraph product of two length-``distance`` repetition
    codes, which is exactly the planar (unrotated) surface code. That gives
    ``distance**2 + (distance - 1)**2`` data qubits, k = 1 logical qubit, and
    a guaranteed ``H_x @ H_z.T == 0``.

    Args:
        distance: Code distance (odd integer >= 3)
        boundary: Only "open" (planar) is implemented

    Returns:
        (H_x, H_z) - X and Z parity check matrices

    Note:
        Earlier releases returned a hand-rolled plaquette pattern on a
        ``distance x distance`` grid. It declared more stabiliser rows than it
        ever filled and did not satisfy the CSS commutation condition, so the
        matrices it produced were not a quantum code.
    """
    if distance % 2 == 0:
        raise ValueError("Distance must be odd")
    if distance < 3:
        raise ValueError("Distance must be at least 3")
    if boundary != "open":
        raise NotImplementedError("only the open (planar) boundary is implemented")

    H = repetition_matrix(distance)
    return hypergraph_product(H, H)


def css_code_matrix(H1: SparseGF2Matrix, H2: SparseGF2Matrix) -> tuple[SparseGF2Matrix, SparseGF2Matrix]:
    """
    Construct CSS (Calderbank-Shor-Steane) code from two classical codes.

    Args:
        H1, H2: Parity check matrices of two classical codes
                Must satisfy H1 * H2^T = 0

    Returns:
        (H_x, H_z) - Quantum CSS code parity check matrices
    """
    # For CSS codes: H_x = [H1 | 0], H_z = [0 | H2]

    n1, n2 = H1.cols, H2.cols
    total_qubits = n1 + n2

    # Construct H_x = [H1 | 0]
    x_coordinates = []
    for i in range(H1.rows):
        row_packed = H1.get_row_bitwise(i)
        x_coordinates.extend([(i, j) for j in range(n1) if (row_packed >> j) & 1])

    # Construct H_z = [0 | H2]
    z_coordinates = []
    for i in range(H2.rows):
        row_packed = H2.get_row_bitwise(i)
        z_coordinates.extend([(i, n1 + j) for j in range(n2) if (row_packed >> j) & 1])

    H_x = create_sparse_matrix(H1.rows, total_qubits, coordinates=x_coordinates)
    H_z = create_sparse_matrix(H2.rows, total_qubits, coordinates=z_coordinates)

    return H_x, H_z


def hypergraph_product(H1: SparseGF2Matrix, H2: SparseGF2Matrix) -> tuple[SparseGF2Matrix, SparseGF2Matrix]:
    """
    Tillich-Zemor hypergraph product of two classical codes.

    With H1 of shape (m1, n1) and H2 of shape (m2, n2) the product code has
    ``n1*n2 + m1*m2`` qubits and

        H_x = [ H1 (x) I_n2 | I_m1 (x) H2^T ]   shape (m1*n2, n)
        H_z = [ I_n1 (x) H2 | H1^T (x) I_m2 ]   shape (n1*m2, n)

    where (x) is the Kronecker product. The two blocks of H_x @ H_z^T are both
    H1 (x) H2^T, so they cancel over GF(2) and the CSS condition
    ``H_x @ H_z.T == 0`` holds by construction for any H1 and H2.

    Args:
        H1, H2: Classical parity check matrices

    Returns:
        (H_x, H_z) - Quantum hypergraph product code matrices

    Note:
        Earlier releases left ``z_coordinates`` empty, so H_z came back as an
        all-zero matrix and the returned pair was not a code.
    """
    m1, n1 = H1.rows, H1.cols
    m2, n2 = H2.rows, H2.cols

    h1_rows = H1.get_all_rows_bitwise()
    h2_rows = H2.get_all_rows_bitwise()

    # Column layout: sector A is (a, b) at a*n2 + b; sector B is (c, d) at
    # n1*n2 + c*m2 + d.
    sector_b = n1 * n2

    # H_x: row (c, b) -> c*n2 + b
    x_rows = [0] * (m1 * n2)
    for c in range(m1):
        row_h1 = h1_rows[c]
        for b in range(n2):
            acc = 0
            # H1 (x) I_n2 : column (a, b) set iff H1[c, a]
            remaining = row_h1
            while remaining:
                low = remaining & -remaining
                acc |= 1 << ((low.bit_length() - 1) * n2 + b)
                remaining ^= low
            # I_m1 (x) H2^T : column (c, d) set iff H2[d, b]
            for d in range(m2):
                if (h2_rows[d] >> b) & 1:
                    acc |= 1 << (sector_b + c * m2 + d)
            x_rows[c * n2 + b] = acc

    # H_z: row (a, d) -> a*m2 + d
    z_rows = [0] * (n1 * m2)
    for a in range(n1):
        for d in range(m2):
            acc = 0
            # I_n1 (x) H2 : column (a, b) set iff H2[d, b]
            remaining = h2_rows[d]
            while remaining:
                low = remaining & -remaining
                acc |= 1 << (a * n2 + (low.bit_length() - 1))
                remaining ^= low
            # H1^T (x) I_m2 : column (c, d) set iff H1[c, a]
            for c in range(m1):
                if (h1_rows[c] >> a) & 1:
                    acc |= 1 << (sector_b + c * m2 + d)
            z_rows[a * m2 + d] = acc

    total_qubits = n1 * n2 + m1 * m2
    H_x = SparseGF2Matrix(m1 * n2, total_qubits)
    H_x.set_from_packed_rows(x_rows)
    H_z = SparseGF2Matrix(n1 * m2, total_qubits)
    H_z.set_from_packed_rows(z_rows)

    return H_x, H_z


def bicycle_codes(block_size: int, circulant_A: list[int], circulant_B: list[int]) -> SparseGF2Matrix:
    """
    Generate bicycle LDPC codes (quantum LDPC codes).

        Args:
        block_size: Size of circulant blocks
        circulant_A, circulant_B: First rows of circulant matrices A and B

    Returns:
        Parity check matrix H = [A | B]
    """
    A = circulant(circulant_A)
    B = circulant(circulant_B)

    # Concatenate A and B horizontally
    coordinates = []

    # Add A block (left)
    for i in range(block_size):
        row_packed = A.get_row_bitwise(i)
        coordinates.extend([(i, j) for j in range(block_size) if (row_packed >> j) & 1])

    # Add B block (right)
    for i in range(block_size):
        row_packed = B.get_row_bitwise(i)
        coordinates.extend([(i, block_size + j) for j in range(block_size) if (row_packed >> j) & 1])

    return create_sparse_matrix(block_size, 2 * block_size, coordinates=coordinates)
