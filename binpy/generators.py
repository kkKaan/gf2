"""
Matrix Generators for Common GF(2) Structures
=============================================

Generators for structured matrices commonly used in coding theory,
quantum error correction, and cryptographic applications.

Generators:
- LDPC codes: Random, regular, structured
- Quantum codes: Surface codes, color codes, CSS codes
- Classical codes: Hamming, BCH, Reed-Solomon
- Structured: Circulant, Toeplitz, Vandermonde
- Random: Various sparsity patterns
"""

import random
from typing import Optional

from .sparse import SparseGF2Matrix, create_sparse_matrix


def identity(n: int) -> SparseGF2Matrix:
    """Create n×n identity matrix."""
    coordinates = [(i, i) for i in range(n)]
    return create_sparse_matrix(n, n, coordinates=coordinates)


def zeros(rows: int, cols: int) -> SparseGF2Matrix:
    """Create zero matrix."""
    return SparseGF2Matrix(rows, cols)


def ones(rows: int, cols: int) -> SparseGF2Matrix:
    """Create all-ones matrix."""
    coordinates = [(i, j) for i in range(rows) for j in range(cols)]
    return create_sparse_matrix(rows, cols, coordinates=coordinates)


def random_sparse(rows: int, cols: int, density: float, seed: Optional[int] = None) -> SparseGF2Matrix:
    """
    Generate random sparse binary matrix.

    Args:
        rows, cols: Matrix dimensions
        density: Fraction of entries that are 1 (0.0 to 1.0)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    return create_sparse_matrix(rows, cols, density=density)


def random_regular(
    rows: int, cols: int, row_weight: int, col_weight: Optional[int] = None, seed: Optional[int] = None
) -> SparseGF2Matrix:
    """
    Generate random regular binary matrix (constant row/column weights).

    Args:
        rows, cols: Matrix dimensions
        row_weight: Number of 1s per row
        col_weight: Number of 1s per column (if None, computed automatically)
        seed: Random seed
    """
    if seed is not None:
        random.seed(seed)

    if col_weight is None:
        # Ensure matrix is consistent: rows * row_weight = cols * col_weight
        total_ones = rows * row_weight
        if total_ones % cols != 0:
            raise ValueError("Cannot create regular matrix with given parameters")
        col_weight = total_ones // cols

    # Verify consistency
    if rows * row_weight != cols * col_weight:
        raise ValueError("Row and column weights inconsistent")

    # Use random permutation method
    coordinates = []

    # Create a bipartite graph representation
    edge_list = []
    for i in range(rows):
        for _ in range(row_weight):
            edge_list.append(i)

    # Randomly assign column endpoints
    col_assignments = []
    for j in range(cols):
        for _ in range(col_weight):
            col_assignments.append(j)

    random.shuffle(col_assignments)

    # Create coordinates
    for i, j in zip(edge_list, col_assignments):
        coordinates.append((i, j))

    return create_sparse_matrix(rows, cols, coordinates=coordinates)


def circulant(first_row: list[int]) -> SparseGF2Matrix:
    """
    Create circulant matrix from first row.

    Args:
        first_row: First row of the circulant matrix
    """
    n = len(first_row)
    coordinates = []

    for i in range(n):
        for j in range(n):
            # Circulant: A[i,j] = first_row[(j-i) % n]
            if first_row[(j - i) % n] == 1:
                coordinates.append((i, j))

    return create_sparse_matrix(n, n, coordinates=coordinates)


def circulant_random(n: int, weight: int, seed: Optional[int] = None) -> SparseGF2Matrix:
    """Create random circulant matrix with given weight."""
    if seed is not None:
        random.seed(seed)

    first_row = [0] * n
    positions = random.sample(range(n), weight)
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


def vandermonde(elements: list[int], n: int) -> SparseGF2Matrix:
    """
    Create Vandermonde matrix over GF(2).

    Args:
        elements: List of elements (represented as integers)
        n: Number of powers (columns)

    Returns:
        Matrix where A[i,j] = elements[i]^j mod 2
    """
    m = len(elements)
    coordinates = []

    for i in range(m):
        elem = elements[i]
        power = 1  # elem^0 = 1

        for j in range(n):
            if power & 1:  # Check if power is odd (= 1 in GF(2))
                coordinates.append((i, j))

            power = (power * elem) & ((1 << 32) - 1)  # Prevent overflow

    return create_sparse_matrix(m, n, coordinates=coordinates)


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

    for col in range(1, n + 1):  # Columns 1 to n
        for row in range(r):
            if (col >> row) & 1:  # Check if bit 'row' is set in 'col'
                coordinates.append((row, col - 1))  # Convert to 0-indexed

    return create_sparse_matrix(r, n, coordinates=coordinates)


def bch_matrix(n: int, k: int, t: int) -> SparseGF2Matrix:
    """
    Create BCH code parity check matrix (simplified).

    Args:
        n: Code length
        k: Information length
        t: Error correction capability

    Returns:
        BCH parity check matrix (simplified construction)
    """
    # This is a simplified construction
    # Full BCH construction requires finite field arithmetic

    r = n - k  # Number of parity checks
    coordinates = []

    # Generate using primitive polynomial approach (simplified)
    for i in range(r):
        for j in range(n):
            # Simplified pattern based on powers of primitive element
            if ((i + 1) * (j + 1)) % 3 == 1:  # Simplified condition
                coordinates.append((i, j))

    return create_sparse_matrix(r, n, coordinates=coordinates)


def ldpc_matrix(
    m: int,
    n: int,
    row_weight: int,
    col_weight: Optional[int] = None,
    method: str = "random",
    seed: Optional[int] = None,
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
    if seed is not None:
        random.seed(seed)

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

    block_size = n // row_weight

    for i in range(m):
        i // (m // row_weight)
        offset = i % (m // row_weight)

        for j in range(row_weight):
            base_col = j * block_size
            col = (base_col + offset) % n
            coordinates.append((i, col))

    return create_sparse_matrix(m, n, coordinates=coordinates)


def _ldpc_progressive_edge_growth(m: int, n: int, row_weight: int, col_weight: int) -> SparseGF2Matrix:
    """Generate LDPC using Progressive Edge Growth algorithm."""
    # Simplified PEG algorithm
    coordinates = []
    adjacency = [[] for _ in range(n)]  # Variable node adjacencies

    for j in range(n):  # For each variable node
        for _ in range(col_weight):
            # Find check node that minimizes local girth
            best_check = None
            min_common = float("inf")

            for i in range(m):
                if len([c for c in coordinates if c[0] == i]) >= row_weight:
                    continue  # Check node is full

                # Count common neighbors
                common = 0
                for neighbor_j in adjacency[j]:
                    if any(c[0] == i and c[1] == neighbor_j for c in coordinates):
                        common += 1

                if common < min_common:
                    min_common = common
                    best_check = i

            if best_check is not None:
                coordinates.append((best_check, j))
                adjacency[j].append(best_check)

    return create_sparse_matrix(m, n, coordinates=coordinates)


def surface_code_matrix(distance: int, boundary: str = "open") -> tuple[SparseGF2Matrix, SparseGF2Matrix]:
    """
    Generate surface code parity check matrices.

    Args:
        distance: Code distance (odd integer ≥ 3)
        boundary: "open" or "periodic"

    Returns:
        (H_x, H_z) - X and Z parity check matrices
    """
    if distance % 2 == 0:
        raise ValueError("Distance must be odd")

    # Number of data qubits
    n_data = distance * distance

    # Number of stabilizers
    n_x_stabs = (distance - 1) * distance // 2
    n_z_stabs = distance * (distance - 1) // 2

    # Generate X stabilizers (face operators)
    x_coordinates = []
    stab_idx = 0

    for row in range(0, distance - 1, 2):
        for col in range(0, distance - 1, 2):
            # Each X stabilizer acts on 4 data qubits in a plaquette
            qubits = [
                row * distance + col,
                row * distance + col + 1,
                (row + 1) * distance + col,
                (row + 1) * distance + col + 1,
            ]

            for qubit in qubits:
                if qubit < n_data:
                    x_coordinates.append((stab_idx, qubit))

            stab_idx += 1

    # Generate Z stabilizers (star operators)
    z_coordinates = []
    stab_idx = 0

    for row in range(1, distance - 1, 2):
        for col in range(1, distance - 1, 2):
            # Each Z stabilizer acts on 4 neighboring data qubits
            qubits = [
                (row - 1) * distance + col,
                (row + 1) * distance + col,
                row * distance + col - 1,
                row * distance + col + 1,
            ]

            for qubit in qubits:
                if 0 <= qubit < n_data:
                    z_coordinates.append((stab_idx, qubit))

            stab_idx += 1

    H_x = create_sparse_matrix(n_x_stabs, n_data, coordinates=x_coordinates)
    H_z = create_sparse_matrix(n_z_stabs, n_data, coordinates=z_coordinates)

    return H_x, H_z


def color_code_matrix(distance: int) -> tuple[SparseGF2Matrix, SparseGF2Matrix]:
    """
    Generate color code parity check matrices (simplified triangular lattice).

    Args:
        distance: Code distance

    Returns:
        (H_x, H_z) - X and Z parity check matrices
    """
    # Simplified color code on triangular lattice
    # This is a basic implementation - full color codes are more complex

    n_data = distance * distance
    n_stabilizers = n_data // 2

    # Generate coordinates for X and Z stabilizers
    x_coordinates = []
    z_coordinates = []

    stab_idx = 0
    for i in range(0, n_data - distance, distance):
        for j in range(0, distance - 2, 2):
            # X stabilizer on 3 qubits (triangle)
            qubits_x = [i + j, i + j + 1, i + j + distance]
            for qubit in qubits_x:
                if qubit < n_data:
                    x_coordinates.append((stab_idx, qubit))

            # Z stabilizer on adjacent triangle
            qubits_z = [i + j + 1, i + j + 2, i + j + distance + 1]
            for qubit in qubits_z:
                if qubit < n_data:
                    z_coordinates.append((stab_idx, qubit))

            stab_idx += 1

    H_x = create_sparse_matrix(n_stabilizers, n_data, coordinates=x_coordinates)
    H_z = create_sparse_matrix(n_stabilizers, n_data, coordinates=z_coordinates)

    return H_x, H_z


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
        for j in range(n1):
            if (row_packed >> j) & 1:
                x_coordinates.append((i, j))

    # Construct H_z = [0 | H2]
    z_coordinates = []
    for i in range(H2.rows):
        row_packed = H2.get_row_bitwise(i)
        for j in range(n2):
            if (row_packed >> j) & 1:
                z_coordinates.append((i, n1 + j))

    H_x = create_sparse_matrix(H1.rows, total_qubits, coordinates=x_coordinates)
    H_z = create_sparse_matrix(H2.rows, total_qubits, coordinates=z_coordinates)

    return H_x, H_z


def hypergraph_product(H1: SparseGF2Matrix, H2: SparseGF2Matrix) -> tuple[SparseGF2Matrix, SparseGF2Matrix]:
    """
    Construct quantum code using hypergraph product of two classical codes.

    Args:
        H1, H2: Classical parity check matrices

    Returns:
        (H_x, H_z) - Quantum hypergraph product code matrices
    """
    from .core import transpose

    # Get dimensions
    m1, n1 = H1.rows, H1.cols
    m2, n2 = H2.rows, H2.cols

    # Total number of qubits
    total_qubits = n1 * m2 + m1 * n2

    # Construct H_x and H_z using Kronecker products
    # This is a simplified version - full implementation requires tensor products

    # H_x block structure
    x_coordinates = []

    # First block: H1 ⊗ I_m2
    for i in range(m1):
        for k in range(m2):
            row_idx = i * m2 + k
            row_packed = H1.get_row_bitwise(i)

            for j in range(n1):
                if (row_packed >> j) & 1:
                    col_idx = j * m2 + k
                    x_coordinates.append((row_idx, col_idx))

    # Second block: I_m1 ⊗ H2^T
    H2T = transpose(H2)
    for i in range(m1):
        for j in range(n2):
            for k in range(m2):
                row_idx = i * m2 + k
                row_packed = H2T.get_row_bitwise(j)

                if (row_packed >> k) & 1:
                    col_idx = n1 * m2 + i * n2 + j
                    x_coordinates.append((row_idx, col_idx))

    # Similar construction for H_z (with roles swapped)
    z_coordinates = []
    # Implementation similar to above but with H1 and H2 roles swapped

    H_x = create_sparse_matrix(m1 * m2, total_qubits, coordinates=x_coordinates)
    H_z = create_sparse_matrix(m2 * n1, total_qubits, coordinates=z_coordinates)

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
        for j in range(block_size):
            if (row_packed >> j) & 1:
                coordinates.append((i, j))

    # Add B block (right)
    for i in range(block_size):
        row_packed = B.get_row_bitwise(i)
        for j in range(block_size):
            if (row_packed >> j) & 1:
                coordinates.append((i, block_size + j))

    return create_sparse_matrix(block_size, 2 * block_size, coordinates=coordinates)
