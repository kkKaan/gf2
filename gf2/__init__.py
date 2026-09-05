"""
gf2: Binary Linear Algebra over GF(2)
======================================

High-performance binary matrix operations with optimized sparse and dense
storage, solvers, and generators. Applications include coding theory,
cryptography, network coding, and quantum error correction.

Quick start:
    >>> from gf2.generators import identity
    >>> from gf2.core import rank
    >>> I = identity(8)
    >>> rank(I)
    8
"""

from .core import *
from .generators import *
from .solvers import *
from .sparse import *

__all__ = [
    # Core matrix classes
    "SparseGF2Matrix",
    "DenseGF2Matrix",
    # Basic operations
    "add",
    "multiply",
    "transpose",
    "rank",
    "det",
    "trace",
    "is_invertible",
    # Linear systems
    "solve",
    "nullspace",
    "nullspace_bitwise",
    "nullspace_fast",
    "inverse",
    "lu_decomposition",
    "least_squares",
    "kernel",
    "image",
    "rank_nullity_theorem",
    "solve_multiple_rhs",
    "iterative_refinement",
    "benchmark_solver",
    # Matrix generators
    "identity",
    "zeros",
    "ones",
    "random_sparse",
    "random_regular",
    "circulant",
    "circulant_random",
    "toeplitz",
    "vandermonde",
    "ldpc_matrix",
    "hamming_matrix",
    "bch_matrix",
    # Advanced properties
    "reduced_row_echelon_form",
    "matrix_power",
    "characteristic_polynomial",
    "minimal_polynomial",
    "matrix_norm",
    "condition_number",
    # Factory
    "create_sparse_matrix",
]

__version__ = "0.1.0"
