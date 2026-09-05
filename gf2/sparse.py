"""
Sparse Matrix Storage for GF(2) Operations
==========================================

Optimized storage formats for binary matrices with focus on memory efficiency
and fast bitwise operations. Supports multiple sparse formats depending on
matrix characteristics and use cases.

Storage Formats:
- CSR (Compressed Sparse Row): General sparse matrices
- Bit-packed: Dense matrices with efficient bit storage
- Structured: LDPC, circulant, and other structured matrices
- Hybrid: Automatic format selection based on sparsity
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def _int_from_words(words: np.ndarray) -> int:
    """Concatenate little-endian uint64 words into one Python int.

    ``int.from_bytes`` does this in C. Building it with a ``|=`` loop is
    quadratic in the word count because each step reallocates the whole int.
    """
    return int.from_bytes(words.tobytes(), "little")


def _set_bit_positions(value: int) -> list[int]:
    """Indices of the set bits of ``value``, cheapest-first.

    Isolating the low bit with ``value & -value`` costs one machine-word pass
    per set bit. Shifting right one bit at a time costs one pass per *column*,
    which is why the previous ``while value: value >>= 1`` loops were
    quadratic on wide rows.
    """
    out = []
    while value:
        low = value & -value
        out.append(low.bit_length() - 1)
        value ^= low
    return out


@dataclass
class SparseStats:
    """Statistics about sparse matrix storage efficiency."""

    nnz: int  # number of non-zeros
    density: float  # nnz / (rows * cols)
    memory_bytes: int
    compression_ratio: float  # vs dense storage


class SparseGF2Matrix:
    """
    Memory-optimized sparse binary matrix with multiple storage formats.

    Automatically selects optimal storage format based on matrix characteristics:
    - Very sparse (< 1%): CSR format with bit-packed indices
    - Moderately sparse (1-50%): CSR with standard indices
    - Dense (> 50%): Bit-packed dense format
    - Structured: Special handling for circulant, LDPC, etc.
    """

    def __init__(self, rows: int, cols: int, data=None, format_hint: str = "auto"):
        """
        Initialize sparse GF(2) matrix.

        Args:
            rows: Number of rows
            cols: Number of columns
            data: Initial data (list of lists, CSR arrays, or coordinate list)
            format_hint: Storage format ("csr", "bitpacked", "structured", "auto")
        """
        self.rows = rows
        self.cols = cols
        self.nnz = 0
        self.format = "empty"

        # Storage arrays (only one will be used based on format)
        # Annotate as Optional so static type checker knows possible types
        self.csr_row_ptr: Sequence[int] | np.ndarray | None = None
        self.csr_col_ind: Sequence[int] | np.ndarray | None = None
        self.bitpacked_rows: np.ndarray | None = None
        self.structured_params: dict | None = None

        # OPTIMIZATION: Cache for packed row representation
        self._packed_rows_cache: list[int] | None = None

        if data is not None:
            self._load_data(data, format_hint)

    def _load_data(self, data, format_hint: str):
        """Load data and select optimal storage format."""
        if isinstance(data, list) and len(data) > 0:
            # Convert from list of lists
            self._from_dense(data, format_hint)
        elif isinstance(data, tuple) and len(data) >= 2:
            # Assume (row_indices, col_indices) or (row_indices, col_indices, values) coordinate format
            self._from_coordinates(data[0], data[1], format_hint)
        else:
            raise ValueError("Unsupported data format")

    def _from_dense(self, matrix: list[list[int]], format_hint: str):
        """Convert from dense matrix representation."""
        # OPTIMIZATION: Skip format detection if hint provided
        if format_hint != "auto":
            self.format = format_hint
            # Quick nnz count
            self.nnz = sum(sum(row) for row in matrix)
        else:
            # Count non-zeros and analyze structure
            self.nnz = sum(sum(row) for row in matrix)
            density = self.nnz / (self.rows * self.cols) if self.rows * self.cols > 0 else 0

            # Select optimal format
            if density < 0.01:  # Very sparse
                self.format = "csr_compact"
            elif density < 0.5:  # Moderately sparse
                self.format = "csr"
            else:  # Dense
                self.format = "bitpacked"

        # Convert to selected format
        if self.format == "csr" or self.format == "csr_compact":
            self._to_csr(matrix)
        elif self.format == "bitpacked":
            self._to_bitpacked(matrix)

    def _from_coordinates(self, rows: list[int], cols: list[int], format_hint: str):
        """Convert from coordinate (COO) format.

        Coordinates carry *set* semantics: listing (i, j) twice sets the bit
        once. Duplicates used to be stored twice, which left ``nnz`` and every
        density-derived decision reporting more entries than the matrix holds.
        """
        if len(rows) != len(cols):
            raise ValueError("row and column index lists must be the same length")
        for i, j in zip(rows, cols, strict=False):
            if not (0 <= i < self.rows and 0 <= j < self.cols):
                raise IndexError(f"coordinate ({i}, {j}) outside {self.rows}x{self.cols} matrix")

        unique = sorted(set(zip(rows, cols, strict=False)))
        rows = [i for i, _ in unique]
        cols = [j for _, j in unique]

        self.nnz = len(rows)
        cells = self.rows * self.cols
        density = self.nnz / cells if cells > 0 else 0.0

        # Auto-select format based on density
        if format_hint == "auto":
            self.format = "csr_compact" if density < 0.05 else "csr"
        else:
            self.format = format_hint

        # Convert coordinates to selected format
        if self.format in ["csr", "csr_compact"]:
            self._coo_to_csr(rows, cols)
        elif self.format == "bitpacked":
            self._coo_to_bitpacked(rows, cols)

    def _to_csr(self, matrix: list[list[int]]):
        """Convert to Compressed Sparse Row format."""
        self.csr_row_ptr = [0]
        self.csr_col_ind = []

        for _i, row in enumerate(matrix):
            row_nnz = 0
            for j, val in enumerate(row):
                if val & 1:  # Only store 1s
                    self.csr_col_ind.append(j)
                    row_nnz += 1
            self.csr_row_ptr.append(self.csr_row_ptr[-1] + row_nnz)

        self.nnz = len(self.csr_col_ind)
        # A mutation path may have called us with a stale cache in place.
        self._packed_rows_cache = None

        # Use compact indices for very sparse matrices
        if self.format == "csr_compact" and self.cols <= 65535:
            # Use 16-bit indices instead of 32-bit
            self.csr_col_ind = np.array(self.csr_col_ind, dtype=np.uint16)
        else:
            self.csr_col_ind = np.array(self.csr_col_ind, dtype=np.uint32)

        self.csr_row_ptr = np.array(self.csr_row_ptr, dtype=np.uint32)

    def _to_bitpacked(self, matrix: list[list[int]]):
        """Convert to bit-packed dense format."""
        # OPTIMIZATION: Use fast row packing with Python integers first
        words_per_row = (self.cols + 63) // 64
        self.bitpacked_rows = np.zeros((self.rows, words_per_row), dtype=np.uint64)

        # OPTIMIZATION: Build packed rows cache at the same time
        packed_cache = []

        # Pack the whole matrix at once. np.packbits is a C loop; the previous
        # per-element ``row_packed |= 1 << j`` was O(cols^2 / 64) per row
        # because every OR reallocated the growing integer.
        dense = np.zeros((self.rows, words_per_row * 64), dtype=np.uint8)
        if matrix:
            supplied = np.asarray(matrix, dtype=np.uint8) & 1
            dense[: supplied.shape[0], : supplied.shape[1]] = supplied
        # packbits is MSB-first inside each byte; reverse each byte's bits so
        # that column j lands on bit j.
        packed = np.packbits(
            dense.reshape(self.rows, words_per_row * 8, 8)[:, :, ::-1].reshape(self.rows, -1),
            axis=1,
        )
        self.bitpacked_rows = np.ascontiguousarray(packed.view(np.uint64))
        packed_cache = [_int_from_words(self.bitpacked_rows[i]) for i in range(self.rows)]

        # Cache the packed rows for fast access
        self._packed_rows_cache = packed_cache

    def _coo_to_csr(self, row_indices: list[int], col_indices: list[int]):
        """Convert coordinate format to CSR."""
        # _from_coordinates already de-duplicated and sorted by (row, col).
        sorted_pairs = sorted(zip(row_indices, col_indices, strict=False))

        self.csr_row_ptr = [0] * (self.rows + 1)
        self.csr_col_ind = []

        current_row = 0
        for row, col in sorted_pairs:
            # Fill empty rows
            while current_row < row:
                current_row += 1
                self.csr_row_ptr[current_row] = len(self.csr_col_ind)

            if current_row == row:
                self.csr_col_ind.append(col)

        # Fill remaining row pointers
        for i in range(current_row + 1, self.rows + 1):
            self.csr_row_ptr[i] = len(self.csr_col_ind)

        # Convert to appropriate numpy arrays
        if self.format == "csr_compact" and self.cols <= 65535:
            self.csr_col_ind = np.array(self.csr_col_ind, dtype=np.uint16)
        else:
            self.csr_col_ind = np.array(self.csr_col_ind, dtype=np.uint32)
        self.csr_row_ptr = np.array(self.csr_row_ptr, dtype=np.uint32)

    def _coo_to_bitpacked(self, row_indices: list[int], col_indices: list[int]):
        """Convert coordinate format to bit-packed."""
        words_per_row = (self.cols + 63) // 64
        self.bitpacked_rows = np.zeros((self.rows, words_per_row), dtype=np.uint64)

        for row, col in zip(row_indices, col_indices, strict=False):
            word_idx = col // 64
            bit_idx = col % 64
            # Read-modify-write using Python int to avoid numpy ufunc casting errors
            current = int(self.bitpacked_rows[row, word_idx])
            self.bitpacked_rows[row, word_idx] = current | (1 << bit_idx)

    def get_all_rows_bitwise(self) -> list[int]:
        """
        Get all rows as packed integers (cached for performance).

        Returns:
            List of packed row integers
        """
        if self._packed_rows_cache is None:
            self._packed_rows_cache = [self.get_row_bitwise(i) for i in range(self.rows)]
        return self._packed_rows_cache

    def rows_bitwise(self) -> list[int]:
        """A private copy of the packed rows, safe for in-place elimination."""
        return self.get_all_rows_bitwise()[:]

    def get_row_bitwise(self, row_idx: int) -> int:
        """
        Get row as packed integer for bitwise operations.
        Optimized for your existing bitwise algorithms.
        """
        # OPTIMIZATION: Use cache if available
        if self._packed_rows_cache is not None:
            return self._packed_rows_cache[row_idx]

        if self.format == "bitpacked":
            # Already bit-packed, just need to combine words
            if self.bitpacked_rows is None:
                raise ValueError("Bitpacked rows not initialized")
            if self.cols <= 64:
                return int(self.bitpacked_rows[row_idx, 0])
            return _int_from_words(self.bitpacked_rows[row_idx])

        elif self.format in ["csr", "csr_compact"]:
            # Convert CSR row to packed integer
            if self.csr_row_ptr is None or self.csr_col_ind is None:
                raise ValueError("CSR format data not initialized")
            result = 0
            start = self.csr_row_ptr[row_idx]
            end = self.csr_row_ptr[row_idx + 1]

            for col_idx in self.csr_col_ind[start:end]:
                # Convert numpy types to Python int to avoid overflow issues
                result |= 1 << int(col_idx)

            return result

        elif self.format == "empty":
            # Empty matrix - all rows are zero
            return 0

        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def set_from_packed_rows(self, packed_rows: list[int]):
        """
        Set matrix from list of packed row integers.
        Efficient interface for your existing algorithms.
        """
        if len(packed_rows) != self.rows:
            raise ValueError(f"expected {self.rows} packed rows, got {len(packed_rows)}")
        # Bits at or beyond ``cols`` have no column to live in. Silently keeping
        # them in the cache made get_row_bitwise and get_bit disagree, and
        # inflated nnz / density / rank.
        width_mask = (1 << self.cols) - 1
        packed_rows = [int(row) & width_mask for row in packed_rows]

        # Analyze sparsity to select format
        total_bits = len(packed_rows) * self.cols
        set_bits = sum(row.bit_count() for row in packed_rows)
        density = set_bits / total_bits if total_bits > 0 else 0

        self.nnz = set_bits

        if density < 0.1:
            self.format = "csr_compact" if self.cols <= 65535 else "csr"
            self._packed_to_csr(packed_rows)
        else:
            self.format = "bitpacked"
            self._packed_to_bitpacked(packed_rows)

        # OPTIMIZATION: Cache the packed rows immediately
        self._packed_rows_cache = packed_rows[:]

    def _packed_to_csr(self, packed_rows: list[int]):
        """Convert packed rows to CSR format."""
        self.csr_row_ptr = [0]
        self.csr_col_ind = []

        for row_val in packed_rows:
            self.csr_col_ind.extend(_set_bit_positions(row_val))
            self.csr_row_ptr.append(len(self.csr_col_ind))

        # Convert to numpy arrays
        if self.format == "csr_compact":
            self.csr_col_ind = np.array(self.csr_col_ind, dtype=np.uint16)
        else:
            self.csr_col_ind = np.array(self.csr_col_ind, dtype=np.uint32)
        self.csr_row_ptr = np.array(self.csr_row_ptr, dtype=np.uint32)

    def _packed_to_bitpacked(self, packed_rows: list[int]):
        """Convert packed rows to bit-packed format."""
        words_per_row = (self.cols + 63) // 64
        self.bitpacked_rows = np.zeros((self.rows, words_per_row), dtype=np.uint64)

        nbytes = words_per_row * 8
        for i, row_val in enumerate(packed_rows):
            row_val = int(row_val)
            # to_bytes slices the integer in one C pass; the previous loop
            # shifted the full-width integer once per word, which is quadratic.
            self.bitpacked_rows[i] = np.frombuffer(row_val.to_bytes(nbytes, "little"), dtype=np.uint64)

    def memory_usage(self) -> SparseStats:
        """Calculate memory usage statistics."""
        if self.format in ["csr", "csr_compact"]:
            # CSR: row_ptr + col_ind arrays
            if self.csr_row_ptr is None or self.csr_col_ind is None:
                raise ValueError("CSR format data not initialized")
            # Handle both numpy arrays and sequences
            if hasattr(self.csr_row_ptr, "nbytes"):
                row_ptr_bytes = self.csr_row_ptr.nbytes
            else:
                row_ptr_bytes = len(self.csr_row_ptr) * 4  # Assume 32-bit ints
            if hasattr(self.csr_col_ind, "nbytes"):
                col_ind_bytes = self.csr_col_ind.nbytes
            else:
                col_ind_bytes = len(self.csr_col_ind) * 4  # Assume 32-bit ints
            total_bytes = row_ptr_bytes + col_ind_bytes

        elif self.format == "bitpacked":
            # Bit-packed: just the packed array
            if self.bitpacked_rows is None:
                raise ValueError("Bitpacked rows not initialized")
            total_bytes = self.bitpacked_rows.nbytes

        else:
            total_bytes = 0

        # Compare with dense storage (1 byte per element)
        dense_bytes = self.rows * self.cols
        compression_ratio = dense_bytes / total_bytes if total_bytes > 0 else 1.0

        return SparseStats(
            nnz=self.nnz,
            density=self.nnz / (self.rows * self.cols) if self.rows * self.cols > 0 else 0,
            memory_bytes=total_bytes,
            compression_ratio=compression_ratio,
        )

    def set_bit(self, row: int, col: int):
        """Set bit at (row, col) to 1."""
        self._assign_bit(row, col, 1)

    def clear_bit(self, row: int, col: int):
        """Set bit at (row, col) to 0."""
        self._assign_bit(row, col, 0)

    def _assign_bit(self, row: int, col: int, value: int):
        """Write one bit, keeping every representation consistent.

        Goes through the packed rows rather than a dense round-trip. The old
        path rebuilt an entire dense list of lists per write (O(rows*cols)),
        never refreshed ``_packed_rows_cache`` on the CSR branch -- so later
        reads returned the pre-write row -- and dropped the write entirely on
        an "empty"-format matrix, because ``_from_dense`` has no branch for it.
        """
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError(f"({row}, {col}) outside {self.rows}x{self.cols} matrix")

        packed = list(self.get_all_rows_bitwise())
        if value & 1:
            packed[row] |= 1 << col
        else:
            packed[row] &= ~(1 << col)
        self.set_from_packed_rows(packed)

    def get(self, row: int, col: int) -> int:
        """Get element at (row, col). Alias for get_bit for compatibility."""
        return self.get_bit(row, col)

    def set(self, row: int, col: int, value: int):
        """Set element at (row, col) to 0 or 1."""
        self._assign_bit(row, col, value)

    def get_bit(self, row: int, col: int) -> int:
        """Get bit at (row, col)."""
        if self.format == "bitpacked":
            if self.bitpacked_rows is None:
                raise ValueError("Bitpacked rows not initialized")
            word_idx = col // 64
            bit_idx = col % 64
            # Convert to Python int first to avoid numpy ufunc casting issues
            value = int(self.bitpacked_rows[row, word_idx])
            return (value >> bit_idx) & 1
        elif self.format in ["csr", "csr_compact"]:
            # Check if bit is set in CSR format
            if self.csr_row_ptr is None or self.csr_col_ind is None:
                raise ValueError("CSR format data not initialized")
            start = self.csr_row_ptr[row]
            end = self.csr_row_ptr[row + 1]
            for idx in range(start, end):
                if self.csr_col_ind[idx] == col:
                    return 1
            return 0
        elif self.format == "empty":
            return 0
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def to_dense(self) -> list[list[int]]:
        """Convert back to dense format for debugging/testing."""
        if self.rows == 0 or self.cols == 0:
            return [[0] * self.cols for _ in range(self.rows)]

        # One C-level unpack for the whole matrix. Both a shift per element and
        # a set-bit walk per row spend their time in the interpreter instead.
        bits = np.unpackbits(self.packed_u64().view(np.uint8), axis=1, bitorder="little")
        return bits[:, : self.cols].astype(np.int64).tolist()

    def packed_u64(self) -> np.ndarray:
        """Rows as a contiguous (rows, words) little-endian uint64 array.

        This is the layout the vectorised GF(2) kernels in :mod:`gf2.core`
        consume, so it is materialised once here rather than per call site.
        """
        words = (self.cols + 63) // 64
        if (
            self.format == "bitpacked"
            and self.bitpacked_rows is not None
            and self.bitpacked_rows.shape[1] == words
        ):
            return self.bitpacked_rows
        out = np.zeros((self.rows, max(words, 1)), dtype=np.uint64)
        nbytes = max(words, 1) * 8
        for i, value in enumerate(self.get_all_rows_bitwise()):
            out[i] = np.frombuffer(int(value).to_bytes(nbytes, "little"), dtype=np.uint64)
        return out

    def invalidate_cache(self) -> None:
        """Drop the packed-row cache. Call after mutating the backing store."""
        self._packed_rows_cache = None

    def __repr__(self):
        stats = self.memory_usage()
        return (
            f"SparseGF2Matrix({self.rows}x{self.cols}, "
            f"nnz={stats.nnz}, density={stats.density:.3f}, "
            f"format={self.format}, memory={stats.memory_bytes}B, "
            f"compression={stats.compression_ratio:.1f}x)"
        )


class DenseGF2Matrix:
    """
    Bit-packed dense matrix for cases where sparsity doesn't help.
    Uses 1 bit per element packed into 64-bit words.
    """

    def __init__(self, rows: int, cols: int, data=None):
        self.rows = rows
        self.cols = cols

        # Pack into 64-bit words
        self.words_per_row = (cols + 63) // 64
        self.data = np.zeros((rows, self.words_per_row), dtype=np.uint64)

        if data is not None:
            self._load_data(data)

    def _load_data(self, matrix: list[list[int]]):
        """Load from dense matrix (list of lists)."""
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val & 1:
                    self.set_bit(i, j)

    def set_bit(self, row: int, col: int):
        """Set bit at (row, col) to 1 using safe read-modify-write."""
        word_idx = col // 64
        bit_idx = col % 64
        # Read-modify-write using Python int to avoid numpy ufunc casting issues
        current = int(self.data[row, word_idx])
        self.data[row, word_idx] = current | (1 << bit_idx)

    def get_bit(self, row: int, col: int) -> int:
        """Get bit at (row, col)."""
        word_idx = col // 64
        bit_idx = col % 64
        # Convert to Python int first to avoid numpy ufunc casting during shifts
        value = int(self.data[row, word_idx])
        return (value >> bit_idx) & 1

    def get(self, row: int, col: int) -> int:
        """Get element at (row, col). Alias for get_bit for compatibility."""
        return self.get_bit(row, col)

    def set(self, row: int, col: int, value: int):
        """Set element at (row, col). Only supports setting to 1 (compatibility)."""
        if value:
            self.set_bit(row, col)

    def get_row_bitwise(self, row_idx: int) -> int:
        """Get row as a packed Python int (concatenate 64-bit words)."""
        if self.cols <= 64:
            return int(self.data[row_idx, 0])
        else:
            result = 0
            for i, word in enumerate(self.data[row_idx]):
                result |= int(word) << (i * 64)
            return result

    def memory_usage(self) -> SparseStats:
        """Calculate memory usage and basic stats for dense matrix."""
        # nnz = int(np.count_nonzero(self.data)) * 64
        # Approximated nnz from stored words is coarse; compute exact count instead
        if hasattr(np, "bitwise_count"):  # numpy >= 2.0, vectorised popcount
            exact_nnz = int(np.bitwise_count(self.data).sum())
        else:
            exact_nnz = int(sum(int(w).bit_count() for w in self.data.ravel()))
        total_bits = self.rows * self.cols
        density = exact_nnz / total_bits if total_bits > 0 else 0
        memory_bytes = self.data.nbytes
        compression_ratio = (self.rows * self.cols) / memory_bytes if memory_bytes > 0 else 1.0

        return SparseStats(
            nnz=exact_nnz, density=density, memory_bytes=memory_bytes, compression_ratio=compression_ratio
        )


def create_sparse_matrix(
    rows: int,
    cols: int,
    coordinates: list[tuple[int, int]] | None = None,
    density: float | None = None,
    format_hint: str = "auto",
) -> SparseGF2Matrix:
    """
    Factory function to create optimized sparse GF(2) matrix.

    Args:
        rows: Number of rows
        cols: Number of columns
        coordinates: List of (row, col) positions to set to 1
        density: If given, create random matrix with this density
        format_hint: Storage format preference

    Returns:
        Optimized sparse matrix
    """
    if coordinates is not None:
        row_indices = [coord[0] for coord in coordinates]
        col_indices = [coord[1] for coord in coordinates]
        return SparseGF2Matrix(rows, cols, (row_indices, col_indices), format_hint)

    elif density is not None:
        # Create random sparse matrix
        import random

        total_elements = rows * cols
        num_ones = int(total_elements * density)

        coordinates = random.sample([(i, j) for i in range(rows) for j in range(cols)], num_ones)

        row_indices = [coord[0] for coord in coordinates]
        col_indices = [coord[1] for coord in coordinates]
        return SparseGF2Matrix(rows, cols, (row_indices, col_indices), format_hint)

    else:
        # Create empty matrix
        return SparseGF2Matrix(rows, cols, format_hint=format_hint)
