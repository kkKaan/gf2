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
        # Count non-zeros and analyze structure
        self.nnz = sum(sum(row) for row in matrix)
        density = self.nnz / (self.rows * self.cols) if self.rows * self.cols > 0 else 0

        # Select optimal format
        if format_hint == "auto":
            if density < 0.01:  # Very sparse
                self.format = "csr_compact"
            elif density < 0.5:  # Moderately sparse
                self.format = "csr"
            else:  # Dense
                self.format = "bitpacked"
        else:
            self.format = format_hint

        # Convert to selected format
        if self.format == "csr" or self.format == "csr_compact":
            self._to_csr(matrix)
        elif self.format == "bitpacked":
            self._to_bitpacked(matrix)

    def _from_coordinates(self, rows: list[int], cols: list[int], format_hint: str):
        """Convert from coordinate (COO) format."""
        self.nnz = len(rows)
        density = self.nnz / (self.rows * self.cols)

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

        # Use compact indices for very sparse matrices
        if self.format == "csr_compact" and self.cols <= 65535:
            # Use 16-bit indices instead of 32-bit
            self.csr_col_ind = np.array(self.csr_col_ind, dtype=np.uint16)
        else:
            self.csr_col_ind = np.array(self.csr_col_ind, dtype=np.uint32)

        self.csr_row_ptr = np.array(self.csr_row_ptr, dtype=np.uint32)

    def _to_bitpacked(self, matrix: list[list[int]]):
        """Convert to bit-packed dense format."""
        # Pack each row into integers (64 bits per integer)
        words_per_row = (self.cols + 63) // 64
        self.bitpacked_rows = np.zeros((self.rows, words_per_row), dtype=np.uint64)

        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val & 1:
                    word_idx = j // 64
                    bit_idx = j % 64
                    self.bitpacked_rows[i, word_idx] |= 1 << bit_idx

    def _coo_to_csr(self, row_indices: list[int], col_indices: list[int]):
        """Convert coordinate format to CSR."""
        # Sort by row index
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
            self.bitpacked_rows[row, word_idx] |= 1 << bit_idx

    def get_row_bitwise(self, row_idx: int) -> int:
        """
        Get row as packed integer for bitwise operations.
        Optimized for your existing bitwise algorithms.
        """
        if self.format == "bitpacked":
            # Already bit-packed, just need to combine words
            if self.bitpacked_rows is None:
                raise ValueError("Bitpacked rows not initialized")
            if self.cols <= 64:
                return int(self.bitpacked_rows[row_idx, 0])
            else:
                # Combine multiple 64-bit words into a Python int
                result = 0
                for i, word in enumerate(self.bitpacked_rows[row_idx]):
                    result |= int(word) << (i * 64)
                return result

        elif self.format in ["csr", "csr_compact"]:
            # Convert CSR row to packed integer
            if self.csr_row_ptr is None or self.csr_col_ind is None:
                raise ValueError("CSR format data not initialized")
            result = 0
            start = self.csr_row_ptr[row_idx]
            end = self.csr_row_ptr[row_idx + 1]

            for col_idx in self.csr_col_ind[start:end]:
                result |= 1 << col_idx

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
        # Analyze sparsity to select format
        total_bits = len(packed_rows) * self.cols
        set_bits = sum(bin(row).count("1") for row in packed_rows)
        density = set_bits / total_bits if total_bits > 0 else 0

        self.nnz = set_bits

        if density < 0.1:
            self.format = "csr_compact" if self.cols <= 65535 else "csr"
            self._packed_to_csr(packed_rows)
        else:
            self.format = "bitpacked"
            self._packed_to_bitpacked(packed_rows)

    def _packed_to_csr(self, packed_rows: list[int]):
        """Convert packed rows to CSR format."""
        self.csr_row_ptr = [0]
        self.csr_col_ind = []

        for row_val in packed_rows:
            len(self.csr_col_ind)

            # Extract set bits
            col = 0
            while row_val > 0:
                if row_val & 1:
                    self.csr_col_ind.append(col)
                row_val >>= 1
                col += 1

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

        for i, row_val in enumerate(packed_rows):
            # Ensure row_val is a Python int
            row_val = int(row_val) if hasattr(row_val, "__int__") else row_val

            for word_idx in range(words_per_row):
                # Extract 64-bit word at the correct offset
                shift = word_idx * 64
                word_bits = (row_val >> shift) & 0xFFFFFFFFFFFFFFFF  # 64-bit mask
                self.bitpacked_rows[i, word_idx] = word_bits

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
        # For simplicity, convert to dense format, modify, and convert back
        # This is not efficient but works for testing
        dense_matrix = self.to_dense()
        dense_matrix[row][col] = 1
        self._from_dense(dense_matrix, self.format)

    def get(self, row: int, col: int) -> int:
        """Get element at (row, col). Alias for get_bit for compatibility."""
        return self.get_bit(row, col)

    def set(self, row: int, col: int, value: int):
        """Set element at (row, col). Only supports setting to 1 (use for compatibility)."""
        if value:
            self.set_bit(row, col)

    def get_bit(self, row: int, col: int) -> int:
        """Get bit at (row, col)."""
        if self.format == "bitpacked":
            if self.bitpacked_rows is None:
                raise ValueError("Bitpacked rows not initialized")
            word_idx = col // 64
            bit_idx = col % 64
            return int((self.bitpacked_rows[row, word_idx] >> bit_idx) & 1)
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
        result = [[0] * self.cols for _ in range(self.rows)]

        for i in range(self.rows):
            row_packed = self.get_row_bitwise(i)
            for j in range(self.cols):
                if (row_packed >> j) & 1:
                    result[i][j] = 1

        return result

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
    Uses 1 bit per element instead of 8 bits (byte) or 64 bits (int).
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
        """Load from dense matrix."""
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val & 1:
                    self.set_bit(i, j)

    def set_bit(self, row: int, col: int):
        """Set bit at (row, col) to 1."""
        word_idx = col // 64
        bit_idx = col % 64
        self.data[row, word_idx] |= 1 << bit_idx

    def get_bit(self, row: int, col: int) -> int:
        """Get bit at (row, col)."""
        word_idx = col // 64
        bit_idx = col % 64
        return int((self.data[row, word_idx] >> bit_idx) & 1)

    def get(self, row: int, col: int) -> int:
        """Get element at (row, col). Alias for get_bit for compatibility."""
        return self.get_bit(row, col)

    def set(self, row: int, col: int, value: int):
        """Set element at (row, col). Only supports setting to 1 (use for compatibility)."""
        if value:
            self.set_bit(row, col)

    def get_row_bitwise(self, row_idx: int) -> int:
        """Get row as packed integer."""
        if self.cols <= 64:
            return int(self.data[row_idx, 0])
        else:
            # Combine multiple words
            result = 0
            for i, word in enumerate(self.data[row_idx]):
                result |= word << (i * 64)
            return result

    def memory_usage(self) -> SparseStats:
        """Calculate memory usage."""
        return SparseStats(
            nnz=np.sum(self.data != 0),  # Approximate
            density=0.5,  # Assume average case
            memory_bytes=self.data.nbytes,
            compression_ratio=(self.rows * self.cols) / self.data.nbytes,
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
