# binpy

High-performance binary (GF(2)) matrix operations library for Python.

[![CI](https://github.com/kkkaan/binpy/workflows/CI/badge.svg)](https://github.com/kkkaan/binpy/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **High-performance**: Bitwise-optimized operations using efficient sparse/dense storage
- **Complete linear algebra**: rank, solve, nullspace, inverse, decompositions  
- **Specialized generators**: LDPC, circulant, Toeplitz, Vandermonde, Hamming codes
- **Memory efficient**: Multiple sparse formats with automatic optimization
- **Production ready**: Full test suite, type hints, modern packaging

## Installation

```bash
# Install from source (development)
git clone https://github.com/kkkaan/binpy.git
cd binpy
pip install -e ".[dev,test]"
```

## Quick Start

```python
import binpy

# Create matrices
A = binpy.identity(5)                    # 5x5 identity matrix
B = binpy.random_sparse(5, 5, density=0.3)  # Random sparse matrix
C = binpy.zeros(3, 4)                    # 3x4 zero matrix

# Basic operations (all in GF(2))
sum_matrix = binpy.add(A, B)             # XOR addition
product = binpy.multiply(A, B)           # Binary matrix multiplication  
A_transpose = binpy.transpose(A)         # Matrix transpose

# Linear algebra
r = binpy.rank(A)                        # Matrix rank
det_A = binpy.det(A)                     # Determinant (0 or 1)
is_inv = binpy.is_invertible(A)          # Invertibility check

# Solve linear systems Ax = b over GF(2)
b = [1, 0, 1, 0, 1]
x = binpy.solve(A, b)                    # Exact solution
null_space = binpy.nullspace(A)          # Null space basis

# Matrix generators for coding theory
H = binpy.hamming_matrix(3)              # Hamming code parity check
ldpc = binpy.ldpc_matrix(100, 200, row_weight=3)  # LDPC code
circ = binpy.circulant([1, 0, 1, 1])     # Circulant matrix
```

## Advanced Usage

### Custom Sparse Matrices

```python
# Create from coordinates
coords = [(0, 1), (1, 2), (2, 0)]  # (row, col) positions  
matrix = binpy.create_sparse_matrix(3, 3, coordinates=coords)

# Different storage formats are automatically chosen
dense_like = binpy.random_sparse(10, 10, density=0.8)   # Uses bit-packed storage
very_sparse = binpy.random_sparse(1000, 1000, density=0.01)  # Uses CSR

# Access internal representation  
print(matrix.memory_usage())  # Shows compression statistics
```

### Coding Theory Applications

```python
# Generate LDPC codes
H = binpy.ldpc_matrix(m=500, n=1000, row_weight=6, method="progressive") 

# Classical codes
hamming_H = binpy.hamming_matrix(r=4)    # [15,11,3] Hamming code
bch_H = binpy.bch_matrix(n=15, k=7, t=2)  # BCH code (simplified)

# Structured matrices
toeplitz_A = binpy.toeplitz([1, 0, 1], [1, 1, 0, 1])
vandermonde_V = binpy.vandermonde([1, 2, 3, 4], n=6)
```

## Performance

Binpy is optimized for speed through:
- **Bitwise operations**: Native CPU bit manipulation  
- **Smart storage**: Automatic format selection (CSR, bit-packed, etc.)
- **Memory efficiency**: Compression ratios of 8x+ for sparse matrices
- **Algorithmic optimization**: Specialized GF(2) algorithms

## Development

```bash
# Clone and setup
git clone https://github.com/kkkaan/binpy.git
cd binpy
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev,test]"

# Run tests
pytest tests/

# Code quality
ruff check .          # Linting
ruff format .         # Formatting  
mypy binpy/           # Type checking

# Install pre-commit hooks
pre-commit install
```
