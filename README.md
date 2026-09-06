# gf2

High-performance binary (GF(2)) matrix operations library for Python.

[![CI](https://github.com/kkKaan/gf2/workflows/CI/badge.svg)](https://github.com/kkKaan/gf2/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
# Install from source (development)
git clone https://github.com/kkKaan/gf2.git
cd gf2
pip install -e ".[dev,test]"
```

## Quick Start

```python
import gf2

# Create matrices
A = gf2.identity(5)  # 5x5 identity matrix
B = gf2.random_sparse(5, 5, density=0.3)  # Random sparse matrix
C = gf2.zeros(3, 4)  # 3x4 zero matrix

# Basic operations (all in GF(2))
sum_matrix = gf2.add(A, B)  # XOR addition
product = gf2.multiply(A, B)  # Binary matrix multiplication
A_transpose = gf2.transpose(A)  # Matrix transpose

# Linear algebra
r = gf2.rank(A)  # Matrix rank
det_A = gf2.det(A)  # Determinant (0 or 1)
is_inv = gf2.is_invertible(A)  # Invertibility check

# Solve linear systems Ax = b over GF(2)
b = [1, 0, 1, 0, 1]
x = gf2.solve(A, b)  # Exact solution
null_space = gf2.nullspace(A)  # Null space basis

# Matrix generators for coding theory
H = gf2.hamming_matrix(3)  # Hamming code parity check
ldpc = gf2.ldpc_matrix(100, 200, row_weight=4)  # LDPC code (m*row_weight must divide n)
circ = gf2.circulant([1, 0, 1, 1])  # Circulant matrix
```

## Advanced Usage

### Custom Sparse Matrices

```python
# Create from coordinates
coords = [(0, 1), (1, 2), (2, 0)]  # (row, col) positions
matrix = gf2.create_sparse_matrix(3, 3, coordinates=coords)

# Different storage formats are automatically chosen
dense_like = gf2.random_sparse(10, 10, density=0.8)  # Uses bit-packed storage
very_sparse = gf2.random_sparse(1000, 1000, density=0.01)  # Uses CSR

# Access internal representation
print(matrix.memory_usage())  # Shows compression statistics
```

### Coding Theory Applications

```python
# Generate LDPC codes
H = gf2.ldpc_matrix(m=500, n=1000, row_weight=6, method="progressive")

# Classical codes
hamming_H = gf2.hamming_matrix(r=4)  # [15,11,3] Hamming code
rep_H = gf2.repetition_matrix(5)  # length-5 repetition code

# Quantum codes (exact CSS commutation: H_x @ H_z.T == 0)
H_x, H_z = gf2.surface_code_matrix(distance=3)  # planar surface code, k=1
Q_x, Q_z = gf2.hypergraph_product(hamming_H, rep_H)  # Tillich-Zemor product

# Structured matrices
toeplitz_A = gf2.toeplitz([1, 0, 1], [1, 1, 0, 1])
circ = gf2.circulant([1, 0, 1, 1])
```

## Performance

gf2 stores rows bit-packed and does GF(2) arithmetic with whole-row bitwise
operations, so a row XOR costs one machine word per 64 columns instead of one
Python step per column.

- **Elimination** (rank, nullspace, solve, inverse) runs on Python big
  integers below n = 384 and switches to vectorised NumPy uint64 rows above
  it, because the crossover between the two was measured, not assumed.
- **Multiplication** uses the Method of Four Russians: one lookup table of
  2^8 pre-combined rows of B serves every row of A, so the XOR count drops
  from O(mn/2) to O(mn/8 + n/8 * 256).
- **Storage** picks CSR or bit-packed automatically from the density.

### Measured results

Numbers, methodology, and the exact environment live in
[`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/kkKaan/gf2/blob/main/benchmarks/BENCHMARK_RESULTS.md), which is
**generated from `benchmarks/results.json`** rather than written by hand.

To reproduce:

```bash
python benchmarks/bench_gf2.py      # measure -> benchmarks/results.json
python benchmarks/make_report.py    # results.json -> BENCHMARK_RESULTS.md
```

Summary against the fastest honest rival at each size, square matrices at 50%
density, Python 3.11 / NumPy 2.2.6 / galois 0.4.6 on an arm64 Mac:

| operation | n=128 | n=512 | n=1024 |
|---|---|---|---|
| **rank** | **2.3x faster** than packed-NumPy | par with packed-NumPy | 1.1x slower than packed-NumPy |
| **nullspace vector** | **1.9x faster** than packed-NumPy | **1.5x faster** | **1.4x faster** |
| **multiply** | 5.9x slower than galois | 1.3x slower than galois | par with galois; **85x faster** than NumPy |

Against `galois`, gf2's rank is 12-27x faster and its nullspace 6-13x faster
across this range. Against a *naive* element-wise NumPy loop gf2 looks 9-13x
faster, but that comparison is not meaningful and the benchmark labels it as a
strawman: the baseline that matters is bit-packed uint64 NumPy, which is the
`numpy-packed` row in the report.

### Honest limitations

- **Dense matrix multiply is not gf2's strength.** `galois` is faster below
  n = 1024 and gf2 only draws level there. If dense GF(2) multiplication
  dominates your workload, use `galois` or `m4ri`.
- **Peak memory during multiply is higher than NumPy's**, because the Four
  Russians table and the unpacked selector are transient allocations. Bit
  packing wins on *stored* size, not on scratch space.
- Everything here is pure Python plus NumPy. A C or Cython kernel would move
  the elimination crossover a long way down.

## Development

```bash
# Clone and setup
git clone https://github.com/kkKaan/gf2.git
cd gf2
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev,test]"

# Run tests
pytest tests/

# Code quality
ruff check .          # Linting
ruff format .         # Formatting  
mypy gf2/           # Type checking

# Install pre-commit hooks
pre-commit install
```

## Example use cases

GF(2) linear algebra is the classical half of several quantum and coding-theory
workflows. Each example below is self-contained and runnable.

### Recovering a hidden string from quantum measurements

Simon's algorithm, and Bernstein-Vazirani-style problems generally, leave you
with measurement outcomes `y` that satisfy `y · s = 0` over GF(2). The hidden
string `s` is the null space of those measurements — it is almost never one of
them, so it has to be solved for rather than searched for.

```python
from gf2 import SparseGF2Matrix, nullspace

# Measurement outcomes from a Simon circuit with hidden string s = 11010.
measurements = [
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 1],
    [0, 0, 1, 0, 0],
    [1, 0, 1, 1, 1],
]

A = SparseGF2Matrix(len(measurements), 5, measurements)
for candidate in nullspace(A):
    print("".join(map(str, candidate)))  # -> 11010
```

For a single vector with no wrapper overhead — the hot path when you are
looping over many circuit runs — use `nullspace_fast`, which takes a plain list
of lists:

```python
from gf2 import nullspace_fast

bits, seconds = nullspace_fast(measurements)
print(bits)  # -> 11010
```

### Testing decodability in linear network coding

A receiver can decode once the coding vectors it has collected are linearly
independent **over GF(2)**. Rank over the reals is a different quantity and
disagrees on roughly 9% of random binary matrices, so it cannot stand in here.

```python
from gf2 import SparseGF2Matrix, rank

coding_vectors = [
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],  # equals the XOR of the other two
]

A = SparseGF2Matrix(3, 3, coding_vectors)
print(rank(A))  # 2 -- rank-deficient, not yet decodable
print(rank(A) == len(coding_vectors))  # False
```

### Syndrome decoding for a linear code

```python
from gf2 import hamming_matrix, multiply, SparseGF2Matrix

H = hamming_matrix(3)  # [7,4] Hamming parity check, 3x7
received = [[1], [0], [1], [1], [0], [0], [1]]

syndrome = multiply(H, SparseGF2Matrix(7, 1, received))
print(syndrome.to_dense())  # non-zero syndrome locates the error
```

### Building quantum error-correcting codes

`hypergraph_product` and `surface_code_matrix` satisfy the CSS commutation
condition `H_x @ H_z.T == 0` exactly, by construction:

```python
from gf2 import surface_code_matrix, multiply, transpose, rank

H_x, H_z = surface_code_matrix(distance=5)
n = H_x.cols

commutes = not any(multiply(H_x, transpose(H_z)).get_all_rows_bitwise())
k = n - rank(H_x) - rank(H_z)
print(n, k, commutes)  # 41 1 True
```
