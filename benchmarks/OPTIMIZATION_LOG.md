# Optimization log

Before/after for the changes made during the performance review. Both columns
are the same machine, same inputs, same harness rules (warm-up, minimum of N,
no profiler attached). "before" is the package at commit `4648955`.

Matrices are square, 50% density, and **the packed-row cache is pre-warmed on
both sides**, so these figures isolate the algorithmic change from the
cache-miss fix described at the bottom.

| operation | before ms | after ms | speedup |
|---|---:|---:|---:|
| construct 128 | 0.694 | 0.648 | 1.1x |
| rank 128 | 0.485 | 0.373 | 1.3x |
| multiply 128 | 2.890 | 0.335 | **8.6x** |
| transpose 128 | 2.554 | 0.858 | 3.0x |
| nullspace_fast 128 | 1.608 | 0.999 | 1.6x |
| nullspace basis 128 | 1.245 | 0.760 | 1.6x |
| solve 128 | 1.231 | 0.810 | 1.5x |
| inverse 128 | 1.045 | 0.786 | 1.3x |
| to_dense 128 | 0.708 | 0.127 | 5.6x |
| rank 256 | 1.997 | 1.487 | 1.3x |
| multiply 256 | 12.066 | 0.756 | **16.0x** |
| transpose 256 | 10.649 | 3.508 | 3.0x |
| nullspace_fast 256 | 6.382 | 3.916 | 1.6x |
| solve 256 | 4.745 | 3.055 | 1.6x |
| to_dense 256 | 2.787 | 0.397 | 7.0x |
| rank 512 | 9.179 | 5.011 | 1.8x |
| multiply 512 | 48.486 | 2.114 | **22.9x** |
| transpose 512 | 44.186 | 16.853 | 2.6x |
| nullspace_fast 512 | 28.019 | 15.841 | 1.8x |
| nullspace basis 512 | 26.033 | 13.889 | 1.9x |
| solve 512 | 21.950 | 14.060 | 1.6x |
| to_dense 512 | 11.701 | 1.498 | 7.8x |
| nullspace basis 256, rank 128 | 122.294 | 2.481 | **49.3x** |
| random_sparse 1000x1000 d=0.01 | 49.304 | 4.362 | 11.3x |
| ones 2000x2000 | 662.736 | 1.305 | **507.7x** |
| ldpc progressive 250x500 | 4256.481 | 85.871 | **49.6x** |
| minimal_polynomial n=16 | 38280.019 | 0.181 | **211249x** |

## What changed, and why each one mattered

**`multiply` - Method of Four Russians (8.6-22.9x).** The old routine
materialised all of B's columns as integers, then took `A.rows * B.cols`
separate parity dot products. Row *i* of `A @ B` is just the XOR of the B-rows
selected by row *i* of A, and with an 8-bit block table one set of 256
pre-combined B-rows serves every row of A. Sparse A takes a plain row-XOR path
instead, since its cost is proportional to `nnz(A)`.

**`nullspace` basis extraction (49.3x on a rank-deficient matrix).** In RREF a
pivot variable's value with one free variable set is literally one stored bit,
`R[i, free_col]`. The old code recovered that bit by scanning every column to
the right of the pivot, making basis extraction `O(free * pivots * cols)`
instead of `O(free * pivots)`. Invisible on a full-rank matrix with one free
column, catastrophic at rank n/2.

**Elimination probes (1.3-1.8x, everywhere).** `(A[i] >> col) & 1` allocates a
shifted copy of the entire row to look at one bit. `A[i] & bit` with a mask
that is shifted once per column does not. Also: `det` was doing full Gauss-
Jordan when forward elimination answers the question, and `solve` and
`solve_multiple_rhs` re-derived values already sitting in the augmented column.

**Vectorised elimination past n = 384 (1.8x at 512).** `numpy.flatnonzero`
picks out every row needing the pivot XOR in one call. Below n = 384 the
per-column NumPy overhead loses to the big-integer loop; the crossover was
measured, not guessed.

**Bit iteration (3.0x on transpose, 5.6-7.8x on to_dense).** Loops of the form
`while row: ... row >>= 1` cost one full-width pass per *column*. Walking set
bits with `row & -row` costs one pass per *non-zero*, and unpacking a whole
matrix is one `np.unpackbits` call.

**Generators.** `ones` and `random_sparse` built one Python tuple per cell
before setting a single bit. `_ldpc_progressive_edge_growth` kept edges in a
flat list and re-scanned all of it inside its innermost loop to read a degree.

**`minimal_polynomial`.** Was enumerating all `2^d` coefficient vectors per
degree - exponential. Now finds the first linear dependence among
`I, A, A^2, ...` by elimination, `O(n^4 / 64)` word operations.

## The cache miss, measured separately

`_to_csr` and `_coo_to_csr` never populated `_packed_rows_cache`, so a CSR
matrix rebuilt each row from its column indices on *every* `get_row_bitwise`
call. A full row sweep of a 300x300 matrix:

| access path | per sweep |
|---|---:|
| CSR, no cache | 3.330 ms |
| bit-packed, cached | 0.014 ms |

**238x.** Because storage format is chosen by a `density < 0.5` threshold, a
random 50%-density matrix landed on either side of it run to run, so the same
benchmark could be 30x slower or faster depending on the coin flip. Algorithms
now read rows through one `_rows_of()` helper that consults the cache once.

Stacking that on the multiply rewrite: a freshly built, uncached 256x256 CSR
matrix went from **584 ms to 0.76 ms** end to end.
