# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-05

First public release.

### Renamed

- The project was developed as `binpy`. That name belongs to an unrelated
  digital-logic package on PyPI, so the import root is `gf2`. There was no
  public `binpy` release, so no upgrade path is needed: use `import gf2`.

### Added

- `SparseGF2Matrix` / `DenseGF2Matrix` with automatic CSR / bit-packed storage
  selection, and `create_sparse_matrix`.
- Core arithmetic: `add`, `multiply`, `transpose`, `rank`, `det`, `trace`,
  `is_invertible`, `reduced_row_echelon_form`, `lu_decomposition`,
  `matrix_power`, `characteristic_polynomial`, `minimal_polynomial`.
- Solvers: `solve`, `solve_multiple_rhs`, `nullspace`, `nullspace_bitwise`,
  `nullspace_fast`, `inverse`, `kernel`, `image`, `rank_nullity_theorem`,
  `iterative_refinement`.
- Generators: `identity`, `zeros`, `ones`, `random_sparse`, `random_regular`,
  `circulant`, `circulant_random`, `toeplitz`, `ldpc_matrix`,
  `hamming_matrix`, `repetition_matrix`.
- Quantum code constructions: `hypergraph_product` (Tillich-Zemor),
  `surface_code_matrix` (planar, built as a hypergraph product of repetition
  codes), `css_code_matrix`, `bicycle_codes`.
- `py.typed`: the package is annotated throughout and ships its types.
- Structural equality on both matrix types: `A == B` compares shape and
  contents, independently of storage format, and a sparse matrix compares equal
  to a dense one holding the same bits. Previously `==` fell through to
  identity and returned False for equal matrices. Defining `__eq__` makes the
  types unhashable, which is correct: they are mutable through `set_bit`.
- Operations returning several values return NamedTuples -- `LUDecomposition`,
  `RowEchelonForm`, `NullspaceVector`, `RankNullity` -- so a caller can write
  `result.pivot_columns` instead of `result[1]`. They unpack positionally
  exactly like the plain tuples they replace.
- `lu_decomposition` returns `(L, U, perm)` with `A[perm] == L @ U`, plus
  `LUDecomposition.permutation_matrix()` for the equivalent `P @ A == L @ U`
  form.
- A benchmark suite (`benchmarks/`) with a documented measurement methodology,
  and a report generated from recorded measurements rather than written by hand.

### Performance

Relative to the pre-release development tree, same inputs and warm caches:

- `multiply` 22.9x faster at n=512 (Method of Four Russians over packed uint64;
  row-XOR accumulation for sparse operands).
- `nullspace` basis extraction 49.3x faster on a rank-deficient 256x256.
- `transpose` 2.6x, `to_dense` 7.8x, `rank` 1.8x at n=512.
- `ldpc_matrix(method="progressive")` 49.6x faster; `ones` 508x.
- `minimal_polynomial` moved from O(2^n) enumeration to O(n^4 / 64) word
  operations: n=16 went from 38 s to 0.18 ms.
- CSR row access no longer rebuilds each row per call; a full row sweep of a
  300x300 matrix went from 3.33 ms to 0.014 ms.

See `benchmarks/OPTIMIZATION_LOG.md`.

### Fixed

Defects found during pre-release review, all of which failed silently:

- `set_bit` discarded the write on an empty-format matrix and left the
  packed-row cache stale on the CSR path, so later reads saw the pre-write
  matrix.
- `hypergraph_product` returned an all-zero `H_z`, so the CSS commutation
  condition held only vacuously.
- `surface_code_matrix` declared more stabiliser rows than it populated and did
  not satisfy `H_x @ H_z.T == 0`; it now produces a genuine planar surface code
  with k = 1.
- `characteristic_polynomial` returned a stub that was wrong for every n >= 3.
- `nullspace_fast` accepted `include_packing_time` and ignored it.
- `iterative_refinement` mutated a NumPy right-hand side in place.
- Coordinate input now de-duplicates, bounds-checks, and handles zero
  dimensions; packed rows are masked to the column count.
- `random_regular` could emit under-weight rows; seeded generators no longer
  reseed the global `random` module.
- `lu_decomposition` swapped rows of U without recording the permutation or
  applying it to L, so its output satisfied neither `A == L @ U` nor
  `P @ A == L @ U`. It reconstructed A in roughly half of random cases and gave
  the caller no way to recover the difference. It now returns the permutation
  as a third element and reconstructs exactly, verified over 500 random
  matrices. The existing test was named `..._reconstruction` but only asserted
  the triangular shapes, which is how this survived.
- The packed representation assumed the host was little-endian: it paired
  `int.to_bytes(..., "little")` with native-order NumPy arrays. On a big-endian
  host (NumPy ships s390x wheels) a single set bit in word 0 read back as
  2**56, and `unpackbits` returned the wrong byte's bits - wrong answers rather
  than an error. The storage dtype is now pinned to `"<u8"`, which is the
  native dtype on little-endian hosts and therefore costs nothing there.

### Removed

Cut before the first release rather than published under a name that promises
something else. None of these had users; all three were exported and would have
been a breaking change to withdraw later.

- `bch_matrix` was not a BCH parity check matrix. It applied a fixed arithmetic
  mask (`(i+1)*(j+1) % 3 == 1`) with no coding-theoretic meaning and ignored
  its error-correction parameter `t` entirely.
- `vandermonde` reduced integer powers mod 2. Since the parity of a power
  depends only on the parity of the base, every row came out all-ones or
  `[1, 0, 0, ...]`, giving a matrix of rank at most 2 — not a Vandermonde
  matrix over GF(2^m).
- `color_code_matrix` did not satisfy `H_x @ H_z.T == 0`, so the pair it
  returned was not a quantum code. Use `surface_code_matrix` or
  `hypergraph_product`, which are exact by construction.

### Known limitations

- Dense GF(2) multiplication is competitive with `galois` only from n = 1024;
  below that `galois` is faster. Use it or `m4ri` if dense multiplication
  dominates your workload.
- No extension-field GF(2^m) arithmetic. `vandermonde` and `bch_matrix` were
  removed before release rather than shipped under names they did not earn
  (see Removed); real versions need GF(2^m) and are deferred.
- Pure Python plus NumPy: no compiled kernel.

[Unreleased]: https://github.com/kkKaan/gf2/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kkKaan/gf2/releases/tag/v0.1.0
