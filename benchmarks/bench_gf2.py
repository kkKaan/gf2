"""
gf2 vs NumPy vs galois over GF(2) - honest, like-for-like comparison.

Every contender receives the SAME input data, is given its own setup step
(timed separately), and has its output checked for equivalence before any
speed claim is made. See benchmarks/harness.py for the measurement rules.

Two NumPy baselines are reported on purpose:

  numpy-packed   bit-packed uint64 rows with vectorised XOR elimination.
                 This is what a competent NumPy user writes for GF(2) and is
                 the baseline any speed claim must be made against.
  numpy-naive    element-wise uint8 indexing. Kept only to show how large the
                 gap between the two NumPy styles is; beating it proves
                 nothing.

Run:  python benchmarks/bench_gf2.py [--json out.json] [--quick]
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf2
from benchmarks.harness import Contender, format_table, run_operation
from gf2.sparse import SparseGF2Matrix

try:
    import galois

    GALOIS = True
except ImportError:
    GALOIS = False


# --------------------------------------------------------------------------
# shared input generation
# --------------------------------------------------------------------------
def make_matrix(rows: int, cols: int, density: float, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[1 if rng.random() < density else 0 for _ in range(cols)] for _ in range(rows)]


def to_packed_u64(matrix: list[list[int]]) -> np.ndarray:
    """Rows as little-endian uint64 words - the NumPy-native GF(2) layout."""
    a = np.array(matrix, dtype=np.uint8)
    rows, cols = a.shape
    words = (cols + 63) // 64
    padded = np.zeros((rows, words * 64), dtype=np.uint8)
    padded[:, :cols] = a
    # packbits is MSB-first per byte; flip within each byte to get LSB-first bit j.
    packed = np.packbits(padded.reshape(rows, words * 8, 8)[:, :, ::-1].reshape(rows, -1), axis=1)
    return packed.view(np.uint64)


# --------------------------------------------------------------------------
# NumPy GF(2) reference implementations
# --------------------------------------------------------------------------
def np_packed_rank(packed: np.ndarray, cols: int) -> int:
    A = packed.copy()
    n_rows = A.shape[0]
    r = 0
    for col in range(cols):
        if r >= n_rows:
            break
        w = col >> 6
        mask = np.uint64(1) << np.uint64(col & 63)
        hits = np.flatnonzero(A[r:, w] & mask)
        if hits.size == 0:
            continue
        p = r + int(hits[0])
        if p != r:
            A[[r, p]] = A[[p, r]]
        below = r + 1 + np.flatnonzero(A[r + 1 :, w] & mask)
        if below.size:
            A[below] ^= A[r]
        r += 1
    return r


def np_packed_nullspace_vector(packed: np.ndarray, cols: int) -> str | None:
    """One nullspace vector, echelon + back-substitution - matches gf2's contract."""
    A = packed.copy()
    n_rows = A.shape[0]
    pivot_cols: list[int] = []
    r = 0
    for col in range(cols):
        if r >= n_rows:
            break
        w = col >> 6
        mask = np.uint64(1) << np.uint64(col & 63)
        hits = np.flatnonzero(A[r:, w] & mask)
        if hits.size == 0:
            continue
        p = r + int(hits[0])
        if p != r:
            A[[r, p]] = A[[p, r]]
        below = r + 1 + np.flatnonzero(A[r + 1 :, w] & mask)
        if below.size:
            A[below] ^= A[r]
        pivot_cols.append(col)
        r += 1

    free = sorted(set(range(cols)) - set(pivot_cols))
    if not free:
        return None
    x = np.zeros(cols, dtype=np.uint8)
    x[free[0]] = 1
    for i in range(len(pivot_cols) - 1, -1, -1):
        pc = pivot_cols[i]
        row = A[i]
        acc = 0
        for j in range(pc + 1, cols):
            if (int(row[j >> 6]) >> (j & 63)) & 1:
                acc ^= int(x[j])
        x[pc] = acc
    return "".join(str(int(b)) for b in x)


def np_naive_rank(matrix: list[list[int]]) -> int:
    """Element-wise uint8 elimination - the naive NumPy style, kept for contrast."""
    A = np.array(matrix, dtype=np.uint8)
    rows, cols = A.shape
    r = 0
    for col in range(cols):
        pivot = None
        for i in range(r, rows):
            if A[i, col] & 1:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        for i in range(r + 1, rows):
            if A[i, col] & 1:
                A[i, :] ^= A[r, :]
        r += 1
    return int(r)


def np_multiply(pair: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """GF(2) matmul. uint8 accumulation wraps mod 256, which preserves mod 2."""
    a, b = pair
    return (a @ b) & 1


# --------------------------------------------------------------------------
# equivalence normalisers
# --------------------------------------------------------------------------
def check_nullvector(matrix: list[list[int]]):
    """Return a normaliser that verifies A @ x == 0 and x != 0 over GF(2)."""

    def _norm(result):
        if result is None:
            return ("none",)
        bits = result if isinstance(result, str) else "".join(str(int(v)) for v in result)
        x = [int(c) for c in bits]
        if not any(x):
            return ("zero-vector",)
        for row in matrix:
            if sum(r & v for r, v in zip(row, x, strict=True)) & 1:
                return ("NOT-IN-NULLSPACE",)
        return ("valid-nullvector", len(x))

    return _norm


def norm_dense(result) -> tuple:
    if isinstance(result, np.ndarray):
        return tuple(map(tuple, (result & 1).tolist()))
    if hasattr(result, "to_dense"):
        return tuple(map(tuple, result.to_dense()))
    return tuple(map(tuple, result))


# --------------------------------------------------------------------------
# suites
# --------------------------------------------------------------------------
def rank_suite(matrix, cols, reps):
    contenders = [
        Contender(
            "gf2",
            setup=lambda m: SparseGF2Matrix(len(m), len(m[0]), m),
            run=gf2.rank,
            normalize=int,
            note="big-int elim; vectorised past n=384",
        ),
        Contender(
            "numpy-packed",
            setup=to_packed_u64,
            run=lambda p: np_packed_rank(p, cols),
            normalize=int,
            note="uint64 rows, vectorised XOR",
        ),
        Contender(
            "numpy-naive",
            setup=lambda m: np.array(m, dtype=np.uint8),
            run=lambda a: np_naive_rank(a.tolist()),
            normalize=int,
            note="element-wise (strawman)",
        ),
        Contender(
            "galois",
            setup=lambda m: galois.GF(2)(m) if GALOIS else None,
            run=lambda a: int(np.linalg.matrix_rank(a)),
            normalize=int,
            available=GALOIS,
            note="galois.GF(2)",
        ),
    ]
    return run_operation(f"rank  ({len(matrix)}x{cols})", matrix, contenders, reps=reps, reference="gf2")


def nullspace_suite(matrix, cols, reps):
    norm = check_nullvector(matrix)
    contenders = [
        Contender(
            "gf2 nullspace_fast",
            setup=lambda m: m,
            run=lambda m: gf2.nullspace_fast(m)[0],
            normalize=norm,
            note="one vector",
        ),
        Contender(
            "numpy-packed",
            setup=to_packed_u64,
            run=lambda p: np_packed_nullspace_vector(p, cols),
            normalize=norm,
            note="one vector",
        ),
        Contender(
            "galois null_space[0]",
            setup=lambda m: galois.GF(2)(m) if GALOIS else None,
            run=lambda a: "".join(str(int(b)) for b in a.null_space()[0]),
            normalize=norm,
            available=GALOIS,
            note="computes FULL basis, then takes row 0",
        ),
    ]
    return run_operation(
        f"nullspace vector  ({len(matrix)}x{cols})",
        matrix,
        contenders,
        reps=reps,
        reference="gf2 nullspace_fast",
    )


def multiply_suite(m1, m2, reps):
    contenders = [
        Contender(
            "gf2",
            setup=lambda p: (
                SparseGF2Matrix(len(p[0]), len(p[0][0]), p[0]),
                SparseGF2Matrix(len(p[1]), len(p[1][0]), p[1]),
            ),
            run=lambda ab: gf2.multiply(*ab),
            normalize=norm_dense,
        ),
        Contender(
            "numpy uint8 matmul",
            setup=lambda p: (np.array(p[0], dtype=np.uint8), np.array(p[1], dtype=np.uint8)),
            run=np_multiply,
            normalize=norm_dense,
            note="BLAS-backed, wraps mod 256",
        ),
        Contender(
            "galois",
            setup=lambda p: (galois.GF(2)(p[0]), galois.GF(2)(p[1])) if GALOIS else None,
            run=lambda ab: np.asarray(ab[0] @ ab[1]).astype(np.uint8),
            normalize=norm_dense,
            available=GALOIS,
        ),
    ]
    n = len(m1)
    label = f"multiply  ({n}x{len(m1[0])} @ {len(m2)}x{len(m2[0])})"
    return run_operation(label, (m1, m2), contenders, reps=reps, reference="gf2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=Path("benchmarks/results.json"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()

    sizes = (
        [(64, 0.5), (128, 0.5), (256, 0.5)]
        if args.quick
        else [(64, 0.5), (128, 0.5), (256, 0.5), (512, 0.5), (1024, 0.5), (128, 0.05), (128, 0.95)]
    )

    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "galois": galois.__version__ if GALOIS else None,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "gf2": gf2.__version__,
    }
    print("environment:", json.dumps(env, indent=2))
    print(
        "\nNOTE: timings are taken with no profiler attached; peak-memory is a\n"
        "separate pass. '=?' verifies the contender agreed with the reference.\n"
    )

    report = {"environment": env, "operations": []}

    for n, density in sizes:
        m = make_matrix(n, n, density, seed=1234 + n)
        for op in (
            rank_suite(m, n, args.reps),
            # (n-1) x n is underdetermined, so a nullspace vector always exists
            nullspace_suite(m[: n - 1], n, args.reps),
        ):
            report["operations"].append(op | {"size": n, "density": density})
            print(format_table(op))

        # Multiply is O(n^3)-ish for every contender; 1024 is the largest size
        # that still finishes quickly, and it is where the blocked kernel's
        # asymptotics start to show against galois.
        if n <= 1024:
            m2 = make_matrix(n, n, density, seed=9876 + n)
            op = multiply_suite(m, m2, args.reps)
            report["operations"].append(op | {"size": n, "density": density})
            print(format_table(op))

    args.json.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
