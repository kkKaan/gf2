"""Quick benchmark to verify optimization improvements."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gf2
from gf2.sparse import SparseGF2Matrix


def pack_vector(vec):
    out = 0
    for i, bit in enumerate(vec):
        out |= (bit & 1) << i
    return out


def unpack_vector(x, n):
    return [(x >> i) & 1 for i in range(n)]


def gaussian_elimination_GF2_bitwise(rows, n):
    A = rows[:]
    pivot_cols = []
    r = 0
    for col in range(n):
        pivot_row = None
        for i in range(r, len(A)):
            if (A[i] >> col) & 1:
                pivot_row = i
                break
        if pivot_row is None:
            continue
        A[r], A[pivot_row] = A[pivot_row], A[r]
        pivot_cols.append(col)
        for i in range(r + 1, len(A)):
            if (A[i] >> col) & 1:
                A[i] ^= A[r]
        r += 1
        if r == len(A):
            break
    return A, pivot_cols


def nullspace_solution_bitwise(rows, pivot_cols, n):
    all_cols = set(range(n))
    free_cols = sorted(all_cols - set(pivot_cols))
    if not free_cols:
        raise ValueError("No free variable found")
    x = [0] * n
    x[free_cols[0]] = 1
    num_pivots = len(pivot_cols)
    for i in reversed(range(num_pivots)):
        p = pivot_cols[i]
        sum_free = 0
        for j in range(p + 1, n):
            if (rows[i] >> j) & 1:
                sum_free ^= x[j]
        x[p] = sum_free
    sol_int = pack_vector(x)
    if sol_int == 0:
        sol_int = 1 << free_cols[0]
    return sol_int


def simon_bitwise(matrix):
    """Original simon_bitwise implementation."""
    n = len(matrix[0])
    rows = [pack_vector(row) for row in matrix]
    start_time = time.time()
    A_echelon, pivot_cols = gaussian_elimination_GF2_bitwise(rows, n)
    sol_int = nullspace_solution_bitwise(A_echelon, pivot_cols, n)
    elapsed_time = time.time() - start_time
    sol_bits = unpack_vector(sol_int, n)
    sol_str = "".join(str(b) for b in sol_bits)
    return sol_str, elapsed_time


def generate_test_matrix(n, density=0.5):
    rows = n - 1
    matrix = []
    for _ in range(rows):
        row = [1 if np.random.random() < density else 0 for _ in range(n)]
        matrix.append(row)
    return matrix


# Test configurations
test_configs = [
    (50, 0.5),
    (100, 0.5),
    (200, 0.5),
]

np.random.seed(42)

print("=" * 80)
print("QUICK PERFORMANCE BENCHMARK - OPTIMIZED vs ORIGINAL")
print("=" * 80)
print()

for n, density in test_configs:
    print(f"\n--- Testing n={n}, density={density:.0%} ---")

    # Generate test matrix
    matrix = generate_test_matrix(n, density=density)

    # Test 1: Original simon_bitwise
    simon_times = []
    for _ in range(10):
        _, t = simon_bitwise(matrix)
        simon_times.append(t * 1000)

    # Test 2: gf2 nullspace_fast (zero-overhead)
    gf2_fast_times = []
    for _ in range(10):
        _, t = gf2.nullspace_fast(matrix)
        gf2_fast_times.append(t * 1000)

    # Test 3: gf2 nullspace_bitwise (with SparseGF2Matrix wrapper)
    gf2_bitwise_times = []
    for _ in range(10):
        sparse_matrix = SparseGF2Matrix(len(matrix), n, matrix)
        _, t = gf2.nullspace_bitwise(sparse_matrix)
        gf2_bitwise_times.append(t * 1000)

    # Results
    simon_avg = np.mean(simon_times)
    gf2_fast_avg = np.mean(gf2_fast_times)
    gf2_bitwise_avg = np.mean(gf2_bitwise_times)

    print(f"  simon_bitwise:              {simon_avg:>8.4f} ms")
    print(f"  gf2.nullspace_fast:       {gf2_fast_avg:>8.4f} ms", end="")

    if gf2_fast_avg < simon_avg:
        speedup = simon_avg / gf2_fast_avg
        print(f"  ({speedup:.2f}x FASTER) ✓✓✓")
    elif gf2_fast_avg < simon_avg * 1.1:
        slowdown = gf2_fast_avg / simon_avg
        print(f"  ({slowdown:.2f}x slower, ~same)")
    else:
        slowdown = gf2_fast_avg / simon_avg
        print(f"  ({slowdown:.2f}x slower)")

    print(f"  gf2.nullspace_bitwise:    {gf2_bitwise_avg:>8.4f} ms", end="")
    slowdown = gf2_bitwise_avg / simon_avg
    if slowdown < 1.2:
        print(f"  ({slowdown:.2f}x slower, acceptable)")
    else:
        print(f"  ({slowdown:.2f}x slower)")

    # Verify correctness
    sol1, _ = simon_bitwise(matrix)
    sol2, _ = gf2.nullspace_fast(matrix)
    sparse_matrix = SparseGF2Matrix(len(matrix), n, matrix)
    sol3, _ = gf2.nullspace_bitwise(sparse_matrix)

    # Check orthogonality for all solutions
    def check_orthogonality(matrix, sol_str):
        sol_bits = [int(b) for b in sol_str]
        for vector in matrix:
            dot = sum(sol_bits[j] & vector[j] for j in range(len(sol_bits))) % 2
            if dot != 0:
                return False
        return True

    all_correct = (
        check_orthogonality(matrix, sol1)
        and check_orthogonality(matrix, sol2)
        and check_orthogonality(matrix, sol3)
    )

    if all_correct:
        print("  ✓ All solutions correct (orthogonality verified)")
    else:
        print("  ✗ ERROR: Some solutions incorrect!")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("If gf2.nullspace_fast is within ~1.1x of simon_bitwise, optimization succeeded!")
print("Target: Match or beat simon_bitwise performance")
