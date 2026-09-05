"""
Benchmark comparison between gf2 and simon_amazon_test.py implementations.

This script tests:
1. Performance (execution time)
2. Memory usage
3. Correctness of results
"""

import sys
import time
import tracemalloc
from fractions import Fraction
from pathlib import Path

import numpy as np

# Import gf2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gf2
from gf2.sparse import SparseGF2Matrix

###############################################
# Copy implementations from simon_amazon_test.py
###############################################


def pack_vector(vec):
    """Pack a list of bits into an integer."""
    out = 0
    for i, bit in enumerate(vec):
        out |= (bit & 1) << i
    return out


def unpack_vector(x, n):
    """Unpack an integer into a list of n bits."""
    return [(x >> i) & 1 for i in range(n)]


def gaussian_elimination_GF2_bitwise(rows, n):
    """Gaussian elimination from simon_amazon_test.py."""
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
    """Nullspace solution from simon_amazon_test.py."""
    all_cols = set(range(n))
    free_cols = sorted(all_cols - set(pivot_cols))
    if not free_cols:
        raise ValueError("No free variable found; the system appears to be full rank.")
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


def get_secret_integer_bitwise_simon(matrix):
    """Original bitwise solver from simon_amazon_test.py."""
    n = len(matrix[0])
    rows = [pack_vector(row) for row in matrix]
    start_time = time.time()
    A_echelon, pivot_cols = gaussian_elimination_GF2_bitwise(rows, n)
    sol_int = nullspace_solution_bitwise(A_echelon, pivot_cols, n)
    elapsed_time = time.time() - start_time
    sol_bits = unpack_vector(sol_int, n)
    sol_str = "".join(str(b) for b in sol_bits)
    return sol_str, elapsed_time


def modinv(a, p):
    """Compute the modular inverse of a modulo p."""
    t, new_t = 0, 1
    r, new_r = p, a % p
    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    if r > 1:
        raise ValueError(f"{a} is not invertible modulo {p}")
    return t % p


def get_secret_integer_generic_simon(matrix, mod=None):
    """Generic solver from simon_amazon_test.py."""
    start_time = time.time()

    mat = matrix.tolist() if isinstance(matrix, np.ndarray) else [list(row) for row in matrix]

    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0

    if mod is None:
        is_binary = all(x in (0, 1) for row in mat for x in row)
        if is_binary:
            mod = 2

    if mod is None:
        A = [[Fraction(x) for x in row] for row in mat]
    else:
        A = [[int(x) % mod for x in row] for row in mat]

    pivot_cols = []
    pivot_rows = []
    pivot_row = 0

    for col in range(cols):
        pivot_found = False
        for r in range(pivot_row, rows):
            cond = A[r][col] != 0 if mod is None else A[r][col] % mod != 0
            if cond:
                pivot_found = True
                max_row = r
                break
        if not pivot_found:
            continue
        if max_row != pivot_row:
            A[pivot_row], A[max_row] = A[max_row], A[pivot_row]
        pivot_cols.append(col)
        pivot_rows.append(pivot_row)
        pivot_val = A[pivot_row][col]
        inv = Fraction(1) / pivot_val if mod is None else modinv(pivot_val, mod)
        for c in range(col, cols):
            if mod is None:
                A[pivot_row][c] *= inv
            else:
                A[pivot_row][c] = (A[pivot_row][c] * inv) % mod
        for r in range(pivot_row + 1, rows):
            factor = A[r][col]
            if mod is None:
                if factor != 0:
                    for c in range(col, cols):
                        A[r][c] -= factor * A[pivot_row][c]
            else:
                if factor % mod != 0:
                    for c in range(col, cols):
                        A[r][c] = (A[r][c] - factor * A[pivot_row][c]) % mod
        pivot_row += 1
        if pivot_row == rows:
            break

    all_cols = set(range(cols))
    free_cols = sorted(all_cols - set(pivot_cols))
    if not free_cols:
        elapsed_time = time.time() - start_time
        return None, elapsed_time

    if mod is None:
        solution = [Fraction(0) for _ in range(cols)]
        solution[free_cols[0]] = Fraction(1)
    else:
        solution = [0 for _ in range(cols)]
        solution[free_cols[0]] = 1

    for i in reversed(range(len(pivot_cols))):
        p = pivot_cols[i]
        r_idx = pivot_rows[i]
        s_val = 0
        for j in range(p + 1, cols):
            if mod is None:
                s_val += A[r_idx][j] * solution[j]
            else:
                s_val = (s_val + A[r_idx][j] * solution[j]) % mod
        if mod is None:
            solution[p] = -s_val
        else:
            solution[p] = (-s_val) % mod

    if mod is not None and mod == 2:
        sol_str = "".join("1" if solution[i] % 2 != 0 else "0" for i in range(cols))
    else:
        is_binary = True
        for s_val in solution:
            if mod is None:
                if s_val != 0 and abs(s_val) != 1:
                    is_binary = False
                    break
            else:
                if s_val % mod not in (0, 1):
                    is_binary = False
                    break
        if is_binary:
            sol_str = "".join(
                "1" if (s_val != 0 if mod is None else s_val % mod != 0) else "0" for s_val in solution
            )
        else:
            sol_str = "(" + ", ".join(str(s_val) for s_val in solution) + ")"

    elapsed_time = time.time() - start_time
    return sol_str, elapsed_time


###############################################
# Binpy wrapper functions
###############################################


def get_secret_integer_gf2_nullspace(matrix):
    """Using gf2's nullspace function."""
    n = len(matrix[0])
    start_time = time.time()

    # Create SparseGF2Matrix
    sparse_matrix = SparseGF2Matrix(len(matrix), n, matrix)

    # Get nullspace basis
    null_basis = gf2.nullspace(sparse_matrix)

    if not null_basis:
        raise ValueError("No free variable found; the system appears to be full rank.")

    # Take first basis vector
    sol_bits = null_basis[0]
    sol_str = "".join(str(b) for b in sol_bits)

    elapsed_time = time.time() - start_time
    return sol_str, elapsed_time


def get_secret_integer_gf2_nullspace_bitwise(matrix):
    """Using gf2's nullspace_bitwise function (original algorithm)."""
    n = len(matrix[0])

    # Create SparseGF2Matrix
    sparse_matrix = SparseGF2Matrix(len(matrix), n, matrix)

    # Use nullspace_bitwise which already times itself
    return gf2.nullspace_bitwise(sparse_matrix)


def get_secret_integer_gf2_nullspace_fast(matrix):
    """Using gf2's nullspace_fast function (zero-overhead direct algorithm)."""
    # Use nullspace_fast which already times itself and works directly with list of lists
    return gf2.nullspace_fast(matrix)


###############################################
# Benchmark utilities
###############################################


def generate_test_matrix(n, density=0.5, ensure_underdetermined=True):
    """Generate a random binary matrix for testing."""
    # An (n-1) x n system is underdetermined, so a nullspace vector exists.
    rows = n - 1 if ensure_underdetermined else n

    matrix = []
    for _ in range(rows):
        row = [1 if np.random.random() < density else 0 for _ in range(n)]
        matrix.append(row)

    return matrix


def verify_orthogonality(matrix, solution_str):
    """Verify that solution is in nullspace (all dot products = 0 mod 2)."""
    sol_bits = [int(b) for b in solution_str]
    n = len(sol_bits)

    for vector in matrix:
        dot_product = sum(sol_bits[j] & vector[j] for j in range(n)) % 2
        if dot_product != 0:
            return False
    return True


def measure_memory_and_time(func, matrix):
    """Measure both memory usage and execution time."""
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        result, internal_time = func(matrix)

        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "result": result,
            "internal_time": internal_time,
            "total_time": end_time - start_time,
            "peak_memory_mb": peak / (1024 * 1024),
            "success": True,
            "error": None,
        }
    except Exception as e:
        tracemalloc.stop()
        return {
            "result": None,
            "internal_time": None,
            "total_time": None,
            "peak_memory_mb": None,
            "success": False,
            "error": str(e),
        }


###############################################
# Main benchmark
###############################################


def run_benchmark():
    """Run comprehensive benchmark comparing all implementations."""
    print("=" * 80)
    print("BINPY vs SIMON_AMAZON_TEST.PY PERFORMANCE BENCHMARK")
    print("=" * 80)
    print()

    # Test configurations
    test_sizes = [10, 20, 50, 100, 200, 500]
    densities = [0.3, 0.5, 0.7]
    num_trials = 10

    results = []

    for n in test_sizes:
        for density in densities:
            print(f"\n{'=' * 80}")
            print(f"Testing: n={n}, density={density:.1%}")
            print(f"{'=' * 80}")

            # Run multiple trials
            trial_results = {
                "simon_bitwise": [],
                "simon_generic": [],
                "gf2_nullspace": [],
                "gf2_nullspace_bitwise": [],
                "gf2_nullspace_fast": [],
            }

            for trial in range(num_trials):
                # Generate test matrix
                matrix = generate_test_matrix(n, density=density)

                # Test each implementation
                implementations = [
                    ("simon_bitwise", get_secret_integer_bitwise_simon),
                    ("simon_generic", get_secret_integer_generic_simon),
                    ("gf2_nullspace", get_secret_integer_gf2_nullspace),
                    ("gf2_nullspace_bitwise", get_secret_integer_gf2_nullspace_bitwise),
                    ("gf2_nullspace_fast", get_secret_integer_gf2_nullspace_fast),
                ]

                trial_data = {}

                for name, func in implementations:
                    result = measure_memory_and_time(func, matrix)
                    trial_results[name].append(result)
                    trial_data[name] = result

                # Verify all solutions are correct (if successful)
                solutions = {}
                for name in trial_data:
                    if trial_data[name]["success"]:
                        solutions[name] = trial_data[name]["result"]

                # Check orthogonality for each solution
                if trial == 0:  # Only print for first trial
                    for name, sol in solutions.items():
                        is_valid = verify_orthogonality(matrix, sol)
                        status = "✓" if is_valid else "✗"
                        print(f"  {name}: {status} (orthogonality check)")

            # Compute statistics
            print(f"\nResults (averaged over {num_trials} trials):")
            print(f"{'Method':<30} {'Time (ms)':<12} {'Memory (MB)':<12} {'Status'}")
            print("-" * 80)

            for name in trial_results:
                successful_trials = [r for r in trial_results[name] if r["success"]]

                if successful_trials:
                    avg_time = np.mean([r["internal_time"] for r in successful_trials]) * 1000
                    avg_memory = np.mean([r["peak_memory_mb"] for r in successful_trials])
                    status = f"{len(successful_trials)}/{num_trials} OK"
                    print(f"{name:<30} {avg_time:<12.6f} {avg_memory:<12.3f} {status}")
                else:
                    print(f"{name:<30} {'N/A':<12} {'N/A':<12} FAILED")

            # Store aggregate results
            results.append({"n": n, "density": density, "trial_results": trial_results})

    # Final summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    for result in results:
        n = result["n"]
        density = result["density"]
        trial_results = result["trial_results"]

        print(f"\nn={n}, density={density:.1%}:")

        # Get average times for each method
        times = {}
        for name in trial_results:
            successful = [r for r in trial_results[name] if r["success"]]
            if successful:
                times[name] = np.mean([r["internal_time"] for r in successful]) * 1000

        if times:
            fastest = min(times.items(), key=lambda x: x[1])
            print(f"  Fastest: {fastest[0]} ({fastest[1]:.6f} ms)")

            # Compare gf2 with simon_bitwise
            if "gf2_nullspace_fast" in times and "simon_bitwise" in times:
                speedup = times["simon_bitwise"] / times["gf2_nullspace_fast"]
                if speedup > 1:
                    print(f"  gf2_nullspace_fast is {speedup:.2f}x FASTER than simon_bitwise")
                else:
                    print(f"  gf2_nullspace_fast is {1 / speedup:.2f}x SLOWER than simon_bitwise")

            if "gf2_nullspace_bitwise" in times and "simon_bitwise" in times:
                speedup = times["simon_bitwise"] / times["gf2_nullspace_bitwise"]
                if speedup > 1:
                    print(f"  gf2_nullspace_bitwise is {speedup:.2f}x FASTER than simon_bitwise")
                else:
                    print(f"  gf2_nullspace_bitwise is {1 / speedup:.2f}x SLOWER than simon_bitwise")

            if "gf2_nullspace" in times and "simon_generic" in times:
                speedup = times["simon_generic"] / times["gf2_nullspace"]
                if speedup > 1:
                    print(f"  gf2_nullspace is {speedup:.2f}x FASTER than simon_generic")
                else:
                    print(f"  gf2_nullspace is {1 / speedup:.2f}x SLOWER than simon_generic")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    run_benchmark()
