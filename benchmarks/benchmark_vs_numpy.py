"""
Performance and Memory Comparison: gf2 vs NumPy vs other libraries

Compares:
- gf2 (optimized GF(2) library)
- NumPy (standard array operations)
- galois (GF(2) arithmetic library, if available)
- scipy.sparse (sparse matrices)
"""

import sys
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gf2
from gf2.sparse import SparseGF2Matrix

# Try to import optional libraries
try:
    import galois

    GALOIS_AVAILABLE = True
except ImportError:
    GALOIS_AVAILABLE = False
    print("Note: galois library not available, skipping galois benchmarks")

try:
    from scipy import sparse as sp_sparse

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Note: scipy not available, skipping scipy benchmarks")


def generate_random_matrix(n, m, density=0.5):
    """Generate random binary matrix."""
    matrix = []
    for _ in range(n):
        row = [1 if np.random.random() < density else 0 for _ in range(m)]
        matrix.append(row)
    return matrix


def measure_performance_and_memory(func, *args, num_trials=10):
    """Measure execution time and peak memory in SEPARATE passes.

    tracemalloc traces every allocation, which inflated the wall time of this
    codebase's big-integer paths by 14-28x - and by different factors for
    different implementations, so timings taken while it ran were not
    comparable at all. Time is now measured with no tracer attached, memory in
    its own pass, and the headline figure is the minimum rather than the mean
    (for deterministic CPU work the minimum is the sample least contaminated by
    scheduler noise).
    """
    try:
        func(*args)  # warm-up, excluded from the measurement
    except Exception as e:
        print(f"  Error in {getattr(func, '__name__', func)}: {e}")
        return {"success": False}

    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        func(*args)
        times.append((time.perf_counter() - start) * 1000)  # ms

    tracemalloc.start()
    try:
        func(*args)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    peak_mib = peak / (1024 * 1024)

    times.sort()
    return {
        "min_time": times[0],
        "median_time": times[len(times) // 2],
        # mean_time is kept for backward compatibility with existing readers,
        # but min_time is the figure to quote.
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "mean_memory": peak_mib,
        "peak_memory": peak_mib,
        "std_memory": 0.0,
        "success": True,
    }


# ============================================================================
# Rank Computation Benchmarks
# ============================================================================


def gf2_rank(matrix):
    """Compute rank using gf2."""
    A = SparseGF2Matrix(len(matrix), len(matrix[0]), matrix)
    return gf2.rank(A)


def numpy_rank(matrix):
    """Rank over GF(2) with element-wise NumPy indexing.

    Warning:
        This is the NAIVE NumPy style: it indexes single uint8 elements from
        Python, so most of its time goes on creating NumPy scalar objects
        rather than on arithmetic. Beating it says nothing about being fast.
        The baseline that matters is bit-packed uint64 rows with vectorised
        XOR - see ``np_packed_rank`` in benchmarks/bench_gf2.py, which is
        roughly 8x faster than this at n=512.
    """
    A = np.array(matrix, dtype=np.uint8)
    rows, cols = A.shape
    rank = 0

    for col in range(cols):
        # Find pivot row with a 1 in current column
        pivot = None
        for r in range(rank, rows):
            if A[r, col] & 1:
                pivot = r
                break
        if pivot is None:
            continue

        # Swap pivot row into position 'rank'
        if pivot != rank:
            A[[rank, pivot]] = A[[pivot, rank]]

        # Eliminate below pivot (GF(2) XOR)
        for r in range(rank + 1, rows):
            if A[r, col] & 1:
                A[r, :] ^= A[rank, :]

        rank += 1

    return int(rank)


def galois_rank(matrix):
    """Compute rank using galois library."""
    if not GALOIS_AVAILABLE:
        raise ImportError("galois not available")
    GF = galois.GF(2)
    A = GF(matrix)
    # Try multiple APIs across galois versions
    try:
        out = A.row_reduce()
        # galois may return (R, pivots) or (R, _, pivots)
        if isinstance(out, tuple):
            pivots = out[2] if len(out) >= 3 else out[1]
            return len(pivots)
        # Fallback: if only matrix returned, use rank() if available
        if hasattr(A, "rank"):
            return int(A.rank())
    except Exception:
        pass

    # Final fallback: compute rank via NumPy GF(2) reduction
    import numpy as _np

    M = _np.array(matrix, dtype=_np.uint8)
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if M[i, c] & 1:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            M[[r, pivot]] = M[[pivot, r]]
        for i in range(r + 1, rows):
            if M[i, c] & 1:
                M[i, :] ^= M[r, :]
        r += 1
    return int(r)


# ============================================================================
# Nullspace Computation Benchmarks
# ============================================================================


def gf2_nullspace(matrix):
    """Compute nullspace using gf2."""
    return gf2.nullspace_fast(matrix)


def numpy_nullspace(matrix):
    """Compute a nullspace vector over GF(2) using NumPy row-reduction."""
    A = np.array(matrix, dtype=np.uint8)
    rows, cols = A.shape

    pivot_cols = []
    pivot_rows = []
    r = 0

    # Forward elimination to echelon form (GF(2))
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
        pivot_cols.append(col)
        pivot_rows.append(r)

        for i in range(r + 1, rows):
            if A[i, col] & 1:
                A[i, :] ^= A[r, :]

        r += 1
        if r == rows:
            break

    all_cols = set(range(cols))
    free_cols = sorted(all_cols - set(pivot_cols))
    if not free_cols:
        return None

    # Construct one nullspace vector by setting first free variable to 1
    x = np.zeros(cols, dtype=np.uint8)
    free = free_cols[0]
    x[free] = 1

    # Back-substitution to compute pivot variables
    for i in range(len(pivot_cols) - 1, -1, -1):
        pcol = pivot_cols[i]
        prow = pivot_rows[i]
        # sum of A[prow, j] * x[j] for j > pcol, in GF(2)
        s = int(np.bitwise_and(A[prow, pcol + 1 :], x[pcol + 1 :]).sum() & 1) if pcol + 1 < cols else 0
        x[pcol] = s

    return "".join(str(int(b)) for b in x.tolist())


def galois_nullspace(matrix):
    """Compute nullspace using galois library."""
    if not GALOIS_AVAILABLE:
        raise ImportError("galois not available")
    GF = galois.GF(2)
    A = GF(matrix)
    null_space = A.null_space()
    if len(null_space) > 0:
        vec = null_space[0]
        return "".join(str(int(b)) for b in vec)
    return None


# ============================================================================
# Matrix Multiplication Benchmarks
# ============================================================================


def gf2_multiply(matrix1, matrix2):
    """Matrix multiplication using gf2."""
    A = SparseGF2Matrix(len(matrix1), len(matrix1[0]), matrix1)
    B = SparseGF2Matrix(len(matrix2), len(matrix2[0]), matrix2)
    return gf2.multiply(A, B)


def numpy_multiply(matrix1, matrix2):
    """Matrix multiplication using NumPy (mod 2)."""
    A = np.array(matrix1, dtype=int)
    B = np.array(matrix2, dtype=int)
    C = np.dot(A, B) % 2
    return C


def scipy_sparse_multiply(matrix1, matrix2):
    """Matrix multiplication using scipy sparse."""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy not available")
    A = sp_sparse.csr_matrix(np.array(matrix1))
    B = sp_sparse.csr_matrix(np.array(matrix2))
    C = (A @ B).toarray() % 2
    return C


# ============================================================================
# Main Benchmark Runner
# ============================================================================


def run_benchmarks():
    """Run all benchmarks and collect results."""
    print("=" * 80)
    print("PERFORMANCE & MEMORY COMPARISON: gf2 vs NumPy vs Others")
    print("=" * 80)

    # Test configurations
    test_configs = [
        ("Small (50x50)", 50, 50, 0.5),
        ("Medium (100x100)", 100, 100, 0.5),
        ("Large (200x200)", 200, 200, 0.5),
        ("Very Large (500x500)", 500, 500, 0.5),
        ("Sparse (100x100, 10%)", 100, 100, 0.1),
        ("Dense (100x100, 90%)", 100, 100, 0.9),
    ]

    results = {
        "rank": [],
        "nullspace": [],
        "multiply": [],
    }

    for name, n, m, density in test_configs:
        print(f"\n{'=' * 80}")
        print(f"Testing: {name} (density={density:.0%})")
        print(f"{'=' * 80}")

        # Generate test matrix
        matrix = generate_random_matrix(n, m, density)

        # Skip very large matrices for slow methods
        skip_large = n >= 500

        # Rank benchmark
        print("\n--- Rank Computation ---")

        gf2_res = measure_performance_and_memory(gf2_rank, matrix, num_trials=5)
        print(f"gf2:  {gf2_res['mean_time']:.4f} ms, {gf2_res['mean_memory']:.2f} MB")

        numpy_res = measure_performance_and_memory(numpy_rank, matrix, num_trials=5)
        print(f"numpy:  {numpy_res['mean_time']:.4f} ms, {numpy_res['mean_memory']:.2f} MB")

        if GALOIS_AVAILABLE and not skip_large:
            galois_res = measure_performance_and_memory(galois_rank, matrix, num_trials=5)
            if galois_res.get("success"):
                print(f"galois: {galois_res['mean_time']:.4f} ms, {galois_res['mean_memory']:.2f} MB")
            else:
                print("galois: skipped (error)")
                galois_res = None
        else:
            galois_res = None

        results["rank"].append(
            {
                "name": name,
                "size": n,
                "density": density,
                "gf2": gf2_res,
                "numpy": numpy_res,
                "galois": galois_res,
            }
        )

        # Nullspace benchmark (use underdetermined (n-1) x n system)
        print("\n--- Nullspace Computation ---")

        nullspace_matrix = matrix[: max(1, n - 1)]

        gf2_null_res = measure_performance_and_memory(gf2_nullspace, nullspace_matrix, num_trials=5)
        print(f"gf2:  {gf2_null_res['mean_time']:.4f} ms, {gf2_null_res['mean_memory']:.2f} MB")

        if not skip_large:
            numpy_null_res = measure_performance_and_memory(numpy_nullspace, nullspace_matrix, num_trials=5)
            if numpy_null_res.get("success"):
                print(f"numpy:  {numpy_null_res['mean_time']:.4f} ms, {numpy_null_res['mean_memory']:.2f} MB")
            else:
                print("numpy: skipped (error)")
                numpy_null_res = None
        else:
            numpy_null_res = None

        if GALOIS_AVAILABLE and not skip_large:
            galois_null_res = measure_performance_and_memory(galois_nullspace, nullspace_matrix, num_trials=5)
            if galois_null_res.get("success"):
                print(
                    f"galois: {galois_null_res['mean_time']:.4f} ms, {galois_null_res['mean_memory']:.2f} MB"
                )
            else:
                print("galois: skipped (error)")
                galois_null_res = None
        else:
            galois_null_res = None

        results["nullspace"].append(
            {
                "name": name,
                "size": n,
                "density": density,
                "gf2": gf2_null_res,
                "numpy": numpy_null_res,
                "galois": galois_null_res,
            }
        )

        # Matrix multiplication benchmark
        if n <= 200:  # Skip for very large matrices
            print("\n--- Matrix Multiplication ---")

            matrix2 = generate_random_matrix(m, n, density)

            gf2_mult_res = measure_performance_and_memory(gf2_multiply, matrix, matrix2, num_trials=5)
            print(f"gf2:  {gf2_mult_res['mean_time']:.4f} ms, {gf2_mult_res['mean_memory']:.2f} MB")

            numpy_mult_res = measure_performance_and_memory(numpy_multiply, matrix, matrix2, num_trials=5)
            print(f"numpy:  {numpy_mult_res['mean_time']:.4f} ms, {numpy_mult_res['mean_memory']:.2f} MB")

            if SCIPY_AVAILABLE:
                scipy_mult_res = measure_performance_and_memory(
                    scipy_sparse_multiply, matrix, matrix2, num_trials=5
                )
                print(f"scipy:  {scipy_mult_res['mean_time']:.4f} ms, {scipy_mult_res['mean_memory']:.2f} MB")
            else:
                scipy_mult_res = None

            results["multiply"].append(
                {
                    "name": name,
                    "size": n,
                    "density": density,
                    "gf2": gf2_mult_res,
                    "numpy": numpy_mult_res,
                    "scipy": scipy_mult_res,
                }
            )

    return results


def plot_results(results):
    """Generate comparison charts."""
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON CHARTS")
    print("=" * 80)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Performance & Memory Comparison: gf2 vs NumPy vs Others", fontsize=16)

    # Plot rank computation time
    ax = axes[0, 0]
    rank_data = results["rank"]
    sizes = [r["size"] for r in rank_data if r["size"] <= 200]
    gf2_times = [r["gf2"]["mean_time"] for r in rank_data if r["size"] <= 200]
    numpy_times = [r["numpy"]["mean_time"] for r in rank_data if r["size"] <= 200]

    ax.plot(sizes, gf2_times, "o-", label="gf2", linewidth=2)
    ax.plot(sizes, numpy_times, "s-", label="numpy", linewidth=2)
    if GALOIS_AVAILABLE:
        galois_times = [r["galois"]["mean_time"] for r in rank_data if r["galois"] and r["size"] <= 200]
        if galois_times:
            ax.plot(sizes[: len(galois_times)], galois_times, "^-", label="galois", linewidth=2)
    ax.set_xlabel("Matrix Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Rank Computation Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot rank computation memory
    ax = axes[0, 1]
    gf2_mem = [r["gf2"]["mean_memory"] for r in rank_data if r["size"] <= 200]
    numpy_mem = [r["numpy"]["mean_memory"] for r in rank_data if r["size"] <= 200]

    ax.plot(sizes, gf2_mem, "o-", label="gf2", linewidth=2)
    ax.plot(sizes, numpy_mem, "s-", label="numpy", linewidth=2)
    ax.set_xlabel("Matrix Size")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Rank Computation Memory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot nullspace computation time
    if results["nullspace"]:
        ax = axes[0, 2]
        null_data = results["nullspace"]
        sizes = [r["size"] for r in null_data if r["size"] <= 200]
        gf2_times = [r["gf2"]["mean_time"] for r in null_data if r["size"] <= 200]

        ax.plot(sizes, gf2_times, "o-", label="gf2", linewidth=2)
        if null_data[0]["numpy"]:
            numpy_times = [r["numpy"]["mean_time"] for r in null_data if r["numpy"] and r["size"] <= 200]
            ax.plot(sizes[: len(numpy_times)], numpy_times, "s-", label="numpy", linewidth=2)
        ax.set_xlabel("Matrix Size")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Nullspace Computation Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot multiplication time
    if results["multiply"]:
        ax = axes[1, 0]
        mult_data = results["multiply"]
        sizes = [r["size"] for r in mult_data]
        gf2_times = [r["gf2"]["mean_time"] for r in mult_data]
        numpy_times = [r["numpy"]["mean_time"] for r in mult_data]

        ax.plot(sizes, gf2_times, "o-", label="gf2", linewidth=2)
        ax.plot(sizes, numpy_times, "s-", label="numpy", linewidth=2)
        if SCIPY_AVAILABLE and mult_data[0]["scipy"]:
            scipy_times = [r["scipy"]["mean_time"] for r in mult_data if r["scipy"]]
            ax.plot(sizes[: len(scipy_times)], scipy_times, "^-", label="scipy.sparse", linewidth=2)
        ax.set_xlabel("Matrix Size")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Matrix Multiplication Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot speedup comparison
    ax = axes[1, 1]
    sizes = [r["size"] for r in rank_data if r["size"] <= 200]
    speedups = [r["numpy"]["mean_time"] / r["gf2"]["mean_time"] for r in rank_data if r["size"] <= 200]

    ax.bar(range(len(sizes)), speedups, alpha=0.7)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s}x{s}" for s in sizes], rotation=45)
    ax.axhline(y=1, color="r", linestyle="--", label="Equal Performance")
    ax.set_ylabel("Speedup (numpy time / gf2 time)")
    ax.set_title("gf2 Speedup vs NumPy (Rank)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Plot memory efficiency
    ax = axes[1, 2]
    mem_ratios = [
        r["numpy"]["mean_memory"] / r["gf2"]["mean_memory"]
        for r in rank_data
        if r["size"] <= 200 and r["gf2"]["mean_memory"] > 0
    ]

    ax.bar(range(len(sizes)), mem_ratios, alpha=0.7, color="orange")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s}x{s}" for s in sizes], rotation=45)
    ax.axhline(y=1, color="r", linestyle="--", label="Equal Memory")
    ax.set_ylabel("Memory Ratio (numpy / gf2)")
    ax.set_title("Memory Efficiency vs NumPy (Rank)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("benchmarks/comparison_charts.png", dpi=300, bbox_inches="tight")
    print("✅ Charts saved to: benchmarks/comparison_charts.png")

    return fig


def generate_summary_table(results):
    """Generate summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    print("\n--- Rank Computation (100x100 matrix) ---")
    medium_rank = [r for r in results["rank"] if r["size"] == 100 and abs(r["density"] - 0.5) < 0.01]
    if medium_rank:
        r = medium_rank[0]
        print(f"{'Library':<15} {'Time (ms)':<15} {'Memory (MB)':<15} {'Speedup':<15}")
        print("-" * 60)
        gf2_time = r["gf2"]["mean_time"]
        print(f"{'gf2':<15} {gf2_time:<15.4f} {r['gf2']['mean_memory']:<15.2f} {'1.00x':<15}")
        print(
            f"{'numpy':<15} {r['numpy']['min_time']:<15.4f} "
            f"{r['numpy']['mean_memory']:<15.2f} "
            f"{r['numpy']['min_time'] / gf2_time:<15.2f}x"
        )
        if r["galois"]:
            print(
                f"{'galois':<15} {r['galois']['min_time']:<15.4f} "
                f"{r['galois']['mean_memory']:<15.2f} "
                f"{r['galois']['min_time'] / gf2_time:<15.2f}x"
            )


if __name__ == "__main__":
    np.random.seed(42)

    # Run benchmarks
    results = run_benchmarks()

    # Generate summary
    generate_summary_table(results)

    # Plot results
    try:
        plot_results(results)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    print("\n" + "=" * 80)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 80)
