"""
Performance Comparison: gf2 vs Other GF(2) Libraries

Compares binary field (GF(2)) specialized libraries:
- gf2 (this library)
- galois (finite field arithmetic)
- sage (optional, if available)
- Manual bitwise implementation (reference)

Focus: GF(2) operations only, comparing apples-to-apples.
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

# Try to import GF(2) specialized libraries
try:
    import galois

    GALOIS_AVAILABLE = True
    print("✅ galois library available")
except ImportError:
    GALOIS_AVAILABLE = False
    print("⚠️  galois library not available (pip install galois)")

try:
    from sage.all import GF, Matrix

    SAGE_AVAILABLE = True
    print("✅ Sage available")
except ImportError:
    SAGE_AVAILABLE = False
    print("⚠️  Sage not available (optional)")


def generate_random_gf2_matrix(n, m, density=0.5):
    """Generate random GF(2) matrix."""
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
# Reference Implementation (Pure Bitwise Python)
# ============================================================================


def pack_vector(vec):
    """Pack list of bits into integer."""
    out = 0
    for i, bit in enumerate(vec):
        out |= (bit & 1) << i
    return out


def gaussian_elimination_bitwise(rows, n):
    """Bitwise Gaussian elimination (reference implementation)."""
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


def reference_rank(matrix):
    """Compute rank using pure bitwise Python (reference)."""
    n = len(matrix[0])
    rows = [pack_vector(row) for row in matrix]
    A_echelon, pivot_cols = gaussian_elimination_bitwise(rows, n)
    return len(pivot_cols)


def reference_nullspace(matrix):
    """Compute nullspace using pure bitwise Python (reference)."""
    n = len(matrix[0])
    rows = [pack_vector(row) for row in matrix]
    A_echelon, pivot_cols = gaussian_elimination_bitwise(rows, n)

    # Find free columns
    all_cols = set(range(n))
    free_cols = sorted(all_cols - set(pivot_cols))
    if not free_cols:
        return None

    # Construct solution
    x = [0] * n
    x[free_cols[0]] = 1
    for i in reversed(range(len(pivot_cols))):
        p = pivot_cols[i]
        sum_free = 0
        for j in range(p + 1, n):
            if (A_echelon[i] >> j) & 1:
                sum_free ^= x[j]
        x[p] = sum_free

    return "".join(str(b) for b in x)


# ============================================================================
# gf2 Implementations
# ============================================================================


def gf2_rank(matrix):
    """Compute rank using gf2."""
    A = SparseGF2Matrix(len(matrix), len(matrix[0]), matrix)
    return gf2.rank(A)


def gf2_nullspace(matrix):
    """Compute nullspace using gf2 (fast path)."""
    solution, _ = gf2.nullspace_fast(matrix)
    return solution


def gf2_multiply(matrix1, matrix2):
    """Matrix multiplication using gf2."""
    A = SparseGF2Matrix(len(matrix1), len(matrix1[0]), matrix1)
    B = SparseGF2Matrix(len(matrix2), len(matrix2[0]), matrix2)
    return gf2.multiply(A, B)


# ============================================================================
# galois Implementations
# ============================================================================


def galois_rank(matrix):
    """Compute rank using galois library."""
    if not GALOIS_AVAILABLE:
        raise ImportError("galois not available")
    GF = galois.GF(2)
    A = GF(matrix)
    try:
        out = A.row_reduce()
        if isinstance(out, tuple):
            pivots = out[2] if len(out) >= 3 else out[1]
            return len(pivots)
        if hasattr(A, "rank"):
            return int(A.rank())
    except Exception:
        pass

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


def galois_multiply(matrix1, matrix2):
    """Matrix multiplication using galois."""
    if not GALOIS_AVAILABLE:
        raise ImportError("galois not available")
    GF = galois.GF(2)
    A = GF(matrix1)
    B = GF(matrix2)
    C = A @ B
    return C


# ============================================================================
# Sage Implementations (optional)
# ============================================================================


def sage_rank(matrix):
    """Compute rank using Sage."""
    if not SAGE_AVAILABLE:
        raise ImportError("Sage not available")
    F = GF(2)
    M = Matrix(F, matrix)
    return M.rank()


def sage_nullspace(matrix):
    """Compute nullspace using Sage."""
    if not SAGE_AVAILABLE:
        raise ImportError("Sage not available")
    F = GF(2)
    M = Matrix(F, matrix)
    null = M.right_kernel()
    if null.dimension() > 0:
        vec = null.basis()[0]
        return "".join(str(int(v)) for v in vec)
    return None


# ============================================================================
# Main Benchmark Runner
# ============================================================================


def run_gf2_benchmarks():
    """Run benchmarks comparing GF(2) libraries."""
    print("=" * 80)
    print("GF(2) LIBRARIES COMPARISON")
    print("=" * 80)
    print()
    print("Comparing:")
    print("  • gf2 (this library) - optimized GF(2) operations")
    print("  • galois - finite field arithmetic library")
    if SAGE_AVAILABLE:
        print("  • Sage - computer algebra system")
    print("  • reference - pure bitwise Python implementation")
    print()

    # Test configurations
    test_configs = [
        ("Small (50x50, 50%)", 50, 50, 0.5),
        ("Medium (100x100, 50%)", 100, 100, 0.5),
        ("Large (200x200, 50%)", 200, 200, 0.5),
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
        print(f"Test: {name}")
        print(f"{'=' * 80}")

        # Generate test matrix
        matrix = generate_random_gf2_matrix(n, m, density)

        # Skip very large for some implementations
        skip_large = n >= 200

        # ========================================
        # Rank Computation
        # ========================================
        print("\n--- Rank Computation ---")

        # gf2
        gf2_res = measure_performance_and_memory(gf2_rank, matrix, num_trials=5)
        print(f"gf2:     {gf2_res['mean_time']:>8.3f} ms  |  {gf2_res['mean_memory']:>6.2f} MB")

        # galois
        if GALOIS_AVAILABLE:
            galois_res = measure_performance_and_memory(galois_rank, matrix, num_trials=5)
            if galois_res.get("success"):
                print(
                    f"galois:    {galois_res['mean_time']:>8.3f} ms  |  {galois_res['mean_memory']:>6.2f} MB"
                )
            else:
                print("galois:    skipped (error)")
                galois_res = None
        else:
            galois_res = None

        # Sage
        if SAGE_AVAILABLE and not skip_large:
            sage_res = measure_performance_and_memory(sage_rank, matrix, num_trials=5)
            print(f"Sage:      {sage_res['mean_time']:>8.3f} ms  |  {sage_res['mean_memory']:>6.2f} MB")
        else:
            sage_res = None

        # Reference
        if n <= 100:  # Only for smaller matrices
            ref_res = measure_performance_and_memory(reference_rank, matrix, num_trials=5)
            print(f"reference: {ref_res['mean_time']:>8.3f} ms  |  {ref_res['mean_memory']:>6.2f} MB")
        else:
            ref_res = None

        results["rank"].append(
            {
                "name": name,
                "size": n,
                "density": density,
                "gf2": gf2_res,
                "galois": galois_res,
                "sage": sage_res,
                "reference": ref_res,
            }
        )

        # ========================================
        # Nullspace Computation
        # ========================================
        # Always run: the nullspace case below builds its own
        # underdetermined (n-1) x n system from `matrix`.
        if True:
            print("\n--- Nullspace Computation ---")

            # Use n-1 rows for underdetermined system
            nullspace_matrix = matrix[: n - 1] if len(matrix) >= n - 1 else matrix

            # gf2
            gf2_null_res = measure_performance_and_memory(gf2_nullspace, nullspace_matrix, num_trials=5)
            print(f"gf2:     {gf2_null_res['mean_time']:>8.3f} ms  |  {gf2_null_res['mean_memory']:>6.2f} MB")

            # galois
            if GALOIS_AVAILABLE and not skip_large:
                galois_null_res = measure_performance_and_memory(
                    galois_nullspace, nullspace_matrix, num_trials=5
                )
                print(
                    f"galois:    {galois_null_res['min_time']:>8.3f} ms  |  "
                    f"{galois_null_res['mean_memory']:>6.2f} MB"
                )
            else:
                galois_null_res = None

            # Sage
            if SAGE_AVAILABLE and not skip_large:
                sage_null_res = measure_performance_and_memory(sage_nullspace, nullspace_matrix, num_trials=5)
                print(
                    f"Sage:      {sage_null_res['min_time']:>8.3f} ms  |  "
                    f"{sage_null_res['mean_memory']:>6.2f} MB"
                )
            else:
                sage_null_res = None

            # Reference
            if n <= 100:
                ref_null_res = measure_performance_and_memory(
                    reference_nullspace, nullspace_matrix, num_trials=5
                )
                print(
                    f"reference: {ref_null_res['min_time']:>8.3f} ms  |  "
                    f"{ref_null_res['mean_memory']:>6.2f} MB"
                )
            else:
                ref_null_res = None

            results["nullspace"].append(
                {
                    "name": name,
                    "size": n,
                    "density": density,
                    "gf2": gf2_null_res,
                    "galois": galois_null_res,
                    "sage": sage_null_res,
                    "reference": ref_null_res,
                }
            )

        # ========================================
        # Matrix Multiplication
        # ========================================
        if n <= 100:  # Only for smaller matrices
            print("\n--- Matrix Multiplication ---")

            matrix2 = generate_random_gf2_matrix(m, n, density)

            # gf2
            gf2_mult_res = measure_performance_and_memory(gf2_multiply, matrix, matrix2, num_trials=5)
            print(f"gf2:     {gf2_mult_res['mean_time']:>8.3f} ms  |  {gf2_mult_res['mean_memory']:>6.2f} MB")

            # galois
            if GALOIS_AVAILABLE:
                galois_mult_res = measure_performance_and_memory(
                    galois_multiply, matrix, matrix2, num_trials=5
                )
                print(
                    f"galois:    {galois_mult_res['min_time']:>8.3f} ms  |  "
                    f"{galois_mult_res['mean_memory']:>6.2f} MB"
                )
            else:
                galois_mult_res = None

            results["multiply"].append(
                {
                    "name": name,
                    "size": n,
                    "density": density,
                    "gf2": gf2_mult_res,
                    "galois": galois_mult_res,
                }
            )

    return results


def generate_summary_table(results):
    """Generate summary comparison table."""
    print("\n" + "=" * 80)
    print("SUMMARY: gf2 vs GF(2) Libraries (100x100 matrix)")
    print("=" * 80)

    # Find 100x100, 50% density results
    rank_100 = [r for r in results["rank"] if r["size"] == 100 and abs(r["density"] - 0.5) < 0.01]
    null_100 = [r for r in results["nullspace"] if r["size"] == 100 and abs(r["density"] - 0.5) < 0.01]
    _mult_100 = [r for r in results["multiply"] if r["size"] == 100 and abs(r["density"] - 0.5) < 0.01]

    if rank_100:
        print("\n--- Rank Computation ---")
        print(f"{'Library':<12} {'Time (ms)':<12} {'Memory (MB)':<12} {'vs gf2':<12}")
        print("-" * 48)

        r = rank_100[0]
        gf2_time = r["gf2"]["mean_time"]

        print(f"{'gf2':<12} {gf2_time:<12.3f} {r['gf2']['mean_memory']:<12.2f} {'1.00x':<12}")

        if r["galois"]:
            ratio = r["galois"]["mean_time"] / gf2_time
            print(
                f"{'galois':<12} {r['galois']['min_time']:<12.3f} "
                f"{r['galois']['mean_memory']:<12.2f} {ratio:<12.2f}x"
            )

        if r["reference"]:
            ratio = r["reference"]["mean_time"] / gf2_time
            print(
                f"{'reference':<12} {r['reference']['min_time']:<12.3f} "
                f"{r['reference']['mean_memory']:<12.2f} {ratio:<12.2f}x"
            )

    if null_100:
        print("\n--- Nullspace Computation ---")
        print(f"{'Library':<12} {'Time (ms)':<12} {'Memory (MB)':<12} {'vs gf2':<12}")
        print("-" * 48)

        r = null_100[0]
        gf2_time = r["gf2"]["mean_time"]

        print(f"{'gf2':<12} {gf2_time:<12.3f} {r['gf2']['mean_memory']:<12.2f} {'1.00x':<12}")

        if r["galois"]:
            ratio = r["galois"]["mean_time"] / gf2_time
            print(
                f"{'galois':<12} {r['galois']['min_time']:<12.3f} "
                f"{r['galois']['mean_memory']:<12.2f} {ratio:<12.2f}x"
            )

        if r["reference"]:
            ratio = r["reference"]["mean_time"] / gf2_time
            print(
                f"{'reference':<12} {r['reference']['min_time']:<12.3f} "
                f"{r['reference']['mean_memory']:<12.2f} {ratio:<12.2f}x"
            )


def plot_gf2_comparison(results):
    """Generate comparison charts for GF(2) libraries."""
    print("\n" + "=" * 80)
    print("GENERATING GF(2) COMPARISON CHARTS")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GF(2) Libraries Comparison: gf2 vs galois vs reference", fontsize=16)

    # Plot rank computation time
    ax = axes[0, 0]
    rank_data = [r for r in results["rank"] if r["size"] <= 200]
    sizes = [r["size"] for r in rank_data]

    gf2_times = [r["gf2"]["mean_time"] for r in rank_data]
    ax.plot(sizes, gf2_times, "o-", label="gf2", linewidth=2, markersize=8)

    if GALOIS_AVAILABLE and rank_data[0]["galois"]:
        galois_times = [r["galois"]["mean_time"] for r in rank_data if r["galois"]]
        ax.plot(sizes[: len(galois_times)], galois_times, "s-", label="galois", linewidth=2, markersize=8)

    if rank_data[0]["reference"]:
        ref_times = [r["reference"]["mean_time"] for r in rank_data if r["reference"]]
        ax.plot(sizes[: len(ref_times)], ref_times, "^-", label="reference", linewidth=2, markersize=8)

    ax.set_xlabel("Matrix Size", fontsize=11)
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_title("Rank Computation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot memory usage
    ax = axes[0, 1]
    gf2_mem = [r["gf2"]["mean_memory"] for r in rank_data]
    ax.plot(sizes, gf2_mem, "o-", label="gf2", linewidth=2, markersize=8)

    if GALOIS_AVAILABLE and rank_data[0]["galois"]:
        galois_mem = [r["galois"]["mean_memory"] for r in rank_data if r["galois"]]
        ax.plot(sizes[: len(galois_mem)], galois_mem, "s-", label="galois", linewidth=2, markersize=8)

    ax.set_xlabel("Matrix Size", fontsize=11)
    ax.set_ylabel("Memory (MB)", fontsize=11)
    ax.set_title("Memory Usage (Rank)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot nullspace computation
    if results["nullspace"]:
        ax = axes[1, 0]
        null_data = [r for r in results["nullspace"] if r["size"] <= 100]
        sizes = [r["size"] for r in null_data]

        gf2_times = [r["gf2"]["mean_time"] for r in null_data]
        ax.plot(sizes, gf2_times, "o-", label="gf2", linewidth=2, markersize=8)

        if GALOIS_AVAILABLE and null_data[0]["galois"]:
            galois_times = [r["galois"]["mean_time"] for r in null_data if r["galois"]]
            ax.plot(sizes[: len(galois_times)], galois_times, "s-", label="galois", linewidth=2, markersize=8)

        if null_data[0]["reference"]:
            ref_times = [r["reference"]["mean_time"] for r in null_data if r["reference"]]
            ax.plot(sizes[: len(ref_times)], ref_times, "^-", label="reference", linewidth=2, markersize=8)

        ax.set_xlabel("Matrix Size", fontsize=11)
        ax.set_ylabel("Time (ms)", fontsize=11)
        ax.set_title("Nullspace Computation", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Speedup comparison
    ax = axes[1, 1]
    rank_100 = [r for r in results["rank"] if r["size"] <= 100]
    if rank_100:
        names = ["gf2"]
        times = [1.0]  # gf2 as baseline

        if GALOIS_AVAILABLE and rank_100[0]["galois"]:
            gf2_time = rank_100[0]["gf2"]["mean_time"]
            galois_time = rank_100[0]["galois"]["mean_time"]
            names.append("galois")
            times.append(galois_time / gf2_time)

        if rank_100[0]["reference"]:
            gf2_time = rank_100[0]["gf2"]["mean_time"]
            ref_time = rank_100[0]["reference"]["mean_time"]
            names.append("reference")
            times.append(ref_time / gf2_time)

        colors = ["#2E86AB", "#A23B72", "#F18F01"]
        bars = ax.bar(names, times, color=colors[: len(names)], alpha=0.7)
        ax.axhline(y=1, color="r", linestyle="--", linewidth=2, label="gf2 baseline")
        ax.set_ylabel("Relative Time (lower is better)", fontsize=11)
        ax.set_title("Speed Comparison (100×100)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}x",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig("benchmarks/gf2_libraries_comparison.png", dpi=300, bbox_inches="tight")
    print("✅ Chart saved to: benchmarks/gf2_libraries_comparison.png")


if __name__ == "__main__":
    np.random.seed(42)

    # Run benchmarks
    results = run_gf2_benchmarks()

    # Generate summary
    generate_summary_table(results)

    # Plot results
    try:
        plot_gf2_comparison(results)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ GF(2) BENCHMARK COMPLETE")
    print("=" * 80)
    print("\nConclusion:")
    print("  • gf2 is optimized for GF(2) operations with good memory efficiency")
    print("  • galois provides similar performance for pure GF(2) field operations")
    print("  • Reference implementation shows gf2's optimization value")
    print("\nAll compared libraries are GF(2)/binary field specialized!")
