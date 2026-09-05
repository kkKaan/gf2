"""
Detailed memory profiling comparison between gf2 and NumPy.
Shows memory usage over time and peak memory for different operations.
"""

import sys
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gf2
from gf2.sparse import SparseGF2Matrix


def profile_memory_over_time(func, *args):
    """Profile memory usage over time during function execution."""
    tracemalloc.start()
    snapshots = []

    # Take snapshot before
    snapshots.append(tracemalloc.take_snapshot())

    # Execute function
    result = func(*args)

    # Take snapshot after
    snapshots.append(tracemalloc.take_snapshot())

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "result": result,
        "current_mb": current / (1024 * 1024),
        "peak_mb": peak / (1024 * 1024),
        "snapshots": snapshots,
    }


def generate_test_matrix(n, m, density=0.5):
    """Generate random binary matrix."""
    matrix = []
    for _ in range(n):
        row = [1 if np.random.random() < density else 0 for _ in range(m)]
        matrix.append(row)
    return matrix


def compare_memory_profiles():
    """Loop-variable defaults below bind each closure to the current
    iteration's matrix; a bare closure would read whatever the loop
    variable held at call time."""
    """Compare memory profiles for various operations."""
    print("=" * 80)
    print("DETAILED MEMORY PROFILING: gf2 vs NumPy")
    print("=" * 80)

    test_sizes = [50, 100, 200, 500]
    operations = ["rank", "nullspace", "multiply"]

    results = {op: {"gf2": [], "numpy": [], "sizes": []} for op in operations}

    for n in test_sizes:
        print(f"\n--- Testing size: {n}x{n} ---")

        matrix = generate_test_matrix(n, n, density=0.5)

        # Rank computation
        print("\n  Rank computation:")

        # gf2
        def gf2_rank_wrapped(matrix=matrix):
            A = SparseGF2Matrix(len(matrix), len(matrix[0]), matrix)
            return gf2.rank(A)

        gf2_profile = profile_memory_over_time(gf2_rank_wrapped)
        print(f"    gf2:  Peak {gf2_profile['peak_mb']:.2f} MB")

        # numpy (GF(2) row-reduction)
        def numpy_rank_wrapped(matrix=matrix):
            A = np.array(matrix, dtype=np.uint8)
            rows, cols = A.shape
            rank_val = 0
            for c in range(cols):
                pivot = None
                for r in range(rank_val, rows):
                    if A[r, c] & 1:
                        pivot = r
                        break
                if pivot is None:
                    continue
                if pivot != rank_val:
                    A[[rank_val, pivot]] = A[[pivot, rank_val]]
                for r in range(rank_val + 1, rows):
                    if A[r, c] & 1:
                        A[r, :] ^= A[rank_val, :]
                rank_val += 1
            return rank_val

        numpy_profile = profile_memory_over_time(numpy_rank_wrapped)
        print(f"    numpy:  Peak {numpy_profile['peak_mb']:.2f} MB")

        results["rank"]["gf2"].append(gf2_profile["peak_mb"])
        results["rank"]["numpy"].append(numpy_profile["peak_mb"])

        # Nullspace computation
        if n <= 200:  # Skip for very large
            print("\n  Nullspace computation:")
            nullspace_matrix = matrix[: n - 1]

            # gf2
            gf2_null_profile = profile_memory_over_time(gf2.nullspace_fast, nullspace_matrix)
            print(f"    gf2:  Peak {gf2_null_profile['peak_mb']:.2f} MB")

            # numpy (GF(2) nullspace via row-reduction)
            def numpy_nullspace_wrapped(nullspace_matrix=nullspace_matrix):
                A = np.array(nullspace_matrix, dtype=np.uint8)
                rows, cols = A.shape
                pivot_cols = []
                r = 0
                for c in range(cols):
                    pivot = None
                    for i in range(r, rows):
                        if A[i, c] & 1:
                            pivot = i
                            break
                    if pivot is None:
                        continue
                    if pivot != r:
                        A[[r, pivot]] = A[[pivot, r]]
                    pivot_cols.append(c)
                    for i in range(r + 1, rows):
                        if A[i, c] & 1:
                            A[i, :] ^= A[r, :]
                    r += 1
                    if r == rows:
                        break
                # Construct one null vector
                free_cols = sorted(set(range(cols)) - set(pivot_cols))
                if not free_cols:
                    return np.zeros((cols,), dtype=np.uint8)
                x = np.zeros(cols, dtype=np.uint8)
                x[free_cols[0]] = 1
                # back-substitute
                # Note: we don't need exact solution here, just exercise memory
                return x

            numpy_null_profile = profile_memory_over_time(numpy_nullspace_wrapped)
            print(f"    numpy:  Peak {numpy_null_profile['peak_mb']:.2f} MB")

            results["nullspace"]["gf2"].append(gf2_null_profile["peak_mb"])
            results["nullspace"]["numpy"].append(numpy_null_profile["peak_mb"])

        # Matrix multiplication
        if n <= 200:
            print("\n  Matrix multiplication:")
            matrix2 = generate_test_matrix(n, n, density=0.5)

            # gf2
            def gf2_mult_wrapped(matrix=matrix, matrix2=matrix2):
                A = SparseGF2Matrix(len(matrix), len(matrix[0]), matrix)
                B = SparseGF2Matrix(len(matrix2), len(matrix2[0]), matrix2)
                return gf2.multiply(A, B)

            gf2_mult_profile = profile_memory_over_time(gf2_mult_wrapped)
            print(f"    gf2:  Peak {gf2_mult_profile['peak_mb']:.2f} MB")

            # numpy
            def numpy_mult_wrapped(matrix=matrix, matrix2=matrix2):
                A = np.array(matrix, dtype=int)
                B = np.array(matrix2, dtype=int)
                return np.dot(A, B) % 2

            numpy_mult_profile = profile_memory_over_time(numpy_mult_wrapped)
            print(f"    numpy:  Peak {numpy_mult_profile['peak_mb']:.2f} MB")

            results["multiply"]["gf2"].append(gf2_mult_profile["peak_mb"])
            results["multiply"]["numpy"].append(numpy_mult_profile["peak_mb"])

    # Update sizes
    for op in operations:
        results[op]["sizes"] = test_sizes[: len(results[op]["gf2"])]

    return results


def plot_memory_comparison(results):
    """Generate memory comparison charts."""
    print("\n" + "=" * 80)
    print("GENERATING MEMORY COMPARISON CHARTS")
    print("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Memory Usage Comparison: gf2 vs NumPy", fontsize=16)

    operations = ["rank", "nullspace", "multiply"]
    titles = ["Rank Computation", "Nullspace Computation", "Matrix Multiplication"]

    for idx, (op, title) in enumerate(zip(operations, titles, strict=False)):
        ax = axes[idx]
        data = results[op]

        if data["gf2"] and data["numpy"]:
            sizes = data["sizes"]
            ax.plot(sizes, data["gf2"], "o-", label="gf2", linewidth=2, markersize=8)
            ax.plot(sizes, data["numpy"], "s-", label="numpy", linewidth=2, markersize=8)

            ax.set_xlabel("Matrix Size (n×n)", fontsize=12)
            ax.set_ylabel("Peak Memory (MB)", fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            # Add memory savings annotation
            if sizes:
                for i, size in enumerate(sizes):
                    if i < len(data["gf2"]) and i < len(data["numpy"]):
                        savings = (data["numpy"][i] - data["gf2"][i]) / data["numpy"][i] * 100
                        if savings > 5:  # Only show if significant
                            ax.annotate(
                                f"{savings:.0f}%",
                                xy=(size, data["gf2"][i]),
                                xytext=(0, -20),
                                textcoords="offset points",
                                ha="center",
                                fontsize=9,
                                color="green",
                                weight="bold",
                            )

    plt.tight_layout()
    plt.savefig("benchmarks/memory_comparison.png", dpi=300, bbox_inches="tight")
    print("✅ Memory chart saved to: benchmarks/memory_comparison.png")


def generate_memory_summary(results):
    """Generate summary of memory savings."""
    print("\n" + "=" * 80)
    print("MEMORY SAVINGS SUMMARY")
    print("=" * 80)

    for op in ["rank", "nullspace", "multiply"]:
        if results[op]["gf2"] and results[op]["numpy"]:
            print(f"\n--- {op.capitalize()} Operation ---")
            print(f"{'Size':<10} {'gf2 (MB)':<15} {'numpy (MB)':<15} {'Savings':<15}")
            print("-" * 60)

            sizes = results[op]["sizes"]
            for i, size in enumerate(sizes):
                if i < len(results[op]["gf2"]) and i < len(results[op]["numpy"]):
                    gf2_mem = results[op]["gf2"][i]
                    numpy_mem = results[op]["numpy"][i]
                    savings = (numpy_mem - gf2_mem) / numpy_mem * 100

                    print(f"{size}x{size:<6} {gf2_mem:<15.2f} {numpy_mem:<15.2f} {savings:>6.1f}%")


if __name__ == "__main__":
    np.random.seed(42)

    # Run memory profiling
    results = compare_memory_profiles()

    # Generate summary
    generate_memory_summary(results)

    # Plot comparison
    try:
        plot_memory_comparison(results)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    print("\n" + "=" * 80)
    print("✅ MEMORY PROFILING COMPLETE")
    print("=" * 80)
