"""
Final verification: Ensure gf2 is production-ready.
This runs all critical checks before pushing.
"""

import subprocess
import sys


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print("=" * 80)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0


def main():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION - gf2 PRODUCTION READINESS")
    print("=" * 80)

    all_passed = True

    # Check 1: Run all repository tests
    print("\n[1/4] Running all repository tests...")
    passed = run_command(
        f"{sys.executable} -m pytest tests/ --override-ini='addopts=' -q", "Repository Test Suite"
    )
    if passed:
        print("✅ All repository tests PASSED")
    else:
        print("❌ Some repository tests FAILED")
        all_passed = False

    # Check 2: Run Simon's algorithm integration tests
    print("\n[2/4] Running Simon's algorithm integration tests...")
    passed = run_command(f"{sys.executable} test_simon_integration.py", "Simon's Algorithm Integration Tests")
    if passed:
        print("✅ Simon's algorithm integration PASSED")
    else:
        print("❌ Simon's algorithm integration FAILED")
        all_passed = False

    # Check 3: Run performance benchmark
    print("\n[3/4] Running performance benchmark...")
    passed = run_command(f"{sys.executable} quick_benchmark.py", "Performance Benchmark")
    if passed:
        print("✅ Performance benchmark PASSED")
    else:
        print("❌ Performance benchmark FAILED")
        all_passed = False

    # Check 4: Test import and basic usage
    print("\n[4/4] Testing import and basic usage...")
    try:
        import gf2

        # Test basic operations
        matrix = [[1, 0, 1], [0, 1, 1]]

        # Test nullspace_fast
        sol1, _ = gf2.nullspace_fast(matrix)

        # Test with SparseGF2Matrix
        A = gf2.SparseGF2Matrix(2, 3, matrix)
        sol2, _ = gf2.nullspace_bitwise(A)
        null_basis = gf2.nullspace(A)

        # Test other operations
        rank = gf2.rank(A)

        print("✅ Import and basic usage PASSED")
        print(f"   nullspace_fast result: {sol1}")
        print(f"   nullspace_bitwise result: {sol2}")
        print(f"   nullspace full basis: {null_basis}")
        print(f"   Matrix rank: {rank}")

    except Exception as e:
        print(f"❌ Import and basic usage FAILED: {e}")
        all_passed = False

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if all_passed:
        print("\n✅✅✅ ALL VERIFICATION CHECKS PASSED ✅✅✅")
        print("\ngf2 is PRODUCTION READY!")
        print("\nYou can safely:")
        print("  - git add .")
        print("  - git commit -m 'Optimize performance to match reference implementation'")
        print("  - git push")
        return 0
    else:
        print("\n❌ SOME VERIFICATION CHECKS FAILED")
        print("\nPlease review the failures above before pushing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
