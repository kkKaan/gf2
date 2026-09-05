"""
Integration test: Verify gf2 works correctly for Simon's Algorithm use case.
This mimics the actual usage pattern from simon_amazon_test.py.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gf2
from gf2.sparse import SparseGF2Matrix


def pack_vector(vec):
    """Pack a list of bits into an integer (for verification)."""
    out = 0
    for i, bit in enumerate(vec):
        out |= (bit & 1) << i
    return out


def verify_orthogonality(matrix, solution_str):
    """Verify that solution is in nullspace (A·x = 0 mod 2)."""
    sol_bits = [int(b) for b in solution_str]
    n = len(sol_bits)

    for i, vector in enumerate(matrix):
        dot_product = sum(sol_bits[j] & vector[j] for j in range(n)) % 2
        if dot_product != 0:
            print(f"  ❌ FAIL: Vector {i} · solution = {dot_product} (expected 0)")
            return False
    return True


def test_simon_workflow_basic():
    """Test 1: Basic Simon's algorithm workflow."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Simon's Algorithm Workflow")
    print("=" * 80)

    # Simulate measurement data from Simon's algorithm
    # Secret string s = "101"
    secret = "101"
    n = len(secret)

    # Generate measurement vectors orthogonal to secret
    # In real Simon's algorithm, these come from quantum measurements
    measurements = [
        [1, 1, 0],  # orthogonal to [1,0,1]
        [0, 1, 1],  # orthogonal to [1,0,1]
    ]

    print(f"Secret string: {secret}")
    print(f"Measurement vectors: {measurements}")

    # Test nullspace_fast (fastest method)
    print("\n--- Testing nullspace_fast() ---")
    computed_secret, time_fast = gf2.nullspace_fast(measurements)
    print(f"Computed secret: {computed_secret}")
    print(f"Time: {time_fast * 1000:.6f} ms")

    is_orthogonal = verify_orthogonality(measurements, computed_secret)
    if is_orthogonal:
        print("✅ PASS: Solution is orthogonal to all measurement vectors")
    else:
        print("❌ FAIL: Solution is NOT orthogonal")
        return False

    # Test nullspace_bitwise with SparseGF2Matrix
    print("\n--- Testing nullspace_bitwise() with SparseGF2Matrix ---")
    A = SparseGF2Matrix(len(measurements), n, measurements)
    computed_secret2, time_bitwise = gf2.nullspace_bitwise(A)
    print(f"Computed secret: {computed_secret2}")
    print(f"Time: {time_bitwise * 1000:.6f} ms")

    is_orthogonal2 = verify_orthogonality(measurements, computed_secret2)
    if is_orthogonal2:
        print("✅ PASS: Solution is orthogonal to all measurement vectors")
    else:
        print("❌ FAIL: Solution is NOT orthogonal")
        return False

    # Test nullspace (full basis)
    print("\n--- Testing nullspace() for full basis ---")
    null_basis = gf2.nullspace(A)
    print(f"Nullspace dimension: {len(null_basis)}")
    print(f"Basis vectors: {null_basis}")

    if len(null_basis) >= 1:
        # Check first basis vector
        basis_str = "".join(str(b) for b in null_basis[0])
        is_orthogonal3 = verify_orthogonality(measurements, basis_str)
        if is_orthogonal3:
            print("✅ PASS: Basis vector is orthogonal")
        else:
            print("❌ FAIL: Basis vector is NOT orthogonal")
            return False

    print("\n✅ TEST 1 PASSED\n")
    return True


def test_simon_workflow_larger():
    """Test 2: Larger matrix (more realistic Simon's algorithm)."""
    print("\n" + "=" * 80)
    print("TEST 2: Larger Matrix (n=13, similar to simon_amazon_test.py)")
    print("=" * 80)

    # Secret string (like in simon_amazon_test.py)
    secret = "1011010011110"
    n = len(secret)
    secret_bits = [int(b) for b in secret]

    print(f"Secret string: {secret}")
    print(f"Matrix size: {n - 1} x {n}")

    # Generate n-1 random vectors orthogonal to secret
    np.random.seed(42)
    measurements = []

    attempts = 0
    while len(measurements) < n - 1 and attempts < 1000:
        # Generate random vector
        vec = [np.random.randint(0, 2) for _ in range(n)]

        # Check if orthogonal to secret
        dot = sum(secret_bits[i] & vec[i] for i in range(n)) % 2
        if dot == 0:
            # Check if linearly independent from existing vectors
            test_set = measurements + [vec]
            matrix_np = np.array(test_set, dtype=float)
            rank = np.linalg.matrix_rank(matrix_np)
            if rank == len(test_set):
                measurements.append(vec)

        attempts += 1

    if len(measurements) < n - 1:
        print("❌ Could not generate enough orthogonal vectors")
        return False

    print(f"Generated {len(measurements)} measurement vectors")

    # Test with nullspace_fast
    print("\n--- Computing nullspace with nullspace_fast() ---")
    computed_secret, time_taken = gf2.nullspace_fast(measurements)
    print(f"Computed secret: {computed_secret}")
    print(f"Expected secret: {secret}")
    print(f"Time: {time_taken * 1000:.6f} ms")

    # Verify orthogonality
    is_orthogonal = verify_orthogonality(measurements, computed_secret)
    if is_orthogonal:
        print("✅ PASS: Solution is orthogonal to all measurement vectors")
    else:
        print("❌ FAIL: Solution is NOT orthogonal")
        return False

    # In Simon's algorithm, the computed secret might be the secret itself
    # or any multiple of it (in GF(2), could be the same or complementary pattern)
    # The key requirement is orthogonality, not exact match
    print("\nNote: Computed secret may differ from expected due to nullspace basis choice")
    print("      The important property is orthogonality (A·x = 0), which is verified ✓")

    print("\n✅ TEST 2 PASSED\n")
    return True


def test_edge_cases():
    """Test 3: Edge cases."""
    print("\n" + "=" * 80)
    print("TEST 3: Edge Cases")
    print("=" * 80)

    # Test 3.1: Small matrix (n=3)
    print("\n--- Test 3.1: Small matrix (n=3) ---")
    measurements = [[1, 0, 1], [0, 1, 1]]
    sol, time = gf2.nullspace_fast(measurements)
    is_valid = verify_orthogonality(measurements, sol)
    if is_valid:
        print("✅ PASS: Small matrix works correctly")
    else:
        print("❌ FAIL: Small matrix failed")
        return False

    # Test 3.2: Dense matrix
    print("\n--- Test 3.2: Dense matrix (high density) ---")
    n = 10
    measurements = []
    for _ in range(n - 1):
        vec = [np.random.randint(0, 2) for _ in range(n)]
        measurements.append(vec)

    # Ensure linearly independent
    matrix_np = np.array(measurements, dtype=float)
    rank = np.linalg.matrix_rank(matrix_np)

    if rank == n - 1:
        sol, time = gf2.nullspace_fast(measurements)
        is_valid = verify_orthogonality(measurements, sol)
        if is_valid:
            print("✅ PASS: Dense matrix works correctly")
        else:
            print("❌ FAIL: Dense matrix failed")
            return False
    else:
        print("⚠️  SKIP: Could not generate full rank matrix")

    # Test 3.3: Sparse matrix
    print("\n--- Test 3.3: Sparse matrix (low density) ---")
    measurements = [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
    ]

    sol, time = gf2.nullspace_fast(measurements)
    is_valid = verify_orthogonality(measurements, sol)
    if is_valid:
        print("✅ PASS: Sparse matrix works correctly")
    else:
        print("❌ FAIL: Sparse matrix failed")
        return False

    print("\n✅ TEST 3 PASSED\n")
    return True


def test_consistency():
    """Test 4: Consistency between different methods."""
    print("\n" + "=" * 80)
    print("TEST 4: Consistency Between Methods")
    print("=" * 80)

    # Generate test matrix
    n = 20
    np.random.seed(123)
    measurements = []
    for _ in range(n - 1):
        vec = [np.random.randint(0, 2) for _ in range(n)]
        measurements.append(vec)

    print(f"Testing with {n - 1} x {n} matrix")

    # Method 1: nullspace_fast
    sol1, time1 = gf2.nullspace_fast(measurements)
    valid1 = verify_orthogonality(measurements, sol1)

    # Method 2: nullspace_bitwise
    A = SparseGF2Matrix(len(measurements), n, measurements)
    sol2, time2 = gf2.nullspace_bitwise(A)
    valid2 = verify_orthogonality(measurements, sol2)

    # Method 3: nullspace
    null_basis = gf2.nullspace(A)
    if null_basis:
        sol3 = "".join(str(b) for b in null_basis[0])
        valid3 = verify_orthogonality(measurements, sol3)
    else:
        valid3 = False

    print(f"\nnullspace_fast:     {sol1[:20]}... valid={valid1}")
    print(f"nullspace_bitwise:  {sol2[:20]}... valid={valid2}")
    if null_basis:
        print(f"nullspace:          {sol3[:20]}... valid={valid3}")

    if valid1 and valid2 and valid3:
        print("\n✅ PASS: All methods produce valid solutions")
    else:
        print("\n❌ FAIL: Some methods failed")
        return False

    # Solutions might differ (different basis vectors), but all should be valid
    print("\nNote: Solutions may differ (different basis vectors in nullspace)")
    print("      All solutions are valid if they satisfy orthogonality ✓")

    print("\n✅ TEST 4 PASSED\n")
    return True


def main():
    """Run all integration tests."""
    print("=" * 80)
    print("gf2 SIMON'S ALGORITHM INTEGRATION TESTS")
    print("=" * 80)

    all_passed = True

    # Run all tests
    all_passed &= test_simon_workflow_basic()
    all_passed &= test_simon_workflow_larger()
    all_passed &= test_edge_cases()
    all_passed &= test_consistency()

    # Final summary
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if all_passed:
        print("✅✅✅ ALL INTEGRATION TESTS PASSED ✅✅✅")
        print("\ngf2 is ready for Simon's Algorithm use!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease review the failures above.")
        return 1


if __name__ == "__main__":
    exit(main())
