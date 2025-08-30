"""
Mathematical verification and performance measurement utilities for binpy testing.

This module provides:
- MathVerifier: Utilities for verifying mathematical properties and correctness
- PerformanceMeasurer: Benchmarking and memory measurement capabilities
- CoverageTracker: Test coverage monitoring across modules
"""

import gc
import os
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Optional imports with fallbacks
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    psutil = None

from binpy.core import add, det, multiply, rank, trace, transpose
from binpy.generators import identity, zeros
from binpy.solvers import nullspace
from binpy.sparse import DenseGF2Matrix, SparseGF2Matrix


@dataclass
class TestResult:
    """Store test execution results."""

    test_name: str
    status: str  # 'passed', 'failed', 'skipped'
    execution_time: float
    memory_usage: int
    error_message: str | None = None
    additional_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Store performance benchmark results."""

    operation_name: str
    matrix_size: tuple[int, int]
    sparsity: float
    execution_time: float
    memory_usage: int
    operations_per_second: float
    additional_metrics: dict[str, float] = field(default_factory=dict)


class MathVerifier:
    """Utilities for verifying mathematical properties and correctness."""

    @staticmethod
    def verify_matrix_equality(
        A: SparseGF2Matrix | DenseGF2Matrix, B: SparseGF2Matrix | DenseGF2Matrix, tolerance: int = 0
    ) -> bool:
        """
        Verify two matrices are equal.

        Args:
            A, B: Matrices to compare
            tolerance: Not used for binary matrices, kept for interface compatibility

        Returns:
            True if matrices are equal, False otherwise
        """
        if A.rows != B.rows or A.cols != B.cols:
            return False

        return all(A.get_row_bitwise(i) == B.get_row_bitwise(i) for i in range(A.rows))

    @staticmethod
    def verify_algebraic_property(
        operation: str, matrices: list[SparseGF2Matrix | DenseGF2Matrix], property_name: str
    ) -> bool:
        """
        Verify algebraic properties like associativity, commutativity.

        Args:
            operation: Operation name ('add', 'multiply')
            matrices: List of matrices to test
            property_name: Property to verify ('associative', 'commutative', 'identity')

        Returns:
            True if property holds, False otherwise
        """
        if operation == "add":
            return MathVerifier._verify_addition_property(matrices, property_name)
        elif operation == "multiply":
            return MathVerifier._verify_multiplication_property(matrices, property_name)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    @staticmethod
    def _verify_addition_property(
        matrices: list[SparseGF2Matrix | DenseGF2Matrix], property_name: str
    ) -> bool:
        """Verify addition properties."""
        if property_name == "commutative" and len(matrices) >= 2:
            A, B = matrices[0], matrices[1]
            if A.rows != B.rows or A.cols != B.cols:
                return False
            AB = add(A, B)
            BA = add(B, A)
            return MathVerifier.verify_matrix_equality(AB, BA)

        elif property_name == "associative" and len(matrices) >= 3:
            A, B, C = matrices[0], matrices[1], matrices[2]
            if not (A.rows == B.rows == C.rows and A.cols == B.cols == C.cols):
                return False
            AB_C = add(add(A, B), C)
            A_BC = add(A, add(B, C))
            return MathVerifier.verify_matrix_equality(AB_C, A_BC)

        elif property_name == "identity" and len(matrices) >= 1:
            A = matrices[0]
            zero = zeros(A.rows, A.cols)
            A_plus_zero = add(A, zero)
            return MathVerifier.verify_matrix_equality(A, A_plus_zero)

        elif property_name == "inverse" and len(matrices) >= 1:
            # In GF(2), A + A = 0 (A is its own inverse)
            A = matrices[0]
            A_plus_A = add(A, A)
            zero = zeros(A.rows, A.cols)
            return MathVerifier.verify_matrix_equality(A_plus_A, zero)

        return False

    @staticmethod
    def _verify_multiplication_property(
        matrices: list[SparseGF2Matrix | DenseGF2Matrix], property_name: str
    ) -> bool:
        """Verify multiplication properties."""
        if property_name == "associative" and len(matrices) >= 3:
            A, B, C = matrices[0], matrices[1], matrices[2]
            if A.cols != B.rows or B.cols != C.rows:
                return False
            AB_C = multiply(multiply(A, B), C)
            A_BC = multiply(A, multiply(B, C))
            return MathVerifier.verify_matrix_equality(AB_C, A_BC)

        elif property_name == "identity" and len(matrices) >= 1:
            A = matrices[0]
            if A.rows != A.cols:
                return False
            identity_matrix = identity(A.rows)
            AI = multiply(A, identity_matrix)
            IA = multiply(identity_matrix, A)
            return MathVerifier.verify_matrix_equality(A, AI) and MathVerifier.verify_matrix_equality(A, IA)

        return False

    @staticmethod
    def verify_rank_nullity(matrix: SparseGF2Matrix | DenseGF2Matrix) -> bool:
        """
        Verify rank-nullity theorem: rank(A) + nullity(A) = cols(A).

        Args:
            matrix: Matrix to verify

        Returns:
            True if rank-nullity theorem holds
        """
        try:
            matrix_rank = rank(matrix)
            null_space = nullspace(matrix)
            nullity = len(null_space) if null_space else 0
            return matrix_rank + nullity == matrix.cols
        except Exception:
            # If nullspace computation fails, we can't verify
            return False

    @staticmethod
    def verify_solution(A: SparseGF2Matrix | DenseGF2Matrix, b: list[int], x: list[int]) -> bool:
        """
        Verify that x solves Ax = b.

        Args:
            A: Coefficient matrix
            b: Right-hand side vector
            x: Solution vector

        Returns:
            True if Ax = b, False otherwise
        """
        if len(x) != A.cols or len(b) != A.rows:
            return False

        # Convert x to column matrix and multiply
        result = []
        for i in range(A.rows):
            row_sum = 0
            for j in range(A.cols):
                if A.get_bit(i, j):
                    row_sum ^= x[j]  # XOR in GF(2)
            result.append(row_sum)

        return result == b

    @staticmethod
    def verify_transpose_properties(A: SparseGF2Matrix | DenseGF2Matrix) -> dict[str, bool]:
        """
        Verify transpose properties.

        Args:
            A: Matrix to test

        Returns:
            Dictionary of property verification results
        """
        results = {}

        # (A^T)^T = A
        AT = transpose(A)
        ATT = transpose(AT)
        results["involution"] = MathVerifier.verify_matrix_equality(A, ATT)

        # rank(A) = rank(A^T)
        results["rank_invariant"] = rank(A) == rank(AT)

        return results

    @staticmethod
    def verify_determinant_properties(
        A: SparseGF2Matrix | DenseGF2Matrix, B: SparseGF2Matrix | DenseGF2Matrix
    ) -> dict[str, bool]:
        """
        Verify determinant properties for square matrices.

        Args:
            A, B: Square matrices of same size

        Returns:
            Dictionary of property verification results
        """
        results = {}

        if A.rows != A.cols or B.rows != B.cols or A.rows != B.rows:
            return {"error": "Matrices must be square and same size"}

        try:
            det_A = det(A)
            det_B = det(B)
            det_AT = det(transpose(A))

            # det(A^T) = det(A)
            results["transpose_invariant"] = det_A == det_AT

            # det(A * B) = det(A) * det(B) in GF(2)
            if A.cols == B.rows:
                AB = multiply(A, B)
                det_AB = det(AB)
                results["multiplicative"] = det_AB == (det_A * det_B) % 2

        except Exception as e:
            results["error"] = str(e)

        return results


class PerformanceMeasurer:
    """Utilities for measuring performance and memory usage."""

    def __init__(self):
        self.results: list[BenchmarkResult] = []
        if _HAS_PSUTIL:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None

    def benchmark_operation(self, operation: Callable, *args, **kwargs) -> BenchmarkResult:
        """
        Benchmark an operation and record results.

        Args:
            operation: Function to benchmark
            *args, **kwargs: Arguments to pass to the operation

        Returns:
            BenchmarkResult with timing and memory data
        """
        # Force garbage collection before measurement
        gc.collect()

        # Start memory tracking
        tracemalloc.start()
        initial_memory = self.process.memory_info().rss if self.process else 0

        # Benchmark the operation
        start_time = time.perf_counter()
        try:
            _ = operation(*args, **kwargs)
            success = True
        except Exception:
            success = False

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Measure memory usage
        current_memory = self.process.memory_info().rss if self.process else 0
        memory_usage = current_memory - initial_memory

        # Stop memory tracking
        tracemalloc.stop()

        # Calculate operations per second
        ops_per_second = 1.0 / execution_time if execution_time > 0 else float("inf")

        # Determine matrix size and sparsity if possible
        matrix_size = (0, 0)
        sparsity = 0.0

        if args and hasattr(args[0], "rows") and hasattr(args[0], "cols"):
            matrix = args[0]
            matrix_size = (matrix.rows, matrix.cols)
            # Estimate sparsity
            total_elements = matrix.rows * matrix.cols
            if total_elements > 0:
                non_zero_count = sum(
                    1 for i in range(matrix.rows) for j in range(matrix.cols) if matrix.get_bit(i, j)
                )
                sparsity = non_zero_count / total_elements

        benchmark_result = BenchmarkResult(
            operation_name=operation.__name__ if hasattr(operation, "__name__") else str(operation),
            matrix_size=matrix_size,
            sparsity=sparsity,
            execution_time=execution_time,
            memory_usage=memory_usage,
            operations_per_second=ops_per_second,
            additional_metrics={"success": success},
        )

        self.results.append(benchmark_result)
        return benchmark_result

    def measure_memory_usage(self, matrix: SparseGF2Matrix | DenseGF2Matrix) -> dict[str, int]:
        """
        Measure memory usage of matrix storage.

        Args:
            matrix: Matrix to measure

        Returns:
            Dictionary with memory usage statistics
        """
        # Get process memory before and after creating matrix reference
        gc.collect()
        initial_memory = self.process.memory_info().rss if self.process else 0

        # Create a reference to force memory allocation if needed
        _ = matrix  # Keep reference to ensure memory allocation
        current_memory = self.process.memory_info().rss if self.process else 0

        # Estimate matrix memory usage
        matrix_memory = current_memory - initial_memory

        # Calculate theoretical memory usage
        total_elements = matrix.rows * matrix.cols
        theoretical_dense = total_elements // 8 + (1 if total_elements % 8 else 0)  # bits to bytes

        # Count actual non-zero elements
        non_zero_count = sum(
            1 for i in range(matrix.rows) for j in range(matrix.cols) if matrix.get_bit(i, j)
        )

        return {
            "measured_bytes": matrix_memory,
            "theoretical_dense_bytes": theoretical_dense,
            "non_zero_elements": non_zero_count,
            "total_elements": total_elements,
            "compression_ratio": theoretical_dense / max(matrix_memory, 1),
            "sparsity": 1.0 - (non_zero_count / max(total_elements, 1)),
        }

    def compare_implementations(
        self, operations: list[tuple[str, Callable]], test_cases: list[tuple]
    ) -> dict[str, list[BenchmarkResult]]:
        """
        Compare performance across different implementations.

        Args:
            operations: List of (name, function) tuples
            test_cases: List of argument tuples for testing

        Returns:
            Dictionary mapping operation names to benchmark results
        """
        comparison_results = defaultdict(list)

        def _failed_result(name: str, msg: str = "Operation is not callable") -> BenchmarkResult:
            """Create a small failure BenchmarkResult to avoid duplicating construction code."""
            return BenchmarkResult(
                operation_name=name,
                matrix_size=(0, 0),
                sparsity=0.0,
                execution_time=float("inf"),
                memory_usage=0,
                operations_per_second=0.0,
                additional_metrics={"error": msg},
            )

        for test_case in test_cases:
            for op_name, op_func in operations:
                if not callable(op_func):
                    comparison_results[op_name].append(_failed_result(op_name))
                    continue

                try:
                    result = self.benchmark_operation(op_func, *test_case)
                    # label the result with the provided operation name
                    result.operation_name = op_name
                    comparison_results[op_name].append(result)
                except Exception as e:
                    comparison_results[op_name].append(_failed_result(op_name, str(e)))

        return dict(comparison_results)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get summary statistics of all benchmarks."""
        if not self.results:
            return {}

        summary = {
            "total_benchmarks": len(self.results),
            "operations": defaultdict(list),
            "average_execution_time": 0.0,
            "total_memory_used": 0,
        }

        total_time = 0
        total_memory = 0

        for result in self.results:
            summary["operations"][result.operation_name].append(result)
            total_time += result.execution_time
            total_memory += result.memory_usage

        summary["average_execution_time"] = total_time / len(self.results)
        summary["total_memory_used"] = total_memory

        return summary


class CoverageTracker:
    """Track test coverage across modules."""

    def __init__(self):
        self.module_coverage: dict[str, dict[str, Any]] = defaultdict(dict)
        self.function_coverage: dict[str, dict[str, int]] = defaultdict(dict)
        self.line_coverage: dict[str, dict[int, int]] = defaultdict(dict)
        self.test_results: list[TestResult] = []

    def record_coverage(self, module: str, function: str, lines_covered: list[int]):
        """
        Record coverage information.

        Args:
            module: Module name
            function: Function name
            lines_covered: List of line numbers covered
        """
        if module not in self.module_coverage:
            self.module_coverage[module] = {
                "functions_tested": set(),
                "total_lines_covered": set(),
                "test_count": 0,
            }

        self.module_coverage[module]["functions_tested"].add(function)
        self.module_coverage[module]["total_lines_covered"].update(lines_covered)
        self.module_coverage[module]["test_count"] += 1

        # Record function-level coverage
        if function not in self.function_coverage[module]:
            self.function_coverage[module][function] = 0
        self.function_coverage[module][function] += 1

        # Record line-level coverage
        for line in lines_covered:
            if line not in self.line_coverage[module]:
                self.line_coverage[module][line] = 0
            self.line_coverage[module][line] += 1

    def record_test_result(self, test_result: TestResult):
        """Record a test result."""
        self.test_results.append(test_result)

    def generate_report(self) -> dict[str, Any]:
        """
        Generate coverage report.

        Returns:
            Dictionary with coverage statistics
        """
        report = {
            "modules": {},
            "summary": {
                "total_modules": len(self.module_coverage),
                "total_tests": len(self.test_results),
                "passed_tests": sum(1 for r in self.test_results if r.status == "passed"),
                "failed_tests": sum(1 for r in self.test_results if r.status == "failed"),
                "skipped_tests": sum(1 for r in self.test_results if r.status == "skipped"),
            },
        }

        for module, coverage_data in self.module_coverage.items():
            report["modules"][module] = {
                "functions_tested": len(coverage_data["functions_tested"]),
                "lines_covered": len(coverage_data["total_lines_covered"]),
                "test_count": coverage_data["test_count"],
                "function_details": dict(self.function_coverage[module]),
                "line_details": dict(self.line_coverage[module]),
            }

        return report

    def get_uncovered_functions(self, module: str, all_functions: list[str]) -> list[str]:
        """
        Get list of functions not covered by tests.

        Args:
            module: Module name
            all_functions: List of all functions in the module

        Returns:
            List of uncovered function names
        """
        if module not in self.module_coverage:
            return all_functions

        tested_functions = self.module_coverage[module]["functions_tested"]
        return [func for func in all_functions if func not in tested_functions]

    def calculate_coverage_percentage(self, module: str, total_lines: int) -> float:
        """
        Calculate coverage percentage for a module.

        Args:
            module: Module name
            total_lines: Total number of lines in the module

        Returns:
            Coverage percentage (0.0 to 100.0)
        """
        if module not in self.module_coverage or total_lines == 0:
            return 0.0

        covered_lines = len(self.module_coverage[module]["total_lines_covered"])
        return (covered_lines / total_lines) * 100.0


# Utility functions for common verification patterns
def verify_matrix_operation_consistency(
    operation_name: str, matrices: list[SparseGF2Matrix | DenseGF2Matrix], expected_properties: list[str]
) -> dict[str, bool]:
    """
    Verify multiple properties of a matrix operation.

    Args:
        operation_name: Name of the operation to test
        matrices: List of matrices for testing
        expected_properties: List of properties to verify

    Returns:
        Dictionary mapping property names to verification results
    """
    verifier = MathVerifier()
    results = {}

    try:
        for prop in expected_properties:
            results[prop] = verifier.verify_algebraic_property(operation_name, matrices, prop)
    except Exception as e:
        # If any exception occurs, mark all properties as False and record the error
        for prop in expected_properties:
            results[prop] = False
            results[f"{prop}_error"] = str(e)

    return results


def benchmark_matrix_operations(
    matrices: list[SparseGF2Matrix | DenseGF2Matrix], operations: list[str]
) -> dict[str, BenchmarkResult]:
    """
    Benchmark multiple operations on given matrices.

    Args:
        matrices: List of matrices to test
        operations: List of operation names to benchmark

    Returns:
        Dictionary mapping operation names to benchmark results
    """
    measurer = PerformanceMeasurer()
    results = {}

    operation_map = {
        "rank": rank,
        "det": det,
        "trace": trace,
        "transpose": transpose,
    }

    for op_name in operations:
        if op_name in operation_map and matrices:
            try:
                # Use first matrix for single-matrix operations
                result = measurer.benchmark_operation(operation_map[op_name], matrices[0])
                results[op_name] = result
            except Exception as e:
                results[f"{op_name}_error"] = str(e)

    return results
