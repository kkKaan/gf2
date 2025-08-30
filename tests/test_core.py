import pytest
from binpy.core import add, det, is_invertible, multiply, rank, transpose
from binpy.generators import identity, zeros


@pytest.mark.unit
def test_identity_rank_and_invertibility():
    identity_matrix = identity(8)
    assert rank(identity_matrix) == 8
    assert det(identity_matrix) == 1
    assert is_invertible(identity_matrix)


@pytest.mark.unit
def test_addition_xor_behavior():
    A = identity(4)
    B = identity(4)
    C = add(A, B)
    # I ^ I = 0
    Z = zeros(4, 4)
    for i in range(Z.rows):
        assert C.get_row_bitwise(i) == Z.get_row_bitwise(i)


@pytest.mark.unit
def test_multiply_and_transpose_consistency():
    A = identity(5)
    AT = transpose(A)
    B = multiply(A, AT)
    assert rank(B) == 5
