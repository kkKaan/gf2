from binpy.generators import identity
from binpy.solvers import inverse, solve


def test_solve_identity_system():
    identity_matrix = identity(6)
    b = [1, 0, 1, 0, 1, 0]
    x = solve(identity_matrix, b)
    assert x == b


def test_inverse_identity():
    identity_matrix = identity(7)
    inv = inverse(identity_matrix)
    for i in range(identity_matrix.rows):
        assert inv.get_row_bitwise(i) == identity_matrix.get_row_bitwise(i)
