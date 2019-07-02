# external imports
import unittest
import numpy as np
import sympy as sp

# internal inputs
from warm_start_hmpc.utils import sym2mat, unpack_bmat

class TestUtils(unittest.TestCase):

    def test_sym2mat(self):
        np.random.seed(1)

        # symbolic variables
        n = 25
        m = 12
        x = sp.Matrix(sp.symbols('x:%d'%n))

        # affine expression
        A = np.random.rand(m, n)
        b = np.random.rand(m, 1)
        expr = A * x + b

        # get matrices affine expression
        A_found, b_found = sym2mat(x, expr)
        np.testing.assert_array_almost_equal(A, A_found)
        np.testing.assert_array_almost_equal(b.flatten(), b_found)

    def test_unpack_bmat(self):

        # block dimensions
        n = 3
        m = 5
        p = 17
        indices = [n, m, p]

        # horizontal unpack
        blocks = [
            np.eye(n),
            np.zeros((n, m)),
            np.ones((n, p))
            ]
        bmat = np.hstack((blocks))
        blocks_found = unpack_bmat(bmat, indices, 'h')
        for i in range(len(indices)):
            np.testing.assert_array_equal(blocks[i], blocks_found[i])

        # vertical unpack
        blocks = [
            np.eye(n),
            np.zeros((m, n)),
            np.ones((p, n))
            ]
        bmat = np.vstack((blocks))
        blocks_found = unpack_bmat(bmat, indices, 'v')
        for i in range(len(indices)):
            np.testing.assert_array_equal(blocks[i], blocks_found[i])

        # unknown direction
        self.assertRaises(ValueError, unpack_bmat, bmat, indices, 'x')

if __name__ == '__main__':
    unittest.main()