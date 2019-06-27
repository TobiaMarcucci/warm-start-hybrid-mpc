# external imports
import unittest
import numpy as np
import sympy as sp

# internal inputs
from warm_start_hmpc.utils import sym2mat, unpack_bmat

class TestUtils(unittest.TestCase):

    def test_sym2mat(self):

    	# symbolic variables
    	x = sp.Matrix(sp.symbols('x1 x2 x3 x4'))
    	n = x.shape[0]

    	# affine expression
    	expr = sp.Matrix([
    		3*x[0] + 5,
    		2*x[2] - x[3] - 1,
    		-1*x[1] + 2,
    		])
    	A = np.array([
    		[3,  0, 0,  0],
    		[0,  0, 2, -1],
    		[0, -1, 0,  0],
    		])
    	b = np.array([5, -1, 2])
    	
    	# get matrices affine expression
    	A_found, b_found = sym2mat(x, expr)
    	np.testing.assert_array_equal(A, A_found)
    	np.testing.assert_array_equal(b, b_found)

    def test_unpack_bmat(self):

        # block dimensions
    	n = 3
    	m = 5
    	p = 17
    	indices = [n, m, p]

    	# horizontal unpack
    	blocks = [np.eye(n), np.zeros((n, m)), np.ones((n, p))]
    	bmat = np.hstack((blocks))
    	blocks_found = unpack_bmat(bmat, indices, 'h')
    	for i in range(len(indices)):
    		np.testing.assert_array_equal(blocks[i], blocks_found[i])

    	# vertical unpack
    	blocks = [np.eye(n), np.zeros((m, n)), np.ones((p, n))]
    	bmat = np.vstack((blocks))
    	blocks_found = unpack_bmat(bmat, indices, 'v')
    	for i in range(len(indices)):
    		np.testing.assert_array_equal(blocks[i], blocks_found[i])

    	# unknown direction
    	self.assertRaises(ValueError, unpack_bmat, bmat, indices, 'x')

if __name__ == '__main__':
    unittest.main()