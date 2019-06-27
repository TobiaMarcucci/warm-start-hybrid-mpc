# external imports
import unittest
import numpy as np
import sympy as sp

# internal inputs
from warm_start_hmpc.mld_system import MLDSystem

class TestMLDSystem(unittest.TestCase):

    def test_init(self):

        # system size
        nx = 10
        nu = 15
        nub = 5 
        nc = 20
        
        # correct-size dynamics
        A = np.ones((nx, nx))
        B = np.ones((nx, nu))

        # correct-size constraints
        F = np.ones((nc, nx))
        G = np.ones((nc, nu))
        h = np.ones(nc)
        
        # wrong size A and B
        d_wrong = [
            [np.ones((nx+1, nx)), B],
            [np.ones((nx, nx+1)), B],
            [A, np.ones((nx+1, nu))],
            [A, np.ones((nx, nu+1))]
            ]
        c = [F, G, h]
        for d in d_wrong:
            self.assertRaises(ValueError, MLDSystem, d, c, nub)

        # wrong size F, G, and h
        d = [A, B]
        c_wrong = [
            [np.ones((nc+1, nx)), G, h],
            [np.ones((nc, nx+1)), G, h],
            [F, np.ones((nc+1, nu)), h],
            [F, np.ones((nc, nu+1)), h],
            [F, G, np.ones(nc+1)],
            ]
        for c in c_wrong:
            self.assertRaises(ValueError, MLDSystem, d, c, nub)

        # from symbolic
        x = sp.Matrix(sp.symbols('x:%d'%nx))
        u = sp.Matrix(sp.symbols('u:%d'%nu))
        d = A * x + B * u
        c = F * x + G * u - h.reshape(nc, 1)
        mld = MLDSystem.from_symbolic(d, c, x, u, nub)
        np.testing.assert_array_almost_equal(A, mld.A)
        np.testing.assert_array_almost_equal(B, mld.B)
        np.testing.assert_array_almost_equal(F, mld.F)
        np.testing.assert_array_almost_equal(G, mld.G)
        np.testing.assert_array_almost_equal(h, mld.h)

        # test selection matrix
        self.assertEqual(sum(mld.V*u - u[-nub:,:]), 0)

        # affine dynamics
        d += np.ones((nx, 1))
        self.assertRaises(ValueError, MLDSystem.from_symbolic, d, c, x, u, nub)

if __name__ == '__main__':
    unittest.main()