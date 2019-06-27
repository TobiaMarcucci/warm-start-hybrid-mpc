# external imports
import unittest
import numpy as np
import gurobipy as grb
from operator import le, ge, eq

# internal inputs
from warm_start_hmpc.bounded_qp import BoundedQP

class TestBoundedQP(unittest.TestCase):

    def test_variables(self):

        # initialize model
        model = BoundedQP()

        # add optimization variables
        n = 15
        x = model.add_variables(n, name='x')

        # check size of the numpy array
        self.assertEqual(x.size, n)

        for i, xi in enumerate(x):

            # check name of the variables
            self.assertEqual(xi.VarName, 'x[%d]'%i)

            # check that the lower bound is -inf
            self.assertEqual(xi.LB, -grb.GRB.INFINITY)

        # try to set a different lower bound
        lb = [5.] * n
        y = model.add_variables(n, name='y', lb=lb)
        for yi in y:
            self.assertEqual(yi.LB, 5)

        # get variables back
        for i, xi in enumerate(model.get_variables('x')):
            self.assertTrue(x[i].sameAs(xi))

        # returns empty vector if variable is not defined
        z = model.get_variables('z')
        self.assertEqual(z.size, 0)

    def test_constraints(self):

        # initialize model
        model = BoundedQP()

        # add optimization variables
        n = 5
        x = model.add_variables(n)
        x_index = {xi: i for i, xi in enumerate(x)}

        # gurobi symbols for the operators
        operators = [(le,'<'), (eq,'='), (ge,'>')]

        # add one set of constraints per operator
        A = np.eye(n)
        b = np.ones(n)
        for op in operators:

            # add constraints, named as the operator
            c = model.add_constraints(A.dot(x), op[0], b, name=op[1])

            # check size of the numpy array
            self.assertEqual(c.size, n)

            for i, ci in enumerate(c):

                # check sense of the constraint
                self.assertEqual(ci.Sense, op[1])

                # check left-hand side
                rowi = model.getRow(ci)
                for j in range(rowi.size()):
                    xij = rowi.getVar(j)
                    self.assertEqual(rowi.getCoeff(j), A[i, x_index[xij]])

                # check right-hand side
                self.assertEqual(ci.RHS, 1.)

                # check name of the constraint
                self.assertEqual(ci.ConstrName, op[1]+'[%d]'%i)

            # get constraints back
            for i, ci in enumerate(model.get_constraints(op[1])):
                self.assertTrue(c[i].sameAs(ci))

        # returns empty vector if variable is not defined
        d = model.get_variables('d')
        self.assertEqual(d.size, 0)

        # raise error if incoherent sizes
        b_wrong = np.ones(n+1)
        self.assertRaises(ValueError, model.add_constraints, x, eq, b_wrong)

        # raise error if lhs is array of floats
        self.assertRaises(ValueError, model.add_constraints, b, le, x)

    def test_optimizer_feasible(self):

        # initialize model
        model = BoundedQP()

        # add optimization variables
        n = 15
        x = model.add_variables(n, name='x')

        # add constraints
        lb = [1.] * n
        model.add_constraints(x, ge, lb, name='c1')

        # set objective
        model.setObjective(.5 * x.dot(x))

        # no solution available yet
        self.assertRaises(ValueError, model.get_primal_optimizer, 'x')
        self.assertRaises(ValueError, model.get_dual_optimizer, 'c1')

        # check primal optimizer
        model.optimize()
        x_found = model.get_primal_optimizer('x')
        x_opt = lb
        np.testing.assert_array_almost_equal(x_found, x_opt)

        # check dual optimizer
        p_found = model.get_dual_optimizer('c1')
        p_opt = [-1.] * n
        np.testing.assert_array_almost_equal(p_found, p_opt)

        # check sign multipliers with opposite sense (le -> positive)
        model.add_constraints([x[1]+2.], le, [x[0]], name='c2')
        model.optimize()
        p_found = model.get_dual_optimizer('c2')
        self.assertTrue(p_found > 0.)

        # check sign multipliers with opposite sense (ge -> negative)
        model.add_constraints([x[0]], ge, [x[1]+3.], name='c3')
        model.optimize()
        p_found = model.get_dual_optimizer('c3')
        self.assertTrue(p_found < 0.)

    def test_optimize_infeasible(self):

        # initialize model
        model = BoundedQP()

        # add optimization variables
        n = 15
        x = model.add_variables(n, name='x')

        # set objective
        model.setObjective(x[0]*x[0])

        # add infeasible constraints
        model.add_constraints(x, ge, [1.]*n, name='c1')
        model.add_constraints(x, le, [0.]*n, name='c2')
        model.optimize()

        # primal None if infeasible
        x_found = model.get_primal_optimizer('x')
        self.assertIsNone(x_found)

        # check signs farkas proof
        p1_found = model.get_dual_optimizer('c1')
        p2_found = model.get_dual_optimizer('c2')
        self.assertTrue(max(p1_found) <= 0.)
        self.assertTrue(min(p2_found) >= 0.)
        np.testing.assert_array_almost_equal(p1_found, -p2_found)

    def test_optimize_unbounded(self):

        # initialize model
        model = BoundedQP()

        # add optimization variables
        n = 15
        x = model.add_variables(n)

        # set objective
        model.setObjective(x[:-1].dot(x[:-1]) + x[-1])

        # add constraints
        model.add_constraints(x, le, [1.]*n)
        self.assertRaises(ValueError, model.optimize)

if __name__ == '__main__':
    unittest.main()