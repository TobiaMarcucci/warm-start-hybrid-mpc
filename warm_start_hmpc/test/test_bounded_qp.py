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
        qp = BoundedQP()

        # add optimization variables
        n = 15
        x = qp.add_variables(n, name='x')

        # check size of the numpy array
        self.assertEqual(x.size, n)

        for i, xi in enumerate(x):

            # check name of the variables
            self.assertEqual(xi.VarName, 'x[%d]'%i)

            # check that the lower bound is -inf
            self.assertEqual(xi.LB, -grb.GRB.INFINITY)

        # get variables back
        for i, xi in enumerate(qp.get_variables('x')):
            self.assertTrue(x[i].sameAs(xi))

        # try to set a bound
        b = [0.] * n
        self.assertRaises(KeyError, qp.add_variables, n, lb=b)
        self.assertRaises(KeyError, qp.add_variables, n, ub=b)

        # returns empty vector if variable is not defined
        self.assertEqual(qp.get_variables('y').size, 0)

    def test_constraints(self):

        # initialize model
        qp = BoundedQP()

        # add optimization variables
        n = 5
        x = qp.add_variables(n)
        x_index = {xi: i for i, xi in enumerate(x)}

        # gurobi symbols for the operators
        operators = [(le,'<'), (eq,'='), (ge,'>')]

        # add one set of constraints per operator
        A = np.eye(n)
        b = np.ones(n)
        for op in operators:

            # add constraints, named as the operator
            c = qp.add_constraints(A.dot(x), op[0], b, name=op[1])

            # check size of the numpy array
            self.assertEqual(c.size, n)

            for i, ci in enumerate(c):

                # check sense of the constraint
                self.assertEqual(ci.Sense, op[1])

                # check left-hand side
                rowi = qp.getRow(ci)
                for j in range(rowi.size()):
                    xij = rowi.getVar(j)
                    self.assertEqual(rowi.getCoeff(j), A[i, x_index[xij]])

                # check right-hand side
                self.assertEqual(ci.RHS, 1.)

                # check name of the constraint
                self.assertEqual(ci.ConstrName, op[1]+'[%d]'%i)

            # get constraints back
            for i, ci in enumerate(qp.get_constraints(op[1])):
                self.assertTrue(c[i].sameAs(ci))

            # change rhs
            qp.set_constraint_rhs(op[1], 2*b)
            np.testing.assert_array_equal(2*b, qp.get_constraint_rhs(op[1]))

            # wrong size rhs
            self.assertRaises(ValueError, qp.set_constraint_rhs, op[1], np.ones(2*n))

        # returns empty vector if variable is not defined
        self.assertEqual(qp.get_constraints('d').size, 0)

        # raise error if incoherent sizes
        self.assertRaises(ValueError, qp.add_constraints, x, eq, np.ones(n+1))

        # raise error if lhs is array of floats
        self.assertRaises(ValueError, qp.add_constraints, b, le, x)

    def test_optimizer_feasible(self):

        # initialize model
        qp = BoundedQP()

        # add optimization variables
        n = 15
        x = qp.add_variables(n, name='x')

        # add constraints
        lb = np.array([1.] * n)
        qp.add_constraints(x, ge, lb, name='c1')

        # set objective
        qp.setObjective(.5 * x.dot(x))

        # no solution available yet
        self.assertRaises(RuntimeError, qp.primal_optimizer, 'x')
        self.assertRaises(RuntimeError, qp.dual_optimizer, 'c1')

        # check primal optimizer
        qp.optimize()
        np.testing.assert_array_equal(qp.primal_optimizer('x'), lb)

        # check dual optimizer
        np.testing.assert_array_almost_equal(qp.dual_optimizer('c1'), [-1.]*n)

        # check optimal value`
        self.assertEqual(qp.primal_objective(), .5*lb.dot(lb))
        self.assertEqual(qp.dual_objective(), .5*lb.dot(lb))

        # check sign multipliers with opposite sense (le -> positive)
        qp.add_constraints([x[1]+2.], le, [x[0]], name='c2')
        qp.optimize()
        self.assertTrue(qp.dual_optimizer('c2') > 0.)

        # check sign multipliers with opposite sense (ge -> negative)
        qp.add_constraints([x[0]], ge, [x[1]+3.], name='c3')
        qp.optimize()
        self.assertTrue(qp.dual_optimizer('c3') < 0.)

    def test_optimize_infeasible(self):
        '''
        Consider a < 0 and b > 0.
        Primal QP: min |x|^2 s.t. x <= a, x >= b.
        Primal LP: min 0 s.t. x <= a, x >= b.
        Lagrangian: L = p'(x - a) + q'(x - b).
        Dual LP: max - a'p - b'q s.t. p + q = 0, p >= 0, q <= 0.
        Farkas proof: any vectors p = - q > 0.
        '''
        np.random.seed(1)

        # initialize model
        qp = BoundedQP()

        # add optimization variables
        n = 15
        x = qp.add_variables(n, name='x')

        # set objective
        qp.setObjective(x.dot(x))

        # add infeasible constraints
        a = - np.random.rand(n)
        b = np.random.rand(n)
        qp.add_constraints(x, le, a, name='p')
        qp.add_constraints(x, ge, b, name='q') 
        qp.optimize()

        # primal None if infeasible
        x_found = qp.primal_optimizer('x')
        self.assertIsNone(x_found)

        # check signs farkas proof
        p = qp.dual_optimizer('p')
        q = qp.dual_optimizer('q')
        self.assertTrue(max(p) > 0.)
        self.assertTrue(min(p) >= 0.)
        self.assertTrue(min(q) < 0.)
        self.assertTrue(max(q) <= 0.)
        np.testing.assert_array_equal(p, - q)

        # check optimal value
        self.assertTrue(np.isinf(qp.primal_objective()))
        obj = - a.dot(p) - b.dot(q)
        self.assertEqual(qp.dual_objective(), obj)

    def test_optimize_unbounded(self):

        # initialize model
        qp = BoundedQP()

        # add optimization variables
        n = 15
        x = qp.add_variables(n)

        # set objective
        qp.setObjective(x[:-1].dot(x[:-1]) + x[-1])

        # add constraints
        qp.add_constraints(x, le, [1.]*n)
        self.assertRaises(AssertionError, qp.optimize)

if __name__ == '__main__':
    unittest.main()