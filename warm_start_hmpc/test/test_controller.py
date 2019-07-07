# external imports
import unittest
import numpy as np
from copy import copy

# internal imports
import cart_pole_with_wall as cp
from warm_start_hmpc.controller import HybridModelPredictiveController

class TestController(unittest.TestCase):

    def test_init(self):

        # wrong size C, D, C_T
        for i, M in enumerate([np.eye(cp.mld.nx+1), np.eye(cp.mld.nu+1), np.eye(cp.mld.nx+1)]):
            objective = copy(cp.objective)
            objective[i] = M
            self.assertRaises(ValueError, HybridModelPredictiveController, cp.mld, cp.T, objective, cp.terminal_set)

        # wrong size terminal constraint
        for i, M in enumerate([np.ones((2*cp.mld.nx+1, cp.mld.nx)), np.ones(2*cp.mld.nx+1)]):
            terminal_set = copy(cp.terminal_set)
            terminal_set[i] = M
            self.assertRaises(ValueError, HybridModelPredictiveController, cp.mld, cp.T, objective, terminal_set)

    def test_update(self):
        np.random.seed(1)

        # update rho
        np.testing.assert_array_equal(cp.controller._update['rho'], cp.C_T_scaling * np.eye(cp.mld.nx))

        # update mu without terminal constraint
        F_T = np.empty((0,cp.mld.nx))
        h_T = np.empty(0)
        terminal_set = [F_T, h_T]
        controller = HybridModelPredictiveController(cp.mld, cp.T, cp.objective, terminal_set)
        np.testing.assert_array_equal(controller._update['mu'], np.eye(cp.mld.F.shape[0]))

        # update mu with terminal constraint
        self.assertTrue(np.min(cp.controller._update['mu']) >= 0)
        mu_Tm1 = np.random.rand(cp.controller.h_Tm1.size)
        mu_Tm2 = cp.controller._update['mu'].dot(mu_Tm1)
        lhs = np.vstack((cp.mld.F.T, cp.mld.G.T))
        rhs = np.vstack((cp.controller.F_Tm1.T, cp.controller.G_Tm1.T))
        np.testing.assert_array_almost_equal(lhs.dot(mu_Tm2), rhs.dot(mu_Tm1))

    def test_bound_binaries(self):
        np.random.seed(1)

        # set random identifiers (bounds on the binaries) and get them back
        for _ in range(100):
            identifier = {}
            for t in range(cp.T):
                identifier[(t, np.random.randint(0, cp.mld.nub))] = np.random.randint(0, 2)

            # set binaries and get bounds back
            cp.controller._set_bound_binaries(identifier)
            ub_lb, ub_ub = cp.controller._get_bound_binaries(identifier)

            # double check with identifier
            for t in range(cp.T):
                for i in range(cp.mld.nub):
                    if (t, i) in identifier:
                        self.assertTrue(ub_lb[t][i] == identifier[(t, i)])
                        self.assertTrue(ub_ub[t][i] == identifier[(t, i)])
                    else:
                        self.assertTrue(ub_lb[t][i] == 0.)
                        self.assertTrue(ub_ub[t][i] == 1.)

    def test_solve_subproblem(self, tol=1.e9):

        # resolve each subproblem associated with a leaf
        for i, leaf in enumerate(cp.leaves):

            # get solution quadratic program
            solution = cp.solution_leaves[i]

            # if infeasible check cost of proof of infeasibility
            if np.isinf(solution.primal.objective):
                self.assertTrue(solution.dual.objective > 0.)

            # if feasible check primal feasibility
            else:
                zero_terms, nonnegative_terms = cp.plug_in_primal_constraints(
                    solution.primal.variables,
                    leaf.identifier
                    )
                self.assertAlmostEqual(np.linalg.norm(zero_terms), 0., places=5)
                self.assertTrue(np.min(nonnegative_terms) >= -tol)

            # check dual feasibility in any case
            zero_terms, nonnegative_terms = cp.plug_in_dual_constraints(solution.dual.variables)
            self.assertAlmostEqual(np.linalg.norm(zero_terms), 0., places=6)
            self.assertTrue(np.min(nonnegative_terms) >= -tol)

    def test_feedforward(self, tol=1.e-8):
        np.random.seed(1)

        # check that the cost stored in the each leaf is a lower bound to its optimal values
        for i, leaf in enumerate(cp.leaves):
            solution = cp.solution_leaves[i]
            self.assertTrue(solution.primal.objective >= leaf.lb - tol) # equal needed if inf >= inf
            self.assertTrue(solution.primal.objective >= cp.solution.objective)

        # check that leaves cover the vertices of the hypercube and are pairwise disjoint
        self.assertTrue(cp.pairwise_disjoint_cover_of_unit_cube(cp.leaves, 100))

    def test_warm_start_implied_bounds(self):

        # check that the cost stored in the each warm start node is a lower bound to its optimal values
        for node in cp.warm_start:
            node_solution = cp.controller._solve_subproblem(node.identifier, cp.x1)
            self.assertTrue(node_solution.primal.objective >= node.lb)

    def test_warm_start_cover(self):
        np.random.seed(1)

        # check that the warm start nodes cover the vertices of the hypercube and are pairwise disjoint
        self.assertTrue(cp.pairwise_disjoint_cover_of_unit_cube(cp.warm_start, 100))

    def test_warm_start_dual_objective(self):

        # check that the implied bounds are correct
        for node in cp.warm_start:
            if node.extra.dual is not None:

                # evaluate cost of shifted duals
                objective = cp.plug_in_dual_objective(
                    node.extra.dual.variables,
                    node.identifier
                    )
                objective = max(0., objective)

                # proofs of infeasibility must have positive objective
                if np.isinf(node.lb):
                    self.assertTrue(objective > 0.)

                # otherwise check that the objectives coincide
                else:
                    self.assertAlmostEqual(objective, node.lb, places=6)

    def test_warm_start_dual_feasibility(self, tol=1.e9):

        # check that the shifted solutions are dual feasible
        for node in cp.warm_start:
            if node.extra.dual is not None:
                zero_terms, nonnegative_terms = cp.plug_in_dual_constraints(node.extra.dual.variables)
                self.assertAlmostEqual(np.linalg.norm(zero_terms), 0., places=5)
                self.assertTrue(np.min(nonnegative_terms) >= -tol)

    def test_warm_start_vs_cold_start(self):

        # solve with and without warm start
        solution_ws, _ = cp.controller.feedforward(cp.x1, printing_period=None, warm_start=cp.warm_start)
        solution, _ = cp.controller.feedforward(cp.x1, printing_period=None)
        self.assertEqual(solution_ws.objective, solution.objective)

if __name__ == '__main__':
    unittest.main()