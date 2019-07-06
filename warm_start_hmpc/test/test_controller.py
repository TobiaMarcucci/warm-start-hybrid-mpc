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

    def test_solve_subproblem(self, tol=1.e-6):
        np.random.seed(1)

        # resolve each subproblem associated with a leaf
        for i, leaf in enumerate(cp.leaves):

            # get solution quadratic program
            solution = cp.solution_leaves[i]

            # if infeasible check cost of proof of infeasibility
            if np.isinf(solution.primal.objective):
                self.assertTrue(solution.dual.objective > 0.)


            # if feasible check primal feasibility
            else:

                # rename primal varibales
                x = solution.primal.variables['x']
                uc = solution.primal.variables['uc']
                ub = solution.primal.variables['ub']
                u = [np.concatenate((uc[t], ub[t])) for t in range(cp.T)]

                # get bounds for this identifier
                ub_lb, ub_ub = cp.controller._get_bound_binaries(leaf.identifier)

                # initial state constraint
                np.testing.assert_array_equal(cp.x0, solution.primal.variables['x'][0])

                # MLD dynamics
                x_next = cp.x0
                for t in range(cp.T):
                    x_next = cp.mld.A.dot(x_next) + cp.mld.B.dot(u[t])
                    np.testing.assert_array_almost_equal(
                        x_next,
                        solution.primal.variables['x'][t+1],
                        decimal=5
                        )

                # MLD constraints
                for t in range(cp.T):
                    residuals = cp.mld.F.dot(x[t]) + cp.mld.G.dot(u[t]) - cp.mld.h
                    self.assertTrue(np.max(residuals) <= tol)

                # binaries constraint
                for t in range(cp.T):
                    self.assertTrue(np.min(ub[t] - ub_lb[t]) >= 0.)
                    self.assertTrue(np.min(ub_ub[t] - ub[t]) >= 0.)

                # terminal constraints
                residuals = cp.terminal_set[0].dot(x[cp.T]) - cp.terminal_set[1]
                self.assertTrue(np.max(residuals) <= tol)

            # check dual feasibility in any case

            # rename dual varibales
            rho = solution.dual.variables['rho']
            lam = solution.dual.variables['lam']
            sigma = solution.dual.variables['sigma']
            mu = solution.dual.variables['mu']
            nu_lb = solution.dual.variables['nu_lb']
            nu_ub = solution.dual.variables['nu_ub']

            # dual terminal conditions
            residuals = cp.controller.C_T.T.dot(rho[cp.T]) + lam[cp.T]
            self.assertAlmostEqual(np.linalg.norm(residuals), 0.)

            # dual dynamics time T-1
            residuals = cp.controller.C.T.dot(rho[cp.T-1]) + lam[cp.T-1] \
                        - cp.mld.A.T.dot(lam[cp.T]) \
                        + cp.controller.F_Tm1.T.dot(mu[cp.T-1])
            self.assertAlmostEqual(np.linalg.norm(residuals), 0.)

            # dual dynamics at time t
            for t in range(cp.T-1):
                residuals = cp.controller.C.T.dot(rho[t]) + lam[t] \
                            - cp.mld.A.T.dot(lam[t+1]) \
                            + cp.mld.F.T.dot(mu[t])
                self.assertAlmostEqual(np.linalg.norm(residuals), 0., places=6)

            # dual constraints at time T-1
            residuals = cp.controller.D.T.dot(sigma[cp.T-1]) \
                        - cp.mld.B.T.dot(lam[cp.T]) \
                        + cp.controller.G_Tm1.T.dot(mu[cp.T-1]) \
                        + cp.mld.V.T.dot(nu_ub[cp.T-1] - nu_lb[cp.T-1])
            self.assertAlmostEqual(np.linalg.norm(residuals), 0.)

            # dual cosntraints at time t
            for t in range(cp.T-1):
                residuals = cp.controller.D.T.dot(sigma[t]) \
                            - cp.mld.B.T.dot(lam[t+1]) \
                            + cp.mld.G.T.dot(mu[t]) \
                            + cp.mld.V.T.dot(nu_ub[t] - nu_lb[t])
                self.assertAlmostEqual(np.linalg.norm(residuals), 0., places=5)

            # nonnegativity
            for m in mu, nu_lb, nu_ub:
                self.assertTrue(np.min(np.concatenate(m)) >= -tol)

    def test_feedforward(self, tol=1.e-8):
        np.random.seed(1)

        # check that the cost stored in the each leaf is a lower bound to its optimal values
        for i, leaf in enumerate(cp.leaves):
            solution = cp.solution_leaves[i]
            self.assertTrue(solution.primal.objective >= leaf.lb - tol) # equal needed if inf >= inf
            self.assertTrue(solution.primal.objective >= cp.solution.objective)

        # check that leaves cover the vertices of the hypercube and are pairwise disjoint
        for _ in range(100):
            ub = np.random.randint(0, 2, cp.mld.nub*cp.T)
            included = 0
            for leaf in cp.leaves:
                ub_lb, ub_ub = cp.controller._get_bound_binaries(leaf.identifier)
                ub_lb = np.concatenate(ub_lb)
                ub_ub = np.concatenate(ub_ub)
                if np.min(ub-ub_lb) >= 0. and np.min(ub_ub-ub) >= 0.:
                    included += 1
            self.assertTrue(included == 1)

    def test_warm_start(self):
        np.random.seed(1)

        # injected inputs and modeling error
        uc0 = cp.solution.variables['uc'][0]
        ub0 = cp.solution.variables['ub'][0]
        u0 = np.concatenate((uc0, ub0))
        e0 = np.random.randn(cp.mld.nx)*.01
        x1 = cp.mld.A.dot(cp.x0) + cp.mld.B.dot(u0) + e0

        # get warm start root nodes
        warm_start = cp.controller.construct_warm_start(cp.leaves, cp.x0, uc0, ub0, e0)

        # check that the cost stored in the each warm start node is a lower bound to its optimal values
        for node in warm_start:
            node_solution = cp.controller._solve_subproblem(node.identifier, x1)
            self.assertTrue(node_solution.primal.objective >= node.lb)

        # check that the warm start nodes cover the vertices of the hypercube and are pairwise disjoint
        for _ in range(100):
            ub = np.random.randint(0, 2, cp.mld.nub*cp.T)
            included = 0
            for node in warm_start:
                ub_lb, ub_ub = cp.controller._get_bound_binaries(node.identifier)
                ub_lb = np.concatenate(ub_lb)
                ub_ub = np.concatenate(ub_ub)
                if np.min(ub-ub_lb) >= 0. and np.min(ub_ub-ub) >= 0.:
                    included += 1
            self.assertTrue(included == 1)

        # check that the implied bounds are correct
        for node in warm_start:
            if node.extra.dual is not None:
                variables = node.extra.dual.variables

                # evaluate quadratic terms
                objective = 0.
                for k in ['rho', 'sigma']:
                    objective -= sum(np.linalg.norm(vt)**2 for vt in variables[k]) / 4.

                # cost initial conditions
                objective -= variables['lam'][0].dot(x1)

                # cost bounds on binaries
                ub_lb, ub_ub = cp.controller._get_bound_binaries(node.identifier)
                objective += sum(ub_lb[t].dot(vt) for t, vt in enumerate(variables['nu_lb']))
                objective -= sum(ub_ub[t].dot(vt) for t, vt in enumerate(variables['nu_ub']))

                # cost mld inequalities
                objective -= sum(cp.mld.h.dot(vt) for vt in variables['mu'][:-1])
                objective -= cp.controller.h_Tm1.dot(variables['mu'][-1])

                # saturate at zero
                objective = max(0., objective)

                # proofs of infeasibility must have positive objective
                if np.isinf(node.lb):
                    self.assertTrue(objective > 0.)

                # otherwise check that the objectives coincide
                else:
                    self.assertAlmostEqual(objective, node.lb, places=6)

        # solve with and without warm start
        solution_ws, _ = cp.controller.feedforward(x1, printing_period=None, warm_start=warm_start)
        solution, _ = cp.controller.feedforward(x1, printing_period=None)
        self.assertEqual(solution_ws.objective, solution.objective)

if __name__ == '__main__':
    unittest.main()