# external imports
import unittest
import numpy as np
from copy import copy

# internal inputs
from warm_start_hmpc.controller import HybridModelPredictiveController
from warm_start_hmpc.mld_system import MLDSystem

class TestController(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        
        # system dimensions
        nx = 2
        nuc = 1

        # generate A and B ensuring controllability
        controllable = False
        while not controllable:
            A = np.random.rand(nx, nx)
            Bc = np.random.rand(nx, nuc)
            R = np.hstack([np.linalg.matrix_power(A, i).dot(Bc) for i in range(nx)])
            controllable = np.linalg.matrix_rank(R) == nx

        # add binary inputs
        Bb = np.hstack((np.eye(nx), -np.eye(nx)))/10.
        nub = Bb.shape[1]
        nu = nuc + nub
        B = np.hstack((Bc, Bb))

        # box constraints
        x_max = np.ones(nx)
        u_max = np.concatenate((np.ones(nuc)/10., np.ones(nub)))
        np.ones(nu)

        # bring in matrix form
        F = np.vstack((
            np.eye(nx),
            -np.eye(nx),
            np.zeros((2*nu, nx))
            ))
        G = np.vstack((
            np.zeros((2*nx, nu)),
            np.eye(nu),
            -np.eye(nu)
            ))
        h = np.concatenate((
            x_max,
            x_max,
            u_max,
            u_max
            ))

        # generate MLD system
        dynamics = [A, B]
        constraints = [F, G, h]
        self.mld = MLDSystem(dynamics, constraints, nub)

        # objective
        C = np.eye(nx)
        self.C_T_scale = 10.
        C_T = np.eye(nx) * self.C_T_scale
        D = np.hstack((np.eye(nuc), np.zeros((nuc, nub))))
        self.objective = [C, D, C_T]

        # terminal constraint
        F_T = np.vstack((np.eye(nx), -np.eye(nx)))
        h_T = np.concatenate((x_max/10., x_max/10.))
        self.terminal_set = [F_T, h_T]

        # hybrid controller
        self.T = 5
        self.controller = HybridModelPredictiveController(self.mld, self.T, self.objective, self.terminal_set)

    def test_init(self):

        # wrong size C, D, C_T
        for i, M in enumerate([np.eye(self.mld.nx+1), np.eye(self.mld.nu+1), np.eye(self.mld.nx+1)]):
            objective = copy(self.objective)
            objective[i] = M
            self.assertRaises(ValueError, HybridModelPredictiveController, self.mld, self.T, objective, self.terminal_set)

        # wrong size terminal constraint
        for i, M in enumerate([np.ones((2*self.mld.nx+1, self.mld.nx)), np.ones(2*self.mld.nx+1)]):
            terminal_set = copy(self.terminal_set)
            terminal_set[i] = M
            self.assertRaises(ValueError, HybridModelPredictiveController, self.mld, self.T, objective, terminal_set)

    def test_update(self):
        np.random.seed(1)

        # update rho
        np.testing.assert_array_equal(self.controller._update['rho'], self.C_T_scale * np.eye(self.mld.nx))

        # update mu without terminal constraint
        F_T = np.empty((0,self.mld.nx))
        h_T = np.empty(0)
        terminal_set = [F_T, h_T]
        controller = HybridModelPredictiveController(self.mld, self.T, self.objective, terminal_set)
        np.testing.assert_array_equal(controller._update['mu'], np.eye(self.mld.F.shape[0]))

        # update mu with terminal constraint
        self.assertTrue(np.min(self.controller._update['mu']) >= 0)
        mu_Tm1 = np.random.rand(self.controller.h_Tm1.size)
        mu_Tm2 = self.controller._update['mu'].dot(mu_Tm1)
        lhs = np.vstack((self.mld.F.T, self.mld.G.T))
        rhs = np.vstack((self.controller.F_Tm1.T, self.controller.G_Tm1.T))
        np.testing.assert_array_almost_equal(lhs.dot(mu_Tm2), rhs.dot(mu_Tm1))

    def test_bound_binaries(self):
        np.random.seed(1)

        # run some random tests
        for _ in range(10):

            # identifier for the subproblmem
            identifier = {}
            for t in range(self.T):
                identifier[(t, np.random.randint(0, self.mld.nub))] = np.random.randint(0, 2)

            # set binaries and get bounds back
            self.controller._set_bound_binaries(identifier)
            ub_lb, ub_ub = self.controller._get_bound_binaries(identifier)

            # double check with identifier
            for t in range(self.T):
                for i in range(self.mld.nub):
                    if (t, i) in identifier:
                        self.assertTrue(ub_lb[t][i] == identifier[(t, i)])
                        self.assertTrue(ub_ub[t][i] == identifier[(t, i)])
                    else:
                        self.assertTrue(ub_lb[t][i] == 0.)
                        self.assertTrue(ub_ub[t][i] == 1.)

    def test_solve_subproblem(self):
        np.random.seed(1)

        # run some random tests
        for _ in range(100):

            # identifier for the subproblmem
            identifier = {}
            for t in range(self.T):
                identifier[(t, np.random.randint(0, self.mld.nub))] = np.random.randint(0, 2)

            # solve quadratic program
            x0 = np.random.rand(self.mld.nx)*1.5
            solution = self.controller._solve_subproblem(x0, identifier)

            # if feasible
            if not np.isinf(solution.primal.objective):

                # rename primal varibales
                x = solution.primal.variables['x']
                uc = solution.primal.variables['uc']
                ub = solution.primal.variables['ub']
                u = [np.concatenate((uc[t], ub[t])) for t in range(self.T)]

                # rename dual varibales
                rho = solution.dual.variables['rho']
                lam = solution.dual.variables['lam']
                sigma = solution.dual.variables['sigma']
                mu = solution.dual.variables['mu']
                nu_lb = solution.dual.variables['nu_lb']
                nu_ub = solution.dual.variables['nu_ub']


                # get bounds for this identifier
                ub_lb, ub_ub = self.controller._get_bound_binaries(identifier)

                # initial state constraint
                np.testing.assert_array_equal(x0, solution.primal.variables['x'][0])

                # MLD dynamics
                x_next = x0
                for t in range(self.T):
                    x_next = self.mld.A.dot(x_next) + self.mld.B.dot(u[t])
                    np.testing.assert_array_almost_equal(x_next, solution.primal.variables['x'][t+1])

                # MLD constraints
                for t in range(self.T):
                    residuals = self.mld.F.dot(x[t]) + self.mld.G.dot(u[t]) - self.mld.h
                    self.assertTrue(np.max(residuals) <= 0.)

                # binaries constraint
                for t in range(self.T):
                    self.assertTrue(np.min(ub[t] - ub_lb[t]) >= 0.)
                    self.assertTrue(np.min(ub_ub[t] - ub[t]) >= 0.)

                # terminal constraints
                residuals = self.terminal_set[0].dot(x[self.T]) - self.terminal_set[1]
                self.assertTrue(np.max(residuals) <= 0.)

                # dual terminal conditions
                residuals = self.controller.C_T.T.dot(rho[self.T]) + lam[self.T]
                self.assertAlmostEqual(np.linalg.norm(residuals), 0.)

                # dual dynamics time T-1
                residuals = self.controller.C.T.dot(rho[self.T-1]) + lam[self.T-1] \
                            - self.mld.A.T.dot(lam[self.T]) \
                            + self.controller.F_Tm1.T.dot(mu[self.T-1])
                self.assertAlmostEqual(np.linalg.norm(residuals), 0.)

                # dual dynamics at time t
                for t in range(self.T-1):
                    residuals = self.controller.C.T.dot(rho[t]) + lam[t] \
                                - self.mld.A.T.dot(lam[t+1]) \
                                + self.mld.F.T.dot(mu[t])
                    self.assertAlmostEqual(np.linalg.norm(residuals), 0.)

                # dual constraints at time T-1
                residuals = self.controller.D.T.dot(sigma[self.T-1]) \
                            - self.mld.B.T.dot(lam[self.T]) \
                            + self.controller.G_Tm1.T.dot(mu[self.T-1]) \
                            + self.mld.V.T.dot(nu_ub[self.T-1] - nu_lb[self.T-1])
                self.assertAlmostEqual(np.linalg.norm(residuals), 0.)

                # dual cosntraints at time t
                for t in range(self.T-1):
                    residuals = self.controller.D.T.dot(sigma[t]) \
                                - self.mld.B.T.dot(lam[t+1]) \
                                + self.mld.G.T.dot(mu[t]) \
                                + self.mld.V.T.dot(nu_ub[t] - nu_lb[t])

                # nonnegativity
                for m in mu, nu_lb, nu_ub:
                    self.assertTrue(np.min(np.concatenate(m)) >= 0.)

    def test_feedforward(self, tol=1.e-8):
        np.random.seed(1)

        # solve OCP in open loop
        x0 = np.random.rand(self.mld.nx)
        # solution, leaves = self.controller.feedforward(x0, draw_label='test')
        solution, leaves = self.controller.feedforward(x0, printing_period=None)

        # check that the cost stored in the each leaf is a lower bound to its optimal values
        for leaf in leaves:
            solution_leaf = self.controller._solve_subproblem(x0, leaf.identifier)
            self.assertTrue(solution_leaf.primal.objective > leaf.lb - tol)
            self.assertTrue(solution_leaf.primal.objective >= solution.objective)

        # check that leaves cover the vertices of the hypercube and are pairwise disjoint
        for _ in range(100):
            ub = np.random.randint(0, 2, self.mld.nub*self.T)
            included = 0
            for leaf in leaves:
                ub_lb, ub_ub = self.controller._get_bound_binaries(leaf.identifier)
                ub_lb = np.concatenate(ub_lb)
                ub_ub = np.concatenate(ub_ub)
                if np.min(ub-ub_lb) >= 0. and np.min(ub_ub-ub) >= 0.:
                    included += 1
            self.assertTrue(included == 1)

    def test_war_start(self):
        np.random.seed(1)

        # solve OCP in open loop
        x0 = np.random.rand(self.mld.nx)
        solution, leaves = self.controller.feedforward(x0, printing_period=None)

        # injected inputs and modeling error
        uc0 = solution.variables['uc'][0]
        ub0 = solution.variables['ub'][0]
        u0 = np.concatenate((uc0, ub0))
        e0 = np.random.randn(self.mld.nx)/100.
        x1 = self.mld.A.dot(x0) + self.mld.B.dot(u0) + e0

        # get warm start root nodes
        warm_start = self.controller.construct_warm_start(leaves, x0, uc0, ub0, e0)

        # solve without warm start
        solution, leaves = self.controller.feedforward(x1, printing_period=None)

        # check that the cost stored in the each warm start node is a lower bound to its optimal values
        for node in warm_start:
            solution_node = self.controller._solve_subproblem(x1, node.identifier)
            self.assertTrue(solution_node.primal.objective >= node.lb)

        # check that the warm start nodes cover the vertices of the hypercube and are pairwise disjoint
        for _ in range(100):
            ub = np.random.randint(0, 2, self.mld.nub*self.T)
            included = 0
            for node in warm_start:
                ub_lb, ub_ub = self.controller._get_bound_binaries(node.identifier)
                ub_lb = np.concatenate(ub_lb)
                ub_ub = np.concatenate(ub_ub)
                if np.min(ub-ub_lb) >= 0. and np.min(ub_ub-ub) >= 0.:
                    included += 1
            self.assertTrue(included == 1)

        # solve with warm start
        solution_ws, leaves_ws = self.controller.feedforward(x1, printing_period=None, warm_start=warm_start)
        self.assertEqual(solution_ws.objective, solution.objective)

if __name__ == '__main__':
    unittest.main()