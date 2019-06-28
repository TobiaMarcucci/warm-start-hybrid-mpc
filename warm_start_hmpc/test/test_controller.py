# external imports
import unittest
import numpy as np

# internal inputs
from warm_start_hmpc.controller import HybridModelPredictiveController

class TestController(unittest.TestCase):

    def _check_dual_solution(self, x0, identifier, solution, tol):
        '''
        Checks that the dual variables given by gurobi are correct.
        Mostly useful for debugging the signs of the multipliers.

        Parameters
        ----------
        x0 : np.array
            Initial state of the system.
        identifier : dict
            Dictionary containing the values for some of the binaries.
            Pass an empty dictionary to reset the bounds of the binaries to 0 and 1.
        solution : dict
            Dictionary with the primal and dual solution of the subproblem.
        tol : float
            Numeric tolerance in the checks.
        '''

        # dual feasibility
        self._check_dual_feasibility(solution, tol)

        # dual objective
        if self.qp.status == 2:
            assert np.isclose(self.qp.objVal, solution['objective'])
        else:
            assert solution['objective'] > tol

    def _check_dual_feasibility(self, solution, tol):
        '''
        Checks that the given dual solution is feasible.

        Parameters
        ----------
        solution : dict
            Dictionary containing the dual solution of the convex subproblem.
            Keys are 'alpha', 'beta', 'gamma', 'delta', 'lbu', 'ubu', 'lbs', 'ubs'.
            Each one of these is a list (ordered in time) of numpy arrays.
        tol : float
            Numeric tolerance in the checks.
        '''

        # check positiveness
        for c in ['mu', 'nu_lb', 'nu_ub']:
            multipliers = np.concatenate(solution[c])
            assert np.min(multipliers) > - tol

        # check stationarity wrt x_t
        for t in range(self.T):
            residuals = self.C.T.dot(solution['rho'][t]) + \
                solution['lam'][t] - \
                self.mld.A.T.dot(solution['lam'][t+1]) + \
                self.mld.F.T.dot(solution['mu'][t])
            if t == self.T-1 and self.X_T is not None:
                residuals += self.mld.A.T.dot(self.X_T[0].T.dot(solution['mu'][self.T]))
            assert np.linalg.norm(residuals) < tol

        # check stationarity wrt x_N
        residuals = self.C_T.T.dot(solution['rho'][self.T]) + solution['lam'][self.T] 
        assert np.linalg.norm(residuals) < tol

        # test stationarity wrt u_t
        for t in range(self.T):
            residuals = self.D.T.dot(solution['sigma'][t]) - \
                self.mld.B.T.dot(solution['lam'][t+1]) + \
                self.mld.G.T.dot(solution['mu'][t]) + \
                self.mld.V.T.dot(solution['nu_ub'][t] - solution['nu_lb'][t])
            if t == self.T-1 and self.X_T is not None:
                residuals += self.mld.B.T.dot(self.X_T[0].T.dot(solution['mu'][self.T]))
            assert np.linalg.norm(residuals) < tol

if __name__ == '__main__':
    unittest.main()