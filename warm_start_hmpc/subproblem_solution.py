# external imports
import numpy as np

class SubproblemSolution(object):

    def __init__(self, primal, dual, active_set=None):
        '''
        Parameters
        ----------
        primal : PrimalSolution
        dual : DualSolution
        '''

        self.primal = primal
        self.dual = dual
        self.active_set = active_set

    @staticmethod
    def from_controller(controller):
        '''
        Extracts the optimal primal and solution from the controller instance.

        Parameters
        ----------
        controller : HybridModelPredictiveController
            MPC controller with solved quadratic subproblem.

        Returns
        -------
        SubproblemSolution
            Primal-dual solution extracted from the controller.
        '''

        primal = PrimalSolution.from_controller(controller)
        dual = DualSolution.from_controller(controller)

        # active set
        if controller.qp.Params.Method == 1:
            active_set = {}
            active_set['c'] = [c.getAttr('CBasis') for c in controller.qp.getConstrs()]
            active_set['v'] = [v.getAttr('VBasis') for v in controller.qp.getVars()]
        else:
            active_set = None

        return SubproblemSolution(primal, dual, active_set)

class PrimalSolution(object):
    '''
    Primal feasible (not necessarily optimal) solution of the quadratic subproblem.
    '''

    def __init__(self, variables, objective, binary_feasible):
        '''
        Parameters
        ----------
        variables : dict
            Feasible optimization variables of the primal problem.
        objective : float
            Cost of the feasible solution.
        binary_feasible : bool
            True if the passed solution is binary feasible, False otherwise.
        '''

        self.variables = variables
        self.objective = objective
        self.binary_feasible = binary_feasible

    @staticmethod
    def from_controller(controller):
        '''
        Extracts the optimal primal solution from the controller instance.
        The methods primal_optimizer and primal_objective already raise runtime erorrs if the problem has not been solved yet.
        These return None if the problem is infeasible.

        Parameters
        ----------
        controller : HybridModelPredictiveController
            MPC controller with solved quadratic subproblem.

        Returns
        -------
        PrimalSolution
            Primal solution extracted from the controller.
        '''

        # get minimizer primal solution
        variables = {}
        opt = controller.qp.primal_optimizer
        variables['x'] = [opt(f'x_{t}') for t in range(controller.T+1)]
        for k in ['uc', 'ub']:
            variables[k] = [opt(f'{k}_{t}') for t in range(controller.T)]

        # check if binary feasible
        rhs = controller.qp.get_constraint_rhs
        lb = np.concatenate([rhs(f'nu_lb_{t}') for t in range(controller.T)])
        ub = np.concatenate([rhs(f'nu_ub_{t}') for t in range(controller.T)])
        binary_feasible = np.array_equal(lb, -ub)

        return PrimalSolution(variables, controller.qp.primal_objective(), binary_feasible)

class DualSolution(object):
    '''
    Dual feasible (not necessarily optimal) solution of the quadratic subproblem.
    '''

    def __init__(self, variables, objective):
        '''
        Parameters
        ----------
        variables : dict
            Feasible optimization variables of the dual problem.
        objective : float
            Cost of the feasible solution if feasible, cost of the Farkas' if infeasible.
        '''
        
        self.variables = variables
        self.objective = objective

    @staticmethod
    def from_controller(controller):
        '''
        Extracts the optimal dual solution from the controller instance.
        The methods dual_optimizer and dual_objective already raise runtime erorrs if the problem has not been solved yet.
        These return the multipliers and the cost of a Farkas proof if the problem is infeasible.

        Parameters
        ----------
        controller : HybridModelPredictiveController
            MPC controller with solved quadratic subproblem.

        Returns
        -------
        DualSolution
            Dual solution extracted from the controller.
        '''

        # shotcuts
        ctrl = controller
        popt = ctrl.qp.primal_optimizer
        dopt = ctrl.qp.dual_optimizer

        # get minimizer dual solution
        variables = {}
        variables['lam'] = [dopt(f'lam_{t}') for t in range(ctrl.T+1)]
        for k in ['mu', 'nu_lb', 'nu_ub']:
            variables[k] = [dopt(f'{k}_{t}') for t in range(ctrl.T)]

        # auxiliary multipliers, if primal feasible
        if ctrl.qp.status == 2:

            # stage multipliers
            variables['rho'] = [2*ctrl.Q.dot(popt(f'x_{t}')) for t in range(ctrl.T)]
            variables['sigma'] = [2*ctrl.R.dot(np.concatenate((popt(f'uc_{t}'),popt(f'ub_{t}')))) for t in range(ctrl.T)]

            # terminal multipliers
            variables['rho'].append(2*ctrl.Q_T.dot(popt(f'x_{ctrl.T}')))

        # auxiliary multipliers, if primal infeasible
        else:

            # get Farkas proof state output
            variables['rho'] = [np.zeros(ctrl.Q.shape[0]) for t in range(ctrl.T)]
            variables['rho'].append(np.zeros(ctrl.Q_T.shape[0]))

            # get Farkas proof input output
            variables['sigma'] = [np.zeros(ctrl.R.shape[0]) for t in range(ctrl.T)]

        return DualSolution(variables, ctrl.qp.dual_objective())
