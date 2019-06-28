# external imports
import numpy as np

class SubproblemSolution(object):

    def __init__(self, primal, dual, integer_feasible):

        # store data
        self.primal = primal
        self.dual = dual
        self.integer_feasible = integer_feasible

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

        # organize primal and dual solutions
        primal = PrimalFeasibleSolution.from_controller(controller)
        dual = DualFeasibleSolution.from_controller(controller)

        # check if integer_feasible
        rhs = controller.qp.get_constraint_rhs
        lb = np.concatenate([rhs('nu_lb_%d'%t) for t in range(controller.T)])
        ub = np.concatenate([rhs('nu_ub_%d'%t) for t in range(controller.T)])
        integer_feasible = np.array_equal(lb, ub)

        return SubproblemSolution(primal, dual, integer_feasible)

    # def objective_lower_bound(self):

    #     # if dual has not been solved yet
    #     if self.dual is None:
    #         return - np.inf

    #     return self.dual.objective

    # def feasible(self):

    #     # if primal has not been solved yet
    #     if self.primal is None:
    #         return None

    #     return not np.isinf(self.primal.objective)

class PrimalFeasibleSolution(object):
    '''
    Dual feasible (not necessarily optimal) solution of the quadratic subproblem.
    '''

    def __init__(self, variables, objective):

        # store primal variables
        self.variables = variables
        self.x = variables['x']
        self.uc = variables['uc']
        self.ub = variables['ub']

        # store primal objective
        self.objective = objective

    @staticmethod
    def from_controller(controller):
        '''
        Extracts the optimal primal solution from the controller instance.
        The methods get_primal_optimizer and primal_objective already raise runtime erorrs if the problem has not been solved yet.
        These return None if the problem is infeasible.

        Parameters
        ----------
        controller : HybridModelPredictiveController
            MPC controller with solved quadratic subproblem.

        Returns
        -------
        PrimalFeasibleSolution
            Primal solution extracted from the controller.
        '''

        # get minimizer primal solution
        variables = {}
        opt = controller.qp.get_primal_optimizer
        variables['x'] = [opt('x_%d'%t) for t in range(controller.T+1)]
        for k in ['uc', 'ud']:
            variables[k] = [opt('%s_%d'%(k,t)) for t in range(controller.T)]

        return PrimalFeasibleSolution(variables, controller.qp.primal_objective())

class DualFeasibleSolution(object):
    '''
    Dual feasible (not necessarily optimal) solution of the quadratic subproblem.
    '''

    def __init__(self, variables, objective):

        # store dual variables
        self.variables = variables
        self.lam = variables['lam']
        self.mu = variables['mu']
        self.nu_lb = variables['nu_lb']
        self.nu_ub = variables['nu_ub']
        self.rho = variables['rho']
        self.sigma = variables['sigma']

        # store dual objective
        self.objective = objective

    @staticmethod
    def from_controller(controller):
        '''
        Extracts the optimal dual solution from the controller instance.
        The methods get_dual_optimizer and dual_objective already raise runtime erorrs if the problem has not been solved yet.
        These return the multipliers and the cost of a Farkas proof if the problem is infeasible.

        Parameters
        ----------
        controller : HybridModelPredictiveController
            MPC controller with solved quadratic subproblem.

        Returns
        -------
        DualFeasibleSolution
            Dual solution extracted from the controller.
        '''

        # shotcuts
        ctrl = controller
        popt = ctrl.qp.get_primal_optimizer
        dopt = ctrl.qp.get_dual_optimizer

        # get minimizer dual solution
        variables = {}
        variables['lam'] = [dopt('lam_%d'%t) for t in range(ctrl.T+1)]
        for k in ['mu', 'nu_lb', 'nu_lb']:
            variables[k] = [dopt('%s_%d'%(k,t)) for t in range(ctrl.T)]

        # auxiliary multipliers, if feasible
        if ctrl.qp.status = 2:

            # stage multipliers
            variables['rho'] = [2*ctrl.C.dot(popt('x_%d'%t)) for t in range(ctrl.T)]
            variables['sigma'] = [2*ctrl.D.dot(popt('u_%d'%t)) for t in range(ctrl.T)]

            # terminal multipliers
            variables['rho'].append(2*ctrl.C_T.dot(popt('x_%d'%ctrl.T)))

        # auxiliary multipliers, if infeasible
        else:

            # get Farkas proof state output
            variables['rho'] = [np.zeros(ctrl.C.shape[0]) for t in range(ctrl.T)]
            variables['rho'].append(np.zeros(ctrl.C_T.shape[0]))

            # get Farkas proof input output
            variables['sigma'] = [np.zeros(ctrl.D.shape[0]) for t in range(ctrl.T)]

        return DualFeasibleSolution(variables, ctrl.qp.dual_objective())