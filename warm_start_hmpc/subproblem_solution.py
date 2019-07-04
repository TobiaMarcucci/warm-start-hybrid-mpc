# external imports
import numpy as np

class SubproblemSolution(object):

    def __init__(self, primal, dual):
        '''
        Parameters
        ----------
        primal : PrimalSolution
        dual : DualSolution
        '''

        self.primal = primal
        self.dual = dual

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

        return SubproblemSolution(primal, dual)

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
        variables['x'] = [opt('x_%d'%t) for t in range(controller.T+1)]
        for k in ['uc', 'ub']:
            variables[k] = [opt('%s_%d'%(k,t)) for t in range(controller.T)]

        # check if binary feasible
        rhs = controller.qp.get_constraint_rhs
        lb = np.concatenate([rhs('nu_lb_%d'%t) for t in range(controller.T)])
        ub = np.concatenate([rhs('nu_ub_%d'%t) for t in range(controller.T)])
        binary_feasible = np.array_equal(lb, -ub)

        return PrimalSolution(variables, controller.qp.primal_objective(), binary_feasible)

    # @staticmethod
    # def infeasible(T):

    #     # store primal variables
    #     variables = {}
    #     variables['x'] = [None for t in range(T)]
    #     variables['uc'] = [None for t in range(T-1)]
    #     variables['ud'] = [None for t in range(T-1)]

    #     # store primal objective
    #     objective = np.inf

    #     return PrimalSolution(variables, objective)


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
        variables['lam'] = [dopt('lam_%d'%t) for t in range(ctrl.T+1)]
        for k in ['mu', 'nu_lb', 'nu_ub']:
            variables[k] = [dopt('%s_%d'%(k,t)) for t in range(ctrl.T)]

        # auxiliary multipliers, if primal feasible
        if ctrl.qp.status == 2:

            # stage multipliers
            variables['rho'] = [2*ctrl.C.dot(popt('x_%d'%t)) for t in range(ctrl.T)]
            variables['sigma'] = [2*ctrl.D.dot(np.concatenate((popt('uc_%d'%t),popt('ub_%d'%t)))) for t in range(ctrl.T)]

            # terminal multipliers
            variables['rho'].append(2*ctrl.C_T.dot(popt('x_%d'%ctrl.T)))

        # auxiliary multipliers, if primal infeasible
        else:

            # get Farkas proof state output
            variables['rho'] = [np.zeros(ctrl.C.shape[0]) for t in range(ctrl.T)]
            variables['rho'].append(np.zeros(ctrl.C_T.shape[0]))

            # get Farkas proof input output
            variables['sigma'] = [np.zeros(ctrl.D.shape[0]) for t in range(ctrl.T)]

        return DualSolution(variables, ctrl.qp.dual_objective())

    # @staticmethod
    # def unbounded():
    #     pass