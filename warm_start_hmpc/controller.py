# external imports
import numpy as np
import gurobipy as grb
from copy import copy, deepcopy
from operator import le, ge, eq

# internal inputs
from warm_start_hmpc.bounded_qp import BoundedQP
from warm_start_hmpc.subproblem_solution import SubproblemSolution, PrimalSolution, DualSolution
from warm_start_hmpc.branch_and_bound import Node, branch_and_bound, best_first, depth_first

class HybridModelPredictiveController(object):
    '''
    Optimal controller for Mixed Logical Dynamical (mld) systems.
    Solves the mixed-integer quadratic optimization problem:
    min | C_T x_T |^2 + sum_{t=0}^{T-1} | C x_t |^2 + | D u_t |^2
    s.t. x_0 given
         x_{t+1} = A x_t + B u_t, t = 0, 1, ..., T-1,
         F x_t + G u_t <= h,      t = 0, 1, ..., T-1,
         F_T x_T <= h_T,
         ub_t binary,             t = 0, 1, ..., T-1,
    '''

    def __init__(self, mld, T, objective, terminal_set):
        '''
        Instantiates the hybrid MPC controller.

        Parameters
        ----------
        mld : instance of MLDSystem
            System to be controlled.
        T : int
            Horizon of the controller.
        objective : list of np.array
            Weight matrices in the following order: C, D, C_T.
        terminal_set : list of np.array or None
            Terminal constraint in the form F_T x_T <= h_T.
            The first element of the list must be F_T, the second h_t.
            If no terminal constraint is required set this to None.
        '''

        # store inputs
        self.mld = mld
        self.T = T
        [self.C, self.D, self.C_T] = objective

        # terminal constraint
        if terminal_set is None:
            terminal_set = [np.empty((0, self.mld.x)), np.empty(0)]
        self.F_Tm1 = np.vstack((mld.F, terminal_set[0].dot(mld.A)))
        self.G_Tm1 = np.vstack((mld.G, terminal_set[0].dot(mld.B)))
        self.h_Tm1 = np.concatenate((mld.h, terminal_set[1]))

        # build mixed integer program 
        self._check_input_sizes()
        self.qp = self._build_mip()

        # warm start construction
        self.update = {
            'mu': self._update_mu(),
            'rho': np.linalg.pinv(self.C.T).dot(self.C_T.T)
        }

    def _check_input_sizes(self):
        '''
        Checks that the matrices passed as inputs in the initialization of the class have the right properties.
        '''

        # weight matrices
        assert self.C.shape[1] == self.mld.nx
        assert self.D.shape[1] == self.mld.nu
        assert self.C_T.shape[1] == self.mld.nx

        # terminal constraint
        assert self.F_Tm1.shape[0] == self.h_Tm1.size
        assert self.G_Tm1.shape[0] == self.h_Tm1.size

    def _build_mip(self):
        '''
        Builds the guorbi qp for the opitmization problem to be solved.

        Returns
        -------
        qp : BoundedQP
            Gurobi qp of the mathematical program.
        '''

        # initialize program
        qp = BoundedQP()
        obj = 0.

        # initial state (initialized to zero)
        x_next = qp.add_variables(self.mld.nx, name='x_0')
        qp.add_constraints(x_next, eq, [0.]*self.mld.nx, name='lam_0')

        # loop over the time horizon
        for t in range(self.T):

            # stage variables
            x = x_next
            x_next = qp.add_variables(self.mld.nx, name='x_%d'%(t+1))
            uc = qp.add_variables(self.mld.nuc, name='uc_%d'%t)
            ub = qp.add_variables(self.mld.nub, name='ub_%d'%t)
            u = np.concatenate((uc, ub))
            
            # bounds on the binaries
            # inequalities must be stated as expr <= num to get negative duals
            # note that num <= expr would be modified to expr => num
            # and would give positive duals
            qp.add_constraints(-ub, le, [0.]*self.mld.nub, name='nu_lb_%d'%t)
            qp.add_constraints(ub, le, [1.]*self.mld.nub, name='nu_ub_%d'%t)

            # mld dynamics
            qp.add_constraints(
                x_next,
                eq,
                self.mld.A.dot(x) + self.mld.B.dot(u),
                name='lam_%d'%(t+1)
                )

            # mld stage constraints
            if t < self.T-1:
                qp.add_constraints(
                    self.mld.F.dot(x) + self.mld.G.dot(u),
                    le,
                    self.mld.h,
                    name='mu_%d'%t
                    )
            
            # mld constraint + terminal constraint
            else:
                qp.add_constraints(
                    self.F_Tm1.dot(x) + self.G_Tm1.dot(u),
                    le,
                    self.h_Tm1,
                    name='mu_%d'%t
                    )

            # stage cost
            Cx = self.C.dot(x)
            Du = self.D.dot(u)
            obj += Cx.dot(Cx) + Du.dot(Du)

        # terminal cost
        CxT = self.C_T.dot(x_next)
        obj += CxT.dot(CxT)

        # set cost
        qp.setObjective(obj)

        return qp

    def _update_mu(self):

        # if no terminal set
        n = self.mld.h.size
        m = self.h_Tm1.size

        # initialize LP, optimization variables, and objective
        lp = BoundedQP()
        mu = lp.add_variables(n, name='mu')
        obj = lp.setObjective(self.mld.h.dot(mu))
        update_mu = []

        # initialize constraints
        lp.add_constraints(self.mld.F.T.dot(mu), eq, [0.]*self.mld.nx, name='x')
        lp.add_constraints(self.mld.G.T.dot(mu), eq, [0.]*self.mld.nu, name='u')
        lp.add_constraints(mu, ge, [0.]*n)

        # loop thorugh the columns of the rhs
        for i in range(m):

            # set rhs constraints
            lp.set_constraint_rhs('x', self.F_Tm1[i])
            lp.set_constraint_rhs('u', self.G_Tm1[i])

            # get columnn of M
            lp.optimize()
            update_mu.append(lp.primal_optimizer('mu'))

        return np.vstack(update_mu).T

    def _solve_subproblem(self, x0, identifier):
        '''
        Solves the QP relaxation uniquely indentified by its identifier for the given initial state.

        Parameters
        ----------
        x0 : np.array
            Initial state of the system.
        identifier : dict
            Dictionary containing the values for some of the binaries.
            Pass an empty dictionary to reset the bounds of the binaries to 0 and 1.

        Returns
        -------
        solution : SubproblemSolution
            Instace of the class which contains the solution of a subproblem.
        '''

        # set initial conditions
        self.qp.set_constraint_rhs('lam_0', x0)
        self._set_bounds_binaries(identifier)

        # run the optimization and initialize the result
        self.qp.optimize()

        return SubproblemSolution.from_controller(self)

    def _set_bounds_binaries(self, identifier):
        '''
        Sets the lower and upper bounds of the binary optimization variables
        in the problem to the values passed in the identifier.
        An identifier is a dictionary with tuples as keys.
        A key is in the form, e.g., (22, 4) where:
        - 22 is the time step,
        - 4 denotes the 4th element of the vector.

        Parameters
        ----------
        identifier : dict
            Dictionary containing the values for some of the binaries.
            Pass an empty dictionary to reset the bounds of the binaries to 0 and 1.
        '''

        # loop over the time horizon
        for t in range(self.T):

            # construct bounds
            lb_t = [-identifier[(t, i)] if (t, i) in identifier else 0. for i in range(self.mld.nub)]
            ub_t = [ identifier[(t, i)] if (t, i) in identifier else 1. for i in range(self.mld.nub)]

            # set bounds
            self.qp.set_constraint_rhs('nu_lb_%d'%t, lb_t)
            self.qp.set_constraint_rhs('nu_ub_%d'%t, ub_t)

    def feedforward(self, x0, **kwargs):
        '''
        Solves the mixed integer program using the branch_and_bound function.

        Parameters
        ----------
        x0 : np.array
            Initial state of the system.

        Returns
        -------
        sol : dict
            Solution associated with the incumbent node at optimality.
            (See documentation of the method _solve_subproblem.)
            None if problem is infeasible.
        optimal_leaves : list of instances of Node
            Leaves of the branch and bound tree that proved optimality of the returned solution. 
        '''

        # generate a solver for the branch and bound algorithm
        def solver(identifier):
            solution = self._solve_subproblem(x0, identifier)
            return solution.integer_feasible, solution.primal.objective, solution

        # solve the mixed integer program using branch and bound
        incumbent, leaves = branch_and_bound(
            solver,
            best_first,
            self.branch_in_chronological_order,
            **kwargs
        )
        uc = incumbent.additional.primal.variables['uc']
        ub = incumbent.additional.primal.variables['ub']
        x = incumbent.additional.primal.variables['x']
        objective = incumbent.additional.primal.objective

        return uc, ub, x, objective, leaves

    def branch_in_chronological_order(self, identifier):
        '''
        Heuristic search for the branch and bound algorithm.

        Parameters
        ----------
        identifier : dict
            Dictionary containing the values for some of the binaries.
            Pass an empty dictionary to reset the bounds of the binaries to 0 and 1.

        Returns
        -------
        branches : list of dict
            List of sub-identifier that, if merged with the identifier of the parent, give the identifier of the children.
        '''

        # idices of the last binary fixed in time
        t = max([k[0] for k in identifier.keys()] + [0])
        index = max([k[1]+1 for k in identifier.keys() if k[0] == t] + [0])

        # try to fix one more ub at time t
        if index < self.mld.nub:
            branches = [{(t,index): 0.}, {(t,index): 1.}]

        # if everything is fixed at time t, move to time t+1
        else:
            branches = [{(t+1,0): 0.}, {(t+1,0): 1.}]

        return branches

    def _shift_dual_solution(self, variables, shifted_identifier, x1):

        # initialize shifted dual variables
        shifted_variables = {}

        # shift backwards by one and append zero
        for k in ['lam', 'nu_lb', 'nu_ub', 'sigma']:
            shifted_variables[k] = [v for v in variables[k][1:]]
            shifted_variables[k].append(0. * variables[k][-1])

        # shift backwards by one, append optimal update and zero
        for k in ['mu', 'rho']:
            shifted_variables[k] = [v for v in variables[k][1:-1]]
            shifted_variables[k].append(self.update[k].dot(variables[k][-1]))
            shifted_variables[k].append(0. * variables[k][-1])

        # new dual objective
        shifted_objective = self._evaluate_dual_solution(shifted_variables, shifted_identifier, x1)

        return DualSolution(shifted_variables, shifted_objective)

    def _evaluate_dual_solution(self, variables, identifier, x0):
        '''
        Given a dual solution, returns it cost.

        Parameters
        ----------
        variables : dict
            Dictionary containing the dual solution of the subproblem.
            Each one of these is a list (ordered in time) of numpy arrays.
        identifier : dict
            Dictionary containing the values for some of the binaries.
        x0 : np.array
            Initial state of the system.

        Returns
        -------
        objective : float
            Dual objective associated with the given dual solution.
        '''

        # evaluate quadratic terms
        objective = 0.
        for k in ['rho', 'sigma']:
            objective -= sum(np.linalg.norm(vt)**2 for vt in variables[k]) / 4.

        # cost initial conditions
        objective -= variables['lam'][0].dot(x0)

        # cost bounds on binaries
        v_lb, v_ub = self._get_bounds_on_binaries(identifier)
        objective += sum(v_lb[t].dot(vt) for t, vt in enumerate(variables['nu_lb']))
        objective -= sum(v_ub[t].dot(vt) for t, vt in enumerate(variables['nu_ub']))

        # cost mld inequalities
        objective -= sum(self.mld.h.dot(vt) for vt in variables['mu'][:-1])
        objective -= self.h_Tm1.dot(variables['mu'][-1])

        return objective

    def _get_bounds_on_binaries(self, identifier):
        '''
        Restates the identifier in terms of lower an upper bounds on the binary variables in the problem.

        Parameters
        ----------
        identifier : dict
            Dictionary containing the values for some of the binaries.
            Pass an empty dictionary to reset the bounds of the binaries to 0 and 1.

        Returns
        -------
        v_lb : list of numpy arrays
            Lower bound imposed by the identifier on the binary inputs in the problem.
        v_ub : list of numpy arrays
            Upper bound imposed by the identifier on the binary inputs in the problem.=
        '''

        # initialize bounds on the binary inputs
        v_lb = [np.zeros(self.mld.nub) for t in range(self.T)]
        v_ub = [np.ones(self.mld.nub) for t in range(self.T)]

        # parse identifier
        for k, v in identifier.items():
            v_lb[k[0]][k[1]] = v
            v_ub[k[0]][k[1]] = v

        return v_lb, v_ub

    def construct_warm_start(self, leaves, x0, uc0, ub0, e0):

        # needed for a (redundant) check
        u0 = np.concatenate((uc0, ub0))
        x1 = self.mld.A.dot(x0) + self.mld.B.dot(u0) + e0

        # initialize nodes for warm start
        warm_start = []

        # check each on of the optimal leaves
        for l in leaves:

            # if the identifier of the leaf does not agree with the stage_variables drop the leaf
            if self._retain_leaf(l.identifier, ub0):
                shifted_identifier = self._shift_identifier(l.identifier)
                print(vars(l))
                dual = self._shift_dual_solution(l.additional.dual.variables, shifted_identifier, x1)

                # propagate lower bounds if leaf is feasible
                if not np.isinf(l.lb):
                    primal = None
                    lb = dual.objective
                    
                # propagate infeasibility if leaf is still infeasible
                else:
                    v_lb, v_ub = self._get_bounds_on_binaries(l.identifier)
                    multipliers = l.additional.dual.variables
                    lam1 = l.additional.dual.variables['lam'][1]
                    pi_3 = - (self.mld.F.dot(x0) + self.mld.G.dot(u0) - self.mld.h).dot(multipliers['mu'][0]) + \
                           - (v_lb[0] - self.mld.V.dot(u0)).dot(multipliers['nu_lb'][0]) + \
                           - (self.mld.V.dot(u0) - v_ub[0]).dot(multipliers['nu_ub'][0])
                    pi_5 = .25 * np.linalg.norm(multipliers['rho'][-1])**2 - \
                           .25 * np.linalg.norm(dual.variables['rho'][-2])**2
                    pi_7 = self.h_Tm1.dot(multipliers['mu'][-1]) - self.mld.h.dot(dual.variables['mu'][-2])

                    if multipliers['lam'][1].dot(e0) < l.additional.dual.objective + pi_3 + pi_5 + pi_7:
                        primal = PrimalSolution.infeasible(self.T)
                        lb = np.inf
                    else:
                        primal = None
                        lb = - np.inf

                # add new node to the list for the warm start
                integer_feasible = None
                solution = SubproblemSolution(primal, dual, integer_feasible)
                node = Node(shifted_identifier, lb=lb, additional=solution)
                warm_start.append(node)

        return warm_start

    @staticmethod
    def _retain_leaf(identifier, ub0):
        '''
        '''

        # retain until proven otherwise
        retain = True
        for k, v in identifier.items():

            # check if the identifier agrees with the input variable at time zero
            if k[0] == 0 and not np.isclose(v, ub0[k[1]]):
                retain = False
                break

        return retain

    @staticmethod
    def _shift_identifier(identifier):
        return {(k[0]-1, k[1]): v for k, v in identifier.items() if k[0] > 0}
