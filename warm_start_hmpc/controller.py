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
         V u_t is binary,         t = 0, 1, ..., T-1,
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
        Builds the guorbi quadratic program for the optimization problem to be solved.

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
            
            # bounds on the binaries: inequalities must be stated as expr <= num, to get negative duals
            # (note that num <= expr would be modified to expr => num and would give positive duals)
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
        '''
        Computes the matrix neeed to compute the shifted value of the multiplier mu_{T-2|1}.
        In the warm start process, this multiplier is set to mu_{T-2|1} := M mu_{T-1|0}.
        This function computes the matrix M.

        Returns
        -------
        M : np.array
            Matrix for the update of the mutiplier.
        '''

        # size of the linea programs
        n = self.mld.h.size # size of mu_{T-2|1}
        m = self.h_Tm1.size # size of mu_{T-1|0}

        # initialize LP, optimization variables, and objective
        lp = BoundedQP()
        mu = lp.add_variables(n, name='mu')
        obj = lp.setObjective(self.mld.h.dot(mu))

        # initialize constraints
        lp.add_constraints(self.mld.F.T.dot(mu), eq, [0.]*self.mld.nx, name='x')
        lp.add_constraints(self.mld.G.T.dot(mu), eq, [0.]*self.mld.nu, name='u')
        lp.add_constraints(mu, ge, [0.]*n)

        # loop thorugh the columns of the rhs
        M = []
        for i in range(m):

            # select the ith column of F_{T-1} and G_{T-1}
            lp.set_constraint_rhs('x', self.F_Tm1[i])
            lp.set_constraint_rhs('u', self.G_Tm1[i])

            # get columnn of M
            lp.optimize()
            M.append(lp.primal_optimizer('mu'))

        return np.vstack(M).T

    def _solve_subproblem(self, x0, identifier, cutoff=None):
        '''
        Solves the QP relaxation indentified by the identifier for the given initial state.

        Parameters
        ----------
        x0 : np.array
            Initial state of the system.
        identifier : dict
            Dictionary containing the values of selected binaries.
        cutoff : float
            Objective threshold above which the solution of the problem can be terminated.
            (Not used at the moment.)

        Returns
        -------
        solution : SubproblemSolution
            Instace of the class which contains the solution of a subproblem.
        '''

        # set initial conditions
        self.qp.set_constraint_rhs('lam_0', x0)
        self._set_bounds_binaries(identifier)

        # run the optimization and initialize the result
        self.qp.optimize() # TODO: use cutoff here

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
            Dictionary containing the values of selected binaries.
            Pass an empty dictionary to reset the bounds of the binaries to [0, 1].
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
            Current state of the system.

        Returns
        -------
        primal : PrimalSolution
            Primal solution associated with the incumbent node at optimality.
            None if problem is infeasible.
        leaves : list of Node
            Leaves of the branch and bound tree that proved optimality of the returned solution. 
        '''

        # generate a solver for the branch and bound algorithm
        def solver(identifier, cutoff):
            solution = self._solve_subproblem(x0, identifier, cutoff)
            return solution.primal.objective, solution.primal.binary_feasible, solution

        # solve the mixed integer program using branch and bound
        incumbent, leaves = branch_and_bound(solver, best_first, self._brancher, **kwargs)

        # infeasible problem
        if incumbent is None:
            return None, leaves

        # feasible problem
        return incumbent.extra.primal, leaves

    def _brancher(self, parent):
        '''
        Given a parent node from the branch and bound tree, generates the children nodes.

        Parameters
        ----------
        parent : Node
            Branch and bound node to be branched.

        Returns
        -------
        children : list of Node
            List of the generated children nodes for the parent node.
        '''

        # get branch dictionaries from any heuristic
        branches = branch_in_time(parent.identifier, self.mld.nub)

        # construct a node per branch
        children = []
        for branch in branches:
            
            # plug parent dual solution in the child objective
            lb = parent.lb
            for k, v in branch.items():
                nu = 'nu_lb' if v == 1 else 'nu_ub' if v == 0 else None
                lb += parent.extra.dual.variables[nu][k[0]][k[1]]

            # instantiate child solution with no primal and parent dual
            identifier = {**parent.identifier, **branch}
            solution = SubproblemSolution(None, parent.extra.dual)
            children.append(Node(identifier, lb, solution))

        return children

    def _get_interval_binaries(self, identifier):
        '''
        Restates the identifier in terms of lower an upper bounds on the binary variables in the problem.

        Parameters
        ----------
        identifier : dict
            Dictionary containing the values of selected binaries.

        Returns
        -------
        v_lb : list of numpy arrays
            Lower bound imposed by the identifier on the binary inputs in the problem.
        v_ub : list of numpy arrays
            Upper bound imposed by the identifier on the binary inputs in the problem.
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
        '''
        Generates the warm start for the MIQP at the next time step.

        Parameters
        ----------
        leaves : list of Node
            Leaves of the branch and bound tree that proved optimality of the previous solution.
        x0 : np.array
            Initial state of the system for the problem solved at the last time step.
        uc0 : np.array
            Value of the continuous inputs injected in the system at the last time step.
        ub0 : np.array
            Value of the bianry inputs injected in the system at the last time step.
        e0 : np.array
            Modeling error e0 = x1 - A x0 - B u0 at the last time step.

        Returns
        -------
        warm_start : list of Node
            Nodes to initialize the branch and bound algorithm.
        '''

        # state update
        u0 = np.concatenate((uc0, ub0))
        x1 = self.mld.A.dot(x0) + self.mld.B.dot(u0) + e0

        # create wrm start checking one leaf per time
        warm_start = []
        for leaf in leaves:
            dual = leaf.extra.dual

            # if the identifier of the leaf does not agree with the ub0 drop the leaf
            if self._retain_leaf(leaf.identifier, ub0):
                shifted_identifier = {(k[0]-1, k[1]): v for k, v in leaf.identifier.items() if k[0] > 0}
                shifted_primal = None
                shifted_dual = self._shift_dual(x1, shifted_identifier, dual.variables)
                shifted_solution = SubproblemSolution(shifted_primal, shifted_dual)

                # propagate lower bounds if leaf is feasible
                if not np.isinf(leaf.lb):
                    shifted_lb = shifted_dual.objective
                    
                # if leaf is infeasible
                else:

                    # elements to check if it could now be feasible
                    pi = self._get_pi(x0, u0, leaf.identifier, dual.variables, shifted_dual.variables)
                    lhs = dual.variables['lam'][1]
                    rhs = dual.objective + sum(pi)

                    # still infeasible
                    if lhs.dot(e0) < rhs:
                        shifted_lb = np.inf

                    # maybe feasible
                    else:
                        shifted_lb = 0.

                # add new node to the list for the warm start
                warm_start.append(Node(shifted_identifier, shifted_lb, shifted_solution))

        return warm_start

    @staticmethod
    def _retain_leaf(identifier, ub0):
        '''
        Given the identifier of a leaf, it checks it agrees with the binary action ub0.

        Parameters
        ----------
        identifier : dict
            Identifier of the leaf to be checked.
        ub0 : np.array
            Binary input of the MLD system.

        Returns
        -------
        retain : Bool
            True if the leaf can be reatained, Flase if it must be dropped.
        '''

        # retain until proven otherwise
        retain = True
        for k, v in identifier.items():

            # check if the identifier agrees with the input variable at time zero
            if k[0] == 0 and not np.isclose(v, ub0[k[1]]):
                retain = False
                break

        return retain

    def _shift_dual(self, x1, shifted_identifier, variables):
        '''
        Shifts a dual solution from one time step to the next.

        Parameters
        ----------
        x1 : np.array
            Value of the initial state for the shifted problem.
        shifted_identifier : dict
            Identifier of the shifted subproblem.
        variables : dict
            Dictionary containing the dual variables to be shifted in time.
            
        Returns
        -------
        shifted_dual : DualSolution
            Dual feasible solution of rthe shifted problem.
        '''

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
        shifted_objective = self._evaluate_dual(x1, shifted_identifier, shifted_variables)
        shifted_objective = max(shifted_objective, 0.)

        return DualSolution(shifted_variables, shifted_objective)

    def _evaluate_dual(self, x0, identifier, variables):
        '''
        Given a dual solution, returns it cost.

        Parameters
        ----------
        x0 : np.array
            Initial state of the system.
        identifier : dict
            Dictionary containing the values of selected binaries.
        variables : dict
            Dictionary containing the dual solution to be evaluated.

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
        v_lb, v_ub = self._get_interval_binaries(identifier)
        objective += sum(v_lb[t].dot(vt) for t, vt in enumerate(variables['nu_lb']))
        objective -= sum(v_ub[t].dot(vt) for t, vt in enumerate(variables['nu_ub']))

        # cost mld inequalities
        objective -= sum(self.mld.h.dot(vt) for vt in variables['mu'][:-1])
        objective -= self.h_Tm1.dot(variables['mu'][-1])

        return objective

    def _get_pi(self, x0, u0, identifier, variables, shifted_variables):
        '''
        Evaluates the terms pi3, pi5, pi7 needed to check if a subproblem is still infeasible after shifting.

        Parameters
        ----------
        x0 : np.array
            Initial state of the system.
        u0 : np.array
            Input injected in the system.
        identifier : dict
            Dictionary containing the values of selected binaries.
        variables : dict
            Dictionary containing the dual solution of the subproblem already solved.
        shifted_variables : dict
            Dictionary containing the dual feasible solution of the shifted subproblem.

        Returns
        -------
        pi : list of float
            Value of pi3, pi5, and pi7.
            (See paper for the meaning of these.)
        '''

        # cost of the complementarity slackness
        v_lb, v_ub = self._get_interval_binaries(identifier)
        residuals = {
            'mu': self.mld.F.dot(x0) + self.mld.G.dot(u0) - self.mld.h,
            'nu_lb': v_lb[0] - self.mld.V.dot(u0),
            'nu_ub': self.mld.V.dot(u0) - v_ub[0]
        }
        pi3 = - sum(residual.dot(variables[k][0]) for k, residual in residuals.items())

        # cost of the terminal cost
        pi5 = .25 * np.linalg.norm(variables['rho'][-1])**2 - \
               .25 * np.linalg.norm(shifted_variables['rho'][-2])**2

        # cost of the terminal constraint
        pi7 = self.h_Tm1.dot(variables['mu'][-1]) - \
               self.mld.h.dot(shifted_variables['mu'][-2])

        return pi3, pi5, pi7

def branch_in_time(identifier, nub):
    '''
    Branching heuristic search for the branch and bound algorithm.
    Select the children nodes proceeding in chronological order.

    Parameters
    ----------
    identifier : dict
        Dictionary containing the values of selected binaries.
        Pass an empty dictionary to reset the bounds of the binaries to 0 and 1.
    nub : int
        Number of binary inputs in the MLD system.

    Returns
    -------
    branches : list of dict
        List of sub-identifier that, if merged with the identifier of the parent, give the identifier of the children.
    '''

    # idices of the last binary fixed in time
    t = max([k[0] for k in identifier.keys()] + [0])
    index = max([k[1]+1 for k in identifier.keys() if k[0] == t] + [0])

    # try to fix one more ub at time t
    if index < nub:
        branches = [{(t,index): 0.}, {(t,index): 1.}]

    # if everything is fixed at time t, move to time t+1
    else:
        branches = [{(t+1,0): 0.}, {(t+1,0): 1.}]

    return branches