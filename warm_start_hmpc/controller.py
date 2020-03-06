# external imports
import gc
import numpy as np
from time import time
from operator import le, ge, eq

# internal inputs
from warm_start_hmpc.bounded_qp import BoundedQP
from warm_start_hmpc.subproblem_solution import SubproblemSolution, PrimalSolution, DualSolution
from warm_start_hmpc.branch_and_bound import Node, branch_and_bound, best_first, depth_first

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
    list of dict
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

class HybridModelPredictiveController(object):
    '''
    Optimal controller for Mixed Logical Dynamical (mld) systems.
    Solves the mixed-integer quadratic optimization problem:
    min | Q_T x_T |^2 + sum_{t=0}^{T-1} | Q x_t |^2 + | R u_t |^2
    s.t. x_0 given,
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
            Weight matrices in the following order: Q, R, Q_T.
        terminal_set : list of np.array or None
            Terminal constraint in the form F_T x_T <= h_T.
            The first element of the list must be F_T, the second h_t.
            If no terminal constraint is required set this to None.
        '''

        # store inputs
        self.mld = mld
        self.T = T
        self.Q, self.R, self.Q_T = objective

        # terminal constraint is enforced via the stage
        # constraints for time step T-1
        if terminal_set is None:
            terminal_set = [np.empty((0, self.mld.x)), np.empty(0)]
        self.F_Tm1 = np.vstack((mld.F, terminal_set[0].dot(mld.A)))
        self.G_Tm1 = np.vstack((mld.G, terminal_set[0].dot(mld.B)))
        self.h_Tm1 = np.concatenate((mld.h, terminal_set[1]))

        # build mixed integer program 
        self._check_input_sizes()
        self.qp = self._build_mip()

        # warm start construction
        self._update = {
            'mu': self._update_mu(), # terminal constraint
            'rho': np.linalg.pinv(self.Q.T).dot(self.Q_T.T) # teminal cost
        }

    def _check_input_sizes(self):
        '''
        Checks that the matrices passed as inputs in the initialization of the
        class have the right properties.
        '''

        # weight matrices
        if self.Q.shape[1] != self.mld.nx:
            raise ValueError('Matrix Q has wrong number of columns.')
        if self.R.shape[1] != self.mld.nu:
            raise ValueError('Matrix R has wrong number of columns.')
        if self.Q_T.shape[1] != self.mld.nx:
            raise ValueError('Matrix Q_T has wrong number of columns.')

        # terminal constraint
        if self.F_Tm1.shape[0] != self.h_Tm1.size:
            raise ValueError('Terminal-set matrices have wrong number of rows.')
        if self.G_Tm1.shape[0] != self.h_Tm1.size:
            raise ValueError('Terminal-set matrices have wrong number of rows.')

    def _build_mip(self):
        '''
        Builds the guorbi quadratic program for the optimization problem to be solved.

        Returns
        -------
        BoundedQP
        Convex relaxation of the MIQP in the form of a Gurobi QP.
        '''

        # initialize program
        qp = BoundedQP()
        obj = 0.

        # initial state
        # initially set to zero, this value will be changed when calling the controller
        x_next = qp.add_variables(self.mld.nx, name='x_0')
        qp.add_constraints(x_next, eq, [0.]*self.mld.nx, name='lam_0')

        # loop over the time horizon
        for t in range(self.T):

            # stage variables
            x = x_next
            x_next = qp.add_variables(self.mld.nx, name=f'x_{t+1}')
            uc = qp.add_variables(self.mld.nuc, name=f'uc_{t}')
            ub = qp.add_variables(self.mld.nub, name=f'ub_{t}')
            u = np.concatenate((uc, ub))
            
            # bounds on the binaries
            # inequalities are stated as [expr <= num] to get negative duals
            # (note that [num <= expr] would be modified by Gurobi
            # to [expr => num] and would give positive duals)
            qp.add_constraints(-ub, le, [0.]*self.mld.nub, name=f'nu_lb_{t}')
            qp.add_constraints(ub, le, [1.]*self.mld.nub, name=f'nu_ub_{t}')

            # mld dynamics
            qp.add_constraints(
                x_next, eq, self.mld.A.dot(x) + self.mld.B.dot(u),
                name=f'lam_{t+1}')

            # mld stage constraints
            if t < self.T-1:
                qp.add_constraints(
                    self.mld.F.dot(x) + self.mld.G.dot(u), le, self.mld.h,
                    name=f'mu_{t}')
            
            # mld constraint plus terminal constraint
            else:
                qp.add_constraints(
                    self.F_Tm1.dot(x) + self.G_Tm1.dot(u), le, self.h_Tm1,
                    name=f'mu_{t}')

            # stage cost
            Qx = self.Q.dot(x)
            Ru = self.R.dot(u)
            obj += Qx.dot(Qx) + Ru.dot(Ru)

        # terminal cost
        QxT = self.Q_T.dot(x_next)
        obj += QxT.dot(QxT)

        # set cost
        qp.setObjective(obj)

        return qp

    def _update_mu(self):
        '''
        Computes the matrix neeed to compute the shifted value of the
        multiplier mu_{T-2|1}. In the warm start process, this multiplier is
        set to mu_{T-2|1} := M mu_{T-1|0}. This function computes the matrix M.

        Returns
        -------
        np.array
            Matrix for the update of the mutiplier mu.
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
            mu_i = lp.primal_optimizer('mu')
            if mu_i is None:
                raise ValueError('The conic hull of [F G] does not contain the one of [F_Tm1 G_Tm1].')
            M.append(mu_i)

        return np.vstack(M).T

    def _solve_subproblem(self, identifier, x0, active_set):
        '''
        Solves the QP relaxation indentified by the identifier for the given
        initial state.

        Parameters
        ----------
        identifier : dict
            Dictionary containing the values of selected binaries.
        x0 : np.array
            Initial state of the system.
        active_set : dict with two list of int
            Active set to warm start the dual simplex method.
            Has entries 'c' for the constraints and 'v' fo the variables.
            Each element is equal to -1 if in the active set , 0 otherwise.
            Constraints and variables are ordered as in the Gurobi methods
            self.qp.getConstrs() and self.qp.getVars().

        Returns
        -------
        SubproblemSolution
            Solution of the subproblem.
        '''

        # set bounds on binaries
        self._set_bound_binaries(identifier)

        # set initial conditions
        self.qp.set_constraint_rhs('lam_0', x0)

        # warm start the dual simplex with the given active set
        # (must be done after _set_bound_binaries and set_constraint_rhs
        # otherwise Gruobi clears it when modifying constraints)
        if active_set is not None and self.qp.Params.Method == 1:
            [c.setAttr('CBasis', active_set['c'][i]) for i, c in enumerate(self.qp.getConstrs())]
            [v.setAttr('VBasis', active_set['v'][i]) for i, v in enumerate(self.qp.getVars())]
        
        # run the optimization and initialize the result
        self.qp.optimize()
        solution = SubproblemSolution.from_controller(self)
        solve_time = self.qp.Runtime

        return solution, solve_time

    def _set_bound_binaries(self, identifier):
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
            self.qp.set_constraint_rhs(f'nu_lb_{t}', lb_t)
            self.qp.set_constraint_rhs(f'nu_ub_{t}', ub_t)

    def _get_bound_binaries(self, identifier):
        '''
        Restates the identifier in terms of lower an upper bounds on the binary variables in the problem.

        Parameters
        ----------
        identifier : dict
            Dictionary containing the values of selected binaries.

        Returns
        -------
        ub_lb : list of numpy arrays
            Lower bound imposed by the identifier on the binary inputs in the problem.
        ub_ub : list of numpy arrays
            Upper bound imposed by the identifier on the binary inputs in the problem.
        '''

        # initialize bounds on the binary inputs
        ub_lb = np.zeros((self.T, self.mld.nub))
        ub_ub = np.ones((self.T, self.mld.nub))

        # parse identifier
        for k, v in identifier.items():
            ub_lb[k] = v
            ub_ub[k] = v

        return ub_lb, ub_ub

    def feedforward(self, x0, gurobi_params={}, search_rule=best_first, branch_rule=branch_in_time, **kwargs):
        '''
        Solves the mixed integer program using the branch_and_bound function.

        Parameters
        ----------
        x0 : np.array
            Current state of the system.
        gurobi_params : dict
            Options for the Gurobi QP solver.
        search_rule : function
            Search heuristic for the branch and bound algorithm.
        branhc_rule : function
            Branching heuristic for the branch and bound algorithm.

        Returns
        -------
        PrimalSolution
            Primal solution associated with the incumbent node at optimality.
            None if problem is infeasible.
        list of Node
            Leaves of the branch and bound tree that proved optimality of the
            returned solution. 
        int
            Number of subproblems solved in the branch and bound algorithm.
        float
            Time spent solving QPs in GUrobi during branch and bound.
        '''

        # set gurobi parameters
        self._set_gurobi_params(gurobi_params)

        # reset model to discard old solutions
        self.qp.reset() # prints "Discarded solution information"

        # solver for the branch and bound algorithm
        def solver(identifier, cutoff, extra):

            # TODO: think about consequences of using the cutoff here
            # self.qp.Params.Cutoff = cutoff

            # solve the qp
            active_set = extra.active_set if extra is not None else None
            solution, solve_time = self._solve_subproblem(identifier, x0, active_set)
            objective = solution.primal.objective
            binary_feasible = solution.primal.binary_feasible

            return objective, binary_feasible, solve_time, solution

        # brancher for the branch and bound algorithm
        def brancher(parent):
            return self._brancher(parent, branch_rule)

        # solve the mixed integer program using branch and bound
        # TODO: search heuristic and tree exploration as params to this function
        # TODO: more compact output for branch_and_bound function
        incumbent, leaves, qp_solves, solver_time = branch_and_bound(solver, search_rule, brancher, **kwargs)

        # infeasible problem
        if incumbent is None:
            primal = None
            return primal, leaves, qp_solves, solver_time

        # feasible problem
        return incumbent.extra.primal, leaves, qp_solves, solver_time

    def _brancher(self, parent, branch_rule):
        '''
        Given a parent node from the branch and bound tree, generates the
        children nodes.

        Parameters
        ----------
        parent : Node
            Branch and bound node to be branched.

        Returns
        -------
        list of Node
            List of the generated children nodes for the parent node.
        '''

        # get branch dictionaries from any branch rule heuristic
        branches = branch_rule(parent.identifier, self.mld.nub)

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
            solution = SubproblemSolution(None, parent.extra.dual, parent.extra.active_set)
            children.append(Node(identifier, lb, solution))

        return children

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

        # disable garbace collector and start stopwatch
        gc.disable()
        construction_time = time()

        # create warm start checking one leaf per time
        warm_start = []
        u0 = np.concatenate((uc0, ub0))
        for leaf in leaves:

            # shortcuts
            identifier = leaf.identifier
            variables = leaf.extra.dual.variables
            objective = leaf.extra.dual.objective

            # if the identifier of the leaf does not agree with ub0 drop the leaf
            if self._retain_leaf(identifier, ub0):

                # construct feasible solution for the warm start node
                shifted_variables = self._shift_dual_variables(variables)
                pi = self._get_pi(identifier, variables, shifted_variables, x0, u0, e0)
                shifted_objective = max(0., objective + sum(pi))

                # if was infeasible and still infeasible
                if np.isinf(leaf.lb):
                    if shifted_objective > 0.:
                        shifted_lb = np.inf
                        shifted_dual = DualSolution(shifted_variables, shifted_objective)

                    # if was infeasible and now could be feasible
                    else:
                        shifted_lb = 0.
                        shifted_dual = None

                # if it was feasible
                else:
                    shifted_lb = shifted_objective
                    shifted_dual = DualSolution(shifted_variables, shifted_objective)

                # add new node to the list for the warm start
                # shifted_identifier = {(k[0]-1, k[1]): v for k, v in identifier.items() if k[0] > 0}
                shifted_solution = SubproblemSolution(None, shifted_dual)
                warm_start.append(Node(shifted_identifier, shifted_lb, shifted_solution))

        # stop startwatch and renable garbace collection
        construction_time = time() - construction_time
        gc.enable()

        return warm_start, construction_time

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
        Bool
            True if the leaf can be reatained, Flase if it must be dropped.
        '''

        return all(v == ub0[k[1]] for k, v in identifier.items() if k[0]==0)

    def _shift_dual_variables(self, variables):
        '''
        Shifts a dual solution from one time step to the next.

        Parameters
        ----------
        variables : dict
            Dictionary containing the dual variables to be shifted in time.
            
        Returns
        -------
        DualSolution
            Dual feasible solution of the shifted problem.
        '''

        # initialize shifted dual variables
        shifted_variables = {}

        # shift backwards by one and append zero
        for k in ['lam', 'nu_lb', 'nu_ub', 'sigma']:
            # shifted_variables[k] = [v for v in variables[k][1:]]
            shifted_variables[k] = variables[k][1:]
            shifted_variables[k].append(np.zeros(variables[k][-1].shape))

        # shift backwards by one, append optimal update and zero
        for k in ['mu', 'rho']:
            # shifted_variables[k] = [v for v in variables[k][1:-1]]
            shifted_variables[k] = variables[k][1:-1]
            shifted_variables[k].append(self._update[k].dot(variables[k][-1]))
            shifted_variables[k].append(np.zeros(variables[k][-1].shape))

        return shifted_variables

    def _get_pi(self, identifier, variables, shifted_variables, x0, u0, e0):
        '''
        Evaluates the terms pi3, pi5, pi7 needed to check if a subproblem is still infeasible after shifting.

        Parameters
        ----------
        identifier : dict
            Dictionary containing the values of selected binaries.
        variables : dict
            Dictionary containing the dual solution of the subproblem already solved.
        shifted_variables : dict
            Dictionary containing the dual feasible solution of the shifted subproblem.
        x0 : np.array
            Initial state of the system.
        u0 : np.array
            Input injected in the system.
        e0 : np.array
            Modeling error e0 = x1 - A x0 - B u0 at the last time step.

        Returns
        -------
        pi : list of float
            Value of the terms that represent the difference in the objective of two consecutive problem.
            (See paper for the meaning of these.)
        '''

        # stage cost
        squared = lambda x : x.dot(x)
        pi = np.zeros(6)
        Qx0 = self.Q.dot(x0)
        Ru0 = self.R.dot(u0)
        pi[0] = - squared(Qx0) - squared(Ru0)

        # suboptimality cost
        pi[1] = squared(.5*variables['rho'][0] - Qx0) + squared(.5*variables['sigma'][0] - Ru0)

        # complementarity slackness
        ub_lb, ub_ub = self._get_bound_binaries(identifier)
        residuals = {
            'mu': self.mld.F.dot(x0) + self.mld.G.dot(u0) - self.mld.h,
            'nu_lb': ub_lb[0] - self.mld.V.dot(u0),
            'nu_ub': self.mld.V.dot(u0) - ub_ub[0]
        }
        pi[2] = - sum(residual.dot(variables[k][0]) for k, residual in residuals.items())

        # model error
        pi[3] = - variables['lam'][1].dot(e0)

        # terminal cost
        pi[4] = .25 * squared(variables['rho'][self.T]) - \
                .25 * squared(shifted_variables['rho'][self.T-1])

        # terminal constraint
        pi[5] = self.h_Tm1.dot(variables['mu'][self.T-1]) - \
                self.mld.h.dot(shifted_variables['mu'][self.T-2])

        return pi

    def feedforward_gurobi(self, x0, gurobi_params={}):
        '''
        Solves the mixed integer program using Gurobi.

        Parameters
        ----------
        x0 : np.array
            Current state of the system.
        gurobi_params : dict
            Options for the Gurobi QP solver.

        Returns
        -------
        np.array
            State trajectory (number of time steps by number of states).
        float
            Optimal objective value.
        int
            Number of subproblems solved in the branch and bound algorithm.
        float
            Time spent in Gurobi solving the MIQP.
        '''

        # reset model to discard old solutions
        self.qp.reset()

        # set up miqp
        self._set_gurobi_params(gurobi_params)
        self._set_bound_binaries({})
        self._set_binaries_type('B')
        self.qp.set_constraint_rhs('lam_0', x0)

        # run the optimization
        self.qp.optimize()

        # unpack solution
        x = np.vstack([self.qp.primal_optimizer(f'x_{t}') for t in range(self.T+1)])
        objective = self.qp.primal_objective()
        n_nodes = int(self.qp.NodeCount)
        solver_time = self.qp.Runtime

        # reset internal qp to default
        self._set_binaries_type('C')
        self.qp.reset()

        return x, objective, n_nodes, solver_time

    def _set_gurobi_params(self, gurobi_params):
        '''
        Sets the parameters in the Gurobi solver.

        Parameters
        ----------
        gurobi_params : dict
            Options for the Gurobi QP solver.
        '''

        # reset to default
        self.qp.resetParams()

        # changes the dafult to not print
        self.qp.setParam('OutputFlag', 0)

        # set the custom parameters
        for param, value in gurobi_params.items():
            self.qp.setParam(param, value)

    def _set_binaries_type(self, type):
        '''
        Sets the type of the bianries in the optimziations problem.

        Parameters
        ----------
        type : sting
            'C' to set the binaries to be continous (between 0 and 1), 'B' to
            force them to assume bianry values only.
        '''

        [ub.setAttr('VType', type) for t in range(self.T) for ub in self.qp.get_variables(f'ub_{t}')]

    # def _evaluate_dual_objective(self, identifier, variables, x0):

    #     # quadratic term
    #     rho_sigma = np.concatenate(variables['rho'] + variables['sigma'])
    #     obj = - .25 * rho_sigma.dot(rho_sigma)

    #     # linear term
    #     ub_lb, ub_ub = self._get_bound_binaries(identifier)
    #     lhs = np.concatenate(
    #         variables['mu'] +
    #         variables['nu_ub'] +
    #         variables['nu_lb'] +
    #         variables['lam'][:1]
    #         )
    #     rhs = np.concatenate(
    #         [self.mld.h] * (self.T-1) +
    #         [self.h_Tm1] +
    #         ub_ub.tolist() +
    #         (-ub_lb).tolist() +
    #         [x0]
    #         )
    #     obj -= lhs.dot(rhs)

    #     return max(0, obj)
