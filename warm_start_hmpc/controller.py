# external imports
import numpy as np
import gurobipy as grb
from copy import copy, deepcopy
from operator import le, ge, eq

# internal inputs
from gurobi_model import GurobiModel
from pympc.control.hybrid_benchmark.branch_and_bound import Node, branch_and_bound, best_first

class HybridModelPredictiveController(object):
    '''
    Optimal controller for Mixed Logical Dynamical (mld) systems.
    Solves the mixed-integer quadratic optimization problem:
    min | C_T x_T |^2 + sum_{t=0}^{T-1} | C x_t |^2 + | D u_t |^2
    s.t. x_0 given
         x_{t+1} = A x_t + B u_t, t = 0, 1, ..., T-1,
         F x_t + G u_t <= h,      t = 0, 1, ..., T-1,
         F_T x_T <= h_T,
         xb_t binary,             t = 0, 1, ..., T,
         ub_t binary,             t = 0, 1, ..., T-1,
    '''

    def __init__(self, mld, T, CD, C_T=None, X_T=None):
        '''
        Instantiates the hybrid MPC controller.

        Parameters
        ----------
        mld : instance of MixedLogicalDynamicalSystem
            System to be controlled.
        T : int
            Horizon of the controller.
        CD : list of np.array
            The first element of the list must be C, the second D.
        C_T : np.array
            Terminal weight matrix.
        X_T : list of np.array
            Termianal constraint in the form F_T x_T <= h_T.
            The first element of the list must be F_T, the second h_t.
        '''

        # store inputs
        self.mld = mld
        self.T = T
        [self.C, self.D] = CD
        if C_T is None:
            self.C_T = self.C
        else:
            self.C_T = C_T
        self.X_T = X_T

        # build mixed integer program 
        self._check_input_sizes()
        self.model = self._build_mip()

        # warm start construction
        if C_T is None:
            update_rho = np.eye(self.C.shape[0])
        else:
            update_rho = np.linalg.pinv(self.C.T).dot(self.C_T)
        update_mu = self._update_mu()

    def _update_mu(self):

        # if no terminal set
        n1 = self.mld.h.size
        n2 = self.X_T[1].size
        if X_T is None:
            return np.eye(n1+n2)[:,:n1], np.eye(n1+n2)[:,n1:]

        # initialize LP, optimization variable, and objective
        model = GurobiModel()
        mu = model.add_variables(self.mld.h.size, lb=[0.]*self.mld.h.size)
        obj = model.setObjective(self.mld.h.dot(mu))
        update_mu = []

        # initialize constraints
        model.add_linear_constraints(
                self.mld.F.T.dot(mu),
                eq,
                np.zeros(self.mld.h.size),
                name'F'
                )
        model.add_linear_constraints(
                self.mld.G.T.dot(mu),
                eq,
                np.zeros(self.mld.h.size),
                name'G'
                )

        # loop thorugh the columns of the rhs
        for i in range(n1+n2):

            # unitary vector
            ei = np.zeros(n1+n2)
            ei[i] = 1.

            # set rhs constraints
            rhsF = np.vstack((self.mld.F, self.X_T[0].dot(self.mld.A))).T.dot(ei)
            rhsG = np.vstack((self.mld.G, self.X_T[0].dot(self.mld.B))).T.dot(ei)
            [ci.RHS = rhsF[i] for i, ci in enumerate(model.get_constraints('F'))]
            [ci.RHS = rhsG[i] for i, ci in enumerate(model.get_constraints('G'))]

            # get columnn of M
            model.optimize()
            update_mu.append(np.array([mui.x for mui in model.get_variables('mu')]))

        return np.hstack(update_mu[:n1]), np.hstack(update_mu[n1:])


    def _check_input_sizes(self):
        '''
        Checks that the matrices passed as inputs in the initialization of the class have the right properties.
        '''

        # weight matrices
        assert self.C.shape[1] == self.mld.nx
        assert self.D.shape[1] == self.mld.nu
        assert self.C_T.shape[1] == self.mld.nx

        # terminal constraint
        if self.X_T is not None:
            assert self.X_T[0].shape[0] == self.X_T[1].size
            assert self.X_T[0].shape[1] == self.mld.nx

    def _build_mip(self):
        '''
        Builds the guorbi model for the opitmization problem to be solved.

        Returns
        -------
        model : GurobiModel
            Gurobi model of the mathematical program.
        '''

        # initialize program
        model = GurobiModel()
        obj = 0.

        # initial state (initialized to zero)
        x_next = model.add_variables(self.mld.nx, name='x_0')
        model.add_linear_constraints(x_next, eq, [0.]*self.mld.nx, name='lambda_0')

        # loop over the time horizon
        for t in range(self.T):

            # stage variables
            x = x_next
            uc = model.add_variables(self.mld.nuc, name='uc_%d'%t)
            ub = model.add_variables(self.mld.nub, name='ub_%d'%t)
            u = np.concatenate((uc, ub))
            x_next = model.add_variables(self.mld.nx, name='x_%d'%(t+1))

            # bounds on the binaries
            # inequalities must be stated as expr <= num to get negative duals
            # note that num <= expr would be modified to expr => num
            # and would give positive duals
            model.add_linear_constraints(-ub, le, [0.]*self.mld.nub, name='nu_b_%d'%t)
            model.add_linear_constraints(ub, le, [1.]*self.mld.nub, name='nu_ub_%d'%t)

            # mld dynamics
            model.add_linear_constraints(
                x_next,
                eq,
                self.mld.A.dot(x) + self.mld.Buc.dot(u),
                name='lambda_%d'%(t+1)
                )

            # # mld constraints
            # lhs = self.mld.F.dot(x) + self.mld.G.dot(u)
            # rhs = self.mld.h
            # if t == self.T-1 and self.X_T is not None:
            #     lhs = np.concatenate((lhs, self.X_T[0].dot(self.mld.A.dot(x) + self.mld.Buc.dot(u))))
            #     rhs = np.concatenate((lhs, self.X_T[1]))
            # model.add_linear_constraints(lhs, le, rhs, name='mu_%d'%t)

            # mld constraints
            model.add_linear_constraints(
                self.mld.F.dot(x) + self.mld.G.dot(u),
                le,
                self.mld.h,
                name='mu_%d'%t
                )
            
            # terminal contraint
            if t == self.T-1 and self.X_T is not None:
                model.add_linear_constraints(
                    self.X_T[0].dot(self.mld.A.dot(x) + self.mld.Buc.dot(u))
                    le,
                    self.X_T[1],
                    name='mu_%d'%self.T
                    )

            # stage cost
            Cx = self.C.dot(x)
            Du = self.D.dot(u)
            obj += Cx.dot(Cx) + Du.dot(Du)

        # terminal cost
        Cx = self.C_T.dot(x_next)
        obj += Cx.dot(Cx)

        # set cost
        model.setObjective(obj)

        return model

    def set_initial_condition(self, x0):
        '''
        Sets the initial state in the model to be equal to x0.

        Parameters
        ----------
        x0 : numpy.array
            Initial conditions for the optimization problem.
        '''

        # check size of x0
        assert self.mld.nx == x0.size

        # get the equality constraint for the initial condtions
        lambda_0 = self.model.get_constraints('lambda_0')

        # set initial conditions
        for k, xk in enumerate(x0):
            lambda_0[k].RHS = xk

        # update gurobi model to be safe
        self.model.update()

    def set_bounds_binaries(self, identifier):
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

            # set the bounds for the binary inputs
            lb = self.model.get_constraints('nu_lb_%d'%t)
            ub = self.model.get_constraints('nu_ub_%d'%t)
            for k in range(self.mld.nub):
                if identifier.has_key((t, k)):
                    # the minus is because constraints are stated as -u <= -u_lb
                    lb[k].RHS = - identifier[(t, k)]
                    ub[k].RHS = identifier[(t, k)]
                else:
                    lb[k].RHS = 0.
                    ub[k].RHS = 1.

        # update gurobi model to be safe
        self.model.update()

    def set_type_binaries(self, var_type):
        '''
        Sets the type of the variables ub in the optimization problem.

        Parameters
        ----------
        var_type : string
            Sting containing the type of ub .
            'C' for continuous, and 'D' for binary.
        '''

        # loop through all the ub in the problem
        for t in range(self.T):
            ub = self.model.get_variables('ub_%d'%t)
            for k in range(self.mld.nub):
                ub[k].VType = var_type

        # update gurobi model to be safe
        self.model.update()

    def solve_subproblem(self, x0, identifier, tol=1.e-5):
        '''
        Solves the QP relaxation uniquely indentified by its identifier for the given initial state.

        Parameters
        ----------
        x0 : np.array
            Initial state of the system.
        identifier : dict
            Dictionary containing the values for some of the binaries.
            Pass an empty dictionary to reset the bounds of the binaries to 0 and 1.
        tol : float
            Numeric tolerance in the checks.

        Returns
        -------
        solution : dict
            Items:
            - objective : float or np.inf (inf if infeasible)
            - integer_feasible : bool or None (None if infeasible)
            - primal_solution : dict or None (None if infeasible)
            - dual_solution : dict (Farkas' proof if infeasible)
        '''

        # reset model (gurobi does not try to use the last solve to warm start)
        self.model.reset()

        # set up miqp
        self.set_type_binaries('C')
        self.set_initial_condition(x0)
        self.set_bounds_binaries(identifier)

        # gurobi parameters
        self.model.setParam('OutputFlag', 0)

        # run the optimization and initialize the result
        self.model.optimize()
        solution = {}

        # if optimal
        if self.model.status == 2:
            solution['objective'] = self.model.objVal
            solution['integer_feasible'] = len(identifier) == self.T * self.mld.nub
            solution['primal_solution'] = self._organize_primal_solution()
            solution['dual_solution'] = self._organize_dual_solution()
            
        # if infeasible, infeasible_or_unbounded, numeric_errors, suboptimal
        elif self.model.status in [3, 4, 12, 13]:
            solution['objective'] = np.inf
            solution['integer_feasible'] = None
            solution['primal_solution'] = None
            self._do_farkas_proof()
            sol['dual_solution'] = self._organize_dual_solution()

        # if none of the previous raise error
        else:
            raise ValueError('unknown model status %d.' % self.model.status)

        # store cost of dual solution and check result
        solution['dual_solution']['objective'] = self._evaluate_dual_solution(x0, identifier, solution['dual_solution'])
        self._check_dual_solution(x0, identifier, solution['dual_solution'], tol)

        return solution

    def _organize_primal_solution(self):
        '''
        Organizes the primal solution of the convex subproblem.

        Returns
        -------
        primal : dict
            Dictionary containing the primal solution of the convex subproblem.
            Keys are 'x', 'uc', 'ub'.
            Each one of these is a list (ordered in time) of numpy arrays.
        '''

        # initialize primal solution
        solution = {'x': [], 'uc': [], 'ub': []}

        # loop states, inputs, and time steps
        for l in ['x','uc', 'ub']:
            for t in range(self.T+1):
                v = self.model.get_variables('%s_%d'%(l,t))
                if v.size > 0: # calling the input at time T we get a zero dimensional array
                    solution[l].append(np.array([vi.x for vi in v]))

        return solution

    def _organize_dual_solution(self):
        '''
        Organizes the dual solution of the convex subproblem.

        Returns
        -------
        dual : dict
            Dictionary containing the dual solution of the convex subproblem.
            Keys are 'lambda', 'mu', 'mu_eq', 'nu_lb', 'nu_ub', 'rho', 'sigma'.
            Each one of these is a list (ordered in time) of numpy arrays.
        '''

        # partial list of dual variables
        solution = {
            'lambda': [],
            'mu': [],
            'nu_lb': [],
            'nu_ub': []
            }

        # loop through the constraints
        for l in solution.keys:
            for t in range(self.T+1):
                c = self.model.get_constraints('%s_%d'%(l,t))
                if c.size > 0: # some multipliers do not appear at stage T

                    # get optimal duals or farkas proof if infeasible
                    # gurobi gives negative multipliers and positive farkas solutions!
                    if self.model.status == 2:
                        solution[l].append(- np.array([ci.Pi for ci in c]))
                    else:
                        solution[l].append(np.array([ci.FarkasDual for ci in c]))

        # state performance output
        solution['rho'] = []
        for t in range(self.T+1):
            C = self.C if t < self.T else self.C_T
            if self.model.status == 2:
                x = self.model.get_variables('x_%d'%t)
                x = np.array([xi.x for xi in x])
                solution['rho'].append[2*C.dot(x)]
            else:
                solution['rho'].append[np.zeros(C.size[0])]

        # input performance output
        solution['sigma'] = []
        for t in range(self.T):
            if self.model.status == 2:
                u = self.model.get_variables('u_%d'%t)
                u = np.array([ui.u for ui in u])
                solution['sigma'].append[2*self.D.dot(u)]
            else:
                solution['sigma'].append[np.zeros(self.D.size[0])]

        return solution

    def _do_farkas_proof(self):
        '''
        Performes the Farkas proof of infeasibility for the subproblem.
        It momentarily sets the objective to zero because gurobi can do the farkas proof only for linear programs.
        This can be very slow.
        '''

        # copy objective
        obj = self.model.getObjective()

        # rerun the optimization with linear objective
        # (only linear accepted for farkas proof)
        self.model.setParam('InfUnbdInfo', 1)
        self.model.setObjective(0.)
        self.model.optimize()

        # ensure new problem is actually infeasible
        assert self.model.status == 3

        # reset quadratic objective
        self.model.setObjective(obj)

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
        if self.model.status == 2:
            assert np.isclose(self.model.objVal, solution['objective'])
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
                solution['lambda'][t] - \
                self.mld.A.T.dot(solution['lambda'][t+1]) + \
                self.mld.F.T.dot(solution['mu'][t])
            if t == self.T-1 and self.X_T is not None:
                residuals += self.mld.A.T.dot(self.X_T[0].T.dot(solution['mu'][self.T]))
            assert np.linalg.norm(residuals) < tol

        # check stationarity wrt x_N
        residuals = self.C_T.T.dot(solution['rho'][self.T]) + solution['lambda'][self.T] 
        assert np.linalg.norm(residuals) < tol

        # test stationarity wrt u_t
        for t in range(self.T):
            residuals = self.D.T.dot(solution['sigma'][t]) - \
                self.mld.B.T.dot(solution['lambda'][t+1]) + \
                self.mld.G.T.dot(solution['mu'][t]) + \
                self.mld.V.T.dot(solution['nu_ub'][t] - solution['nu_lb'][t])
            if t == self.T-1 and self.X_T is not None:
                residuals += self.mld.B.T.dot(self.X_T[0].T.dot(solution['mu'][self.T]))
            assert np.linalg.norm(residuals) < tol

    def _evaluate_dual_solution(self, x0, identifier, solution):
        '''
        Given a dual solution, returns it cost.

        Parameters
        ----------
        x0 : np.array
            Initial state of the system.
        identifier : dict
            Dictionary containing the values for some of the binaries.
            Pass an empty dictionary to reset the bounds of the binaries to 0 and 1.
        solution : dict
            Dictionary containing the dual solution of the convex subproblem.
            Each one of these is a list (ordered in time) of numpy arrays.

        Returns
        -------
        obj : float
            Dual objective associated with the given dual solution.
        '''

        # evaluate quadratic terms
        objective = - .25 * np.linalg.norm(solution['rho'][self.T])**2
        for t in range(self.T):
            objective -= .25 * np.linalg.norm(solution['rho'][t])**2
            objective -= .25 * np.linalg.norm(solution['sigma'][t])**2

        # evaluate linear terms
        v_lb, v_ub = self._get_bounds_on_binaries(identifier)
        objective -= x0.dot(solution['lambda'][0])
        for t in range(self.T):
            objective -= self.mld.h.dot(solution['mu'][t])
            objective += v_lb[t].dot(solution['nu_lb'][t])
            objective -= v_ub[t].dot(solution['nu_ub'][t])
        if self.X_T is not None:
            objective -= self.X_T[1].dot(solution['mu'][self.T])

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
            v_lb[k[1]][k[2]] = v
            v_ub[k[1]][k[2]] = v

        return v_lb, v_ub

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
            (See documentation of the method solve_subproblem.)
            None if problem is infeasible.
        optimal_leaves : list of instances of Node
            Leaves of the branch and bound tree that proved optimality of the returned solution. 
        '''

        # generate a solver for the branch and bound algorithm
        def solver(identifier):
            solution = self.solve_subproblem(x0, identifier)
            feasible = not np.isinf(solution['objective'])
            return feasible, solution['integer_feasible'], solution['objective'], solution

        # solve the mixed integer program using branch and bound
        return branch_and_bound(
            solver,
            best_first,
            self.explore_in_chronological_order,
            **kwargs
        )

    def explore_in_chronological_order(self, identifier):
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
        if index_u < self.mld.nub:
            branches = [{(t,index): 0.}, {(t,index): 1.}]

        # if everything is fixed at time t, move to time t+1
        else:
            branches = [{(t+1,0): 0.}, {(t+1,0): 1.}]

        return branches

    @staticmethod
    def _propagate_dual_solution(old_solution):
        '''
        '''

        # copy the old solution
        solution = deepcopy(old_solution)

        # for some of the dual variables delete first and add zero
        for l in ['lambda', 'nu_lb', 'nu_ub', 'rho', 'sigma']:
            solution[l].append(0.*solution[l][-1])
            del solution[l][0]

        # rho
        solution['rho'][-2] = self.update_rho.dot(solution['rho'][-2])

        # terminal constraints
        mu_extend = [
            self.update_mu[0].dot(solution['mu'][-2]) + self.update_mu[1].dot(solution['mu'][-1]),
            0.*solution['mu'][-2],
            0.*solution['mu'][-1]
            ]
        for i in [0, -1, -1]:
            del solution['mu'][i]
        solution['mu'].extend(mu_extend)

        # evaluate new solution
        solution['objective'] = self._evaluate_dual_solution(solution)

        return solution

    def construct_warm_start(self, leaves, x0, u0, e0):

        # needed for a (redundant) check
        x1 = self.mld.A.dot(x0) + self.mld.B.dot(u0) + e0

        # initialize nodes for warm start
        warm_start = []

        # check each on of the optimal leaves
        for l in leaves:

            # if the identifier of the leaf does not agree with the stage_variables drop the leaf
            if self._retain_leaf(l.identifier, u0):
                identifier = self._get_new_identifier(l.identifier)
                solution = {}

                # propagate lower bounds if leaf is feasible
                if l.feasible() in [True, None]:
                    feasible = None
                    solution['dual_solution'] = 
                    lower_bound = l.objective + sum(lam)
                    extra_data['objective_dual'] = lower_bound
                    extra_data['dual'] = self._propagate_dual_solution(l.extra_data['dual_solution'])
                    
                    solution['dual_solution'] = self._propagate_dual_solution(l.solution['dual_solution'])
                    solution['objective'] = self._evaluate_dual_solution(x0, identifier, solution['dual_solution'])

                else:

                    # propagate infeasibility if leaf is still infeasible
                    feasible = False
                    lam = self._get_lambdas(l.identifier, l.extra_data['farkas_proof'], stage_variables)
                    if stage_variables['e_0'].dot(l.extra_data['farkas_proof']['alpha'][1]) < l.extra_data['farkas_objective'] + lam[2]:
                        lower_bound = np.inf
                        extra_data['farkas_objective'] = l.extra_data['farkas_objective'] + lam[2] + lam[4]
                        extra_data['farkas_proof'] = self._propagate_dual_solution(l.extra_data['farkas_proof'])

                    # if potentially feasible
                    else:
                        lower_bound = - np.inf

                # add new node to the list for the warm start
                warm_start.append(Node(None, identifier, feasible, lower_bound, extra_data))

        return warm_start

    @staticmethod
    def _retain_leaf(identifier, u0):
        '''
        '''

        # retain until proven otherwise
        retain = True

        # loop over the elements of the identifier and check if they agree with stage variables at time zero
        for k, v in identifier.items():
            if k[0] == 0 and not np.isclose(v, u0[k[1]]):
                retain = False
                break

        return retain

    @staticmethod
    def _get_new_identifier(identifier):
        return {(k[0]-1, k[1]): v for k, v in identifier.items() if k[0] > 0}

############

class SubproblemSolution(object):

    def __init__(self, primal, dual, integer_feasible):

        self.primal = primal
        self.dual = dual
        self.integer_feasible = integer_feasible

    @staticmethod
    def from_controller(controller):

        primal = SubproblemPrimalSolution.from_controller(controller)
        dual = SubproblemDualSolution.from_controller(controller)

        integer_feasible = ???

        return SubproblemSolution(primal, dual, integer_feasible)


class SubproblemPrimalSolution(object):

    def __init__(self, variables, objective):

        # store primal variables
        self.x = variables['x']
        self.uc = variables['uc']
        self.ub = variables['ub']

        # store primal objective
        self.objective = objective

    @staticmethod
    def from_controller(controller):

        # initialize dictionary of primal variables
        variables = {}

        # if solved to optimality
        if controller.model.status == 2:

            # get minimizer primal solution
            variables['x'] = [controller.model.get_minimizer('x_%d'%t) for t in range(controller.T+1)]
            variables['uc'] = [controller.model.get_minimizer('uc_%d'%t) for t in range(controller.T)]
            variables['ub'] = [controller.model.get_minimizer('ub_%d'%t) for t in range(controller.T)]

            # optimla value
            objective = controller.model.objVal

        # if primal infeasible
        else:
            variables = None
            objective = None

        return SubproblemPrimalSolution(variables, objective)

class SubproblemDualSolution(object):

    def __init__(self, variables, objective):

        # store dual variables
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

        # initialize dictionary of dual variables
        variables = {
            'lam': [],
            'mu': [],
            'nu_lb': [],
            'nu_ub': [],
            'rho': [],
            'sigma': []
            }

        for k in ['lam', 'mu']:
            for t in range(self.T+1):
                c = self.model.get_constraints('%s_%d'%(l,t))

        for t in range(self.T+1):


        # loop through the constraints
        for l in solution.keys:
            for t in range(self.T+1):
                c = self.model.get_constraints('%s_%d'%(l,t))
                if c.size > 0: # some multipliers do not appear at stage T

                    # get optimal duals or farkas proof if infeasible
                    # gurobi gives negative multipliers and positive farkas solutions!
                    if self.model.status == 2:
                        solution[l].append(- np.array([ci.Pi for ci in c]))
                    else:
                        solution[l].append(np.array([ci.FarkasDual for ci in c]))

        # state performance output
        solution['rho'] = []
        for t in range(self.T+1):
            C = self.C if t < self.T else self.C_T
            if self.model.status == 2:
                x = self.model.get_variables('x_%d'%t)
                x = np.array([xi.x for xi in x])
                solution['rho'].append[2*C.dot(x)]
            else:
                solution['rho'].append[np.zeros(C.size[0])]

        # input performance output
        solution['sigma'] = []
        for t in range(self.T):
            if self.model.status == 2:
                u = self.model.get_variables('u_%d'%t)
                u = np.array([ui.u for ui in u])
                solution['sigma'].append[2*self.D.dot(u)]
            else:
                solution['sigma'].append[np.zeros(self.D.size[0])]

        return solution



    @staticmethod
    def from_shiting(???):

        
