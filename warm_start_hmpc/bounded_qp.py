# external imports
import numpy as np
import gurobipy as grb

class BoundedQP(grb.Model):
    '''
    Quadratic program with bounded objective.
    It facilitates the process of adding (retrieving) multiple variables and constraints to (from) the optimization problem.
    It provides a proof of infeasibility when an infeasible QP is solved. 
    '''

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # change some default parameters
        self.Params.OutputFlag = 0
        self.Params.InfUnbdInfo = 1

    def add_variables(self, n, **kwargs):
        '''
        Adds n optimization variables to the problem.
        It stores the new variables in a numpy array so that they can be easily used for computations.

        Parameters
        ----------
        n : int
            Number of optimization variables to be added to the problem.

        Returns
        -------
        x : np.array
            Numpy array that collects the new optimization variables.
        '''

        # prevent from setting bounds here
        # otherwise retrieving the multipliers for the Farkas proof is impossible (?)
        if 'lb' in kwargs or 'ub' in kwargs:
            raise KeyError('Cannot set bounds with add_variables, use add_constraints instead.')

        # change the default lower bound to -inf, gurobi uses 0 by default
        kwargs['lb'] = [-grb.GRB.INFINITY]*n

        # add variables to the optimization problem
        x = self.addVars(n, **kwargs)

        # update model to make the new variables visible
        # this can inefficient but prevents bugs
        self.update()

        # return new variables in a numpy array
        return np.array(x.values())

    def get_variables(self, name):
        '''
        Gets a set of variables from the problem and returns them in a numpy array.

        Parameters
        ----------
        name : string
            Name of the family of variables we want to get from the problem.

        Returns
        -------
        x : np.array
            Numpy array that collects the asked variables.
            Returns an empty array if the variables is not defined.
        '''

        # initilize vector of variables
        x = np.array([])

        # there cannnot be more x than optimization variables
        for i in range(self.NumVars):

            # get new element and append
            xi = self.getVarByName(name+'[%d]'%i)
            if xi:
                x = np.append(x, xi)

            # if no more elements are available break the for loop
            else:
                break

        return x

    def add_constraints(self, x, op, y, **kwargs):
        '''
        Adds a constraint of the form x (<=, ==, or >=) y to the optimization problem.

        Parameters
        ----------
        x : np.array of gurobi.Var, or gurobi.LinExpr
            Left hand side of the constraint.
            Note that Gurobi assumes the lhs to not be a float and have some variables in it.
        op : python operator
            Either le (less than or equal to), ge (greater than or equal to), or eq (equal to).
        y : np.array of floats, gurobi.Var, or gurobi.LinExpr
            Right hand side of the constraint.

        Returns
        -------
        c : np.array of gurobi.Constr
            Numpy array that collects the new constraints.
        '''

        # ensure that the size of the lhs and the rhs match
        if len(x) != len(y):
            raise ValueError('Left- and right-hand side must have the same size.')

        # ensure that lhs is not an array of floats
        # otherwise gurobi the behavior of gurobi is unpredictable
        if all(isinstance(xi, float) for xi in x):
            raise ValueError('Left-hand side cannot be an array of floats.')

        # add linear constraints to the problem
        c = self.addConstrs((op(x[k], y[k]) for k in range(len(x))), **kwargs)

        # update model to make the new variables visible
        # this can inefficient but prevents bugs
        self.update()

        # organize the constraints in a numpy array
        c = np.array(c.values())

        return c

    def get_constraints(self, name):
        '''
        Gets a set of constraints from the problem and returns them in a numpy array.

        Parameters
        ----------
        name : string
            Name of the family of constraints we want to get from the problem.

        Returns
        -------
        c : np.array
            Numpy array that collects the asked constraints.
        '''

        # initilize vector of constraints
        c = np.array([])

        # there cannnot be more c than constraints in the problem
        for i in range(self.NumConstrs):

            # get new constraint and append
            ci = self.getConstrByName(name+'[%d]'%i)
            if ci:
                c = np.append(c, ci)

            # if no more constraints are available break the for loop
            else:
                break

        return c

    def set_constraint_rhs(self, name, rhs):
        '''
        Sets the right-hand side of a constraint to the given value.

        Parameters
        ----------
        name : string
            Name of the family of constraints of wich we want to reset the rhs.
        rhs : np.array
            New value of the right-hand side.
        '''

        # check input size
        c = self.get_constraints(name)
        if len(c) != len(rhs):
            raise ValueError('The rhs does not have the right dimension.')

        # reset rhs one by one
        for i, ci in enumerate(c):
            ci.RHS = rhs[i]

        # update model to see modifications
        self.update()

    def get_constraint_rhs(self, name):
        '''
        Gets the right-hand side of a constraint.

        Parameters
        ----------
        name : string
            Name of the family of constraints of wich we want to get the rhs.
        '''

        return np.array([ci.RHS for ci in self.get_constraints(name)])

    def optimize(self):
        '''
        Overloads the grb.Model method.
        If the problem is infeasible, retrieves a Farkas' proof solving a linear program.
        After using this method, if the problem is infeasible, it is possible to retrive the variables FarkasDual in the gurobi constraint class.
        '''

        # reset model (gurobi does not try to use the last solve to warm start)
        self.reset()

        # run the optimization
        super().optimize()

        # if not optimal then infeasible, do Farkas proof
        if self.status != 2:
        
            # copy objective
            obj = self.getObjective()

            # rerun the optimization with linear objective
            # (only linear accepted for farkas proof)
            self.setObjective(0.)
            super().optimize()

            # ensure new problem is actually infeasible
            if self.status != 3:
                raise AssertionError('The problem seems to be unbounded.')

            # reset quadratic objective
            self.setObjective(obj)

    def primal_optimizer(self, name):
        '''
        Gets the optimal value of a set of primal variables and returns them in a numpy array.
        Raises a RuntimeError if the problem has not been solved yet.
        Returns the optimal primal variables if the problem is solved to optimality.
        Returns None if the problem is infeasible.
        Note that, because of the optimize method, no other options are possible.

        Parameters
        ----------
        name : string
            Name of the family of primal variables of which we want to get the optimal value.

        Returns
        -------
        np.array
            Array that collects the optimal values of the primal variables.
            Returns the primal optimizer if the problem is solved to optimality.
            Returns None if infeasible.
        '''

        # if optimal, return optimizer
        self._throw_if_not_solved()
        if self.status == 2:
            return np.array([xi.x for xi in self.get_variables(name)])

        # if infeasible
        else:
            return None

    def dual_optimizer(self, name):
        '''
        Gets the optimal value of a set of dual variables and returns them in a numpy array.
        Multipliers are positive for inequalities with sense '<', and negative with sense '>'.
        Raises a RuntimeError if the problem has not been solved yet.
        Returns the optimal dual variables if the problem is solved to optimality.
        Returns None if the problem is infeasible.
        Note that, because of the optimize method, no other options are possible.
        
        Parameters
        ----------
        name : string
            Name of the family of dual variables of which we want to get the optimal value.

        Returns
        -------
        np.array
            Array that collects the optimal values of the dual variables.
            Returns the dual optimizer if the problem is solved to optimality.
            Returns the Farkas dual if the problem is certified to be infeasible.
        '''

        # if optimal, return optimizer
        self._throw_if_not_solved()
        if self.status == 2:
            return - np.array([ci.Pi for ci in self.get_constraints(name)])

        # if certified infeasible, return Farkas dual
        # note that Gurobi Pi and FarkasDual of opposite sign!
        else:
            return np.array([ci.FarkasDual for ci in self.get_constraints(name)])

    def primal_objective(self):
        '''
        Returns the optimal value of the primal problem.
        This is the optimal value of the problem if feasible.
        Returns inf if the problem is infeasible.

        Returns
        -------
        float or None
            Primal objective.
        '''

        # if optimal, return optimal value
        self._throw_if_not_solved()
        if self.status == 2:
            return self.objVal

        # if infeasible
        else:
            return np.inf

    def dual_objective(self):
        '''
        Returns the optimal value of the dual problem.
        This is the optimal value of the problem if feasible.
        This is the objective of the Farkas proof if infeasible.

        Returns
        -------
        float
            Dual objective.
        '''
        
        # if optimal, return optimal value
        self._throw_if_not_solved()
        if self.status == 2:
            return self.objVal

        # if infeasible, return cost of Farkas' proof
        else:
            return - sum(c.RHS*c.FarkasDual for c in self.getConstrs())

    def _throw_if_not_solved(self):
        '''
        Raises an RuntimeError error if the problem has not been solved yet.
        '''

        # if status is loaded only
        if self.status == 1:
            raise RuntimeError('Problem not solved yet.')