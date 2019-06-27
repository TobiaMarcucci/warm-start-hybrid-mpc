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

        super(BoundedQP, self).__init__(**kwargs)

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

        # change the default lower bound to -inf
        # since by default gurobi uses 0
        if not kwargs.has_key('lb'):
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

    def optimize(self):
        '''
        Solves the optimization problem.
        If the problem is infeasible, retrieves a Farkas' proof solving a linear program.
        After using this method, if the problem is infeasible, it is possible to retrive the variables FarkasDual in the gurobi constraint class.
        '''

        # reset model (gurobi does not try to use the last solve to warm start)
        self.reset()

        # run the optimization
        super(BoundedQP, self).optimize()

        # if not optimal then infeasible, do Farkas proof
        if self.status != 2:
        
            # copy objective
            obj = self.getObjective()

            # rerun the optimization with linear objective
            # (only linear accepted for farkas proof)
            self.setObjective(0.)
            super(BoundedQP, self).optimize()

            # ensure new problem is actually infeasible
            if self.status != 3:
                raise ValueError('The problem seems to be unbounded.')

            # reset quadratic objective
            self.setObjective(obj)

    def get_primal_optimizer(self, name):
        '''
        Gets the optimal value of a set of primal variables and returns them in a numpy array.

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

        # check that problem has been solved
        if self.status == 1:
            raise ValueError('Problem not solved yet.')

        # if optimal, return optimizer
        elif self.status == 2:
            return np.array([xi.x for xi in self.get_variables(name)])

        # if infeasible (or others)
        else:
            return None

    def get_dual_optimizer(self, name):
        '''
        Gets the optimal value of a set of dual variables and returns them in a numpy array.
        Multipliers are positive for inequalities with sense '<', and negative with sense '>'.

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

        # check that problem has been solved
        if self.status == 1:
            raise ValueError('Problem not solved yet.')

        # if optimal, return optimizer
        elif self.status == 2:
            return - np.array([ci.Pi for ci in self.get_constraints(name)])

        # if certified infeasible, return Farkas dual
        # note that Gurobi Pi and FarkasDual of opposite sign!
        else:
            return np.array([ci.FarkasDual for ci in self.get_constraints(name)])