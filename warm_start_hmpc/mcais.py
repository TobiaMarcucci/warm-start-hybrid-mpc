# external imports
import numpy as np
from copy import copy
from operator import le, ge, eq
from scipy.linalg import solve_discrete_are

# internal inputs
from warm_start_hmpc.bounded_qp import BoundedQP

def solve_dare(A, B, Q, R):
    '''
    Returns the solution of the Discrete Algebraic Riccati Equation (DARE).
    Consider the linear quadratic control problem V*(x(0)) = min_{x(.), u(.)} 1/2 sum_{t=0}^inf x'(t) Q x(t) + u'(t) R u(t) subject to x(t+1) = A x(t) + B u(t).
    The optimal solution is u(0) = K x(0) which leads to V*(x(0)) = 1/2 x'(0) P x(0).
    The pair A, B is assumed to be controllable.

    Parameters
    ----------
    A : numpy.ndarray
        State to state matrix in the linear dynamics.
    B : numpy.ndarray
        Input to state matrix in the linear dynamics.
    Q : numpy.ndarray
        Quadratic cost for the state (positive semidefinite).
    R : numpy.ndarray
        Quadratic cost for the input (positive definite).

    Returns
    -------
    P : numpy.ndarray
        Hessian of the cost-to-go (positive definite).
    K : numpy.ndarray
        Optimal feedback gain matrix.
    '''

    # cost to go
    P = solve_discrete_are(A, B, Q, R)

    # feedback
    K = - np.linalg.inv(B.T.dot(P).dot(B)+R).dot(B.T).dot(P).dot(A)

    return P, K

def mcais(A, D, e, verbose=False):
    '''
    Returns the maximal constraint-admissible (positive) invariant set O_inf for the system x(t+1) = A x(t) subject to the constraint x in X := {x | D x <= e}.
    O_inf is also known as maximum output admissible set.
    It holds that x(0) in O_inf <=> x(t) in X for all t >= 0.
    (Implementation of Algorithm 3.2 from: Gilbert, Tan - Linear Systems with State and Control Constraints, The Theory and Application of Maximal Output Admissible Sets.)
    Sufficient conditions for this set to be finitely determined (i.e. defined by a finite number of facets) are: A stable, X bounded and containing the origin.

    Math
    ----
    At each time step t, we want to verify if at the next time step t+1 the system will go outside X.
    Let's consider X := {x | D_i x <= e_i, i = 1,...,n} and t = 0.
    In order to ensure that x(1) = A x(0) is inside X, we need to consider one by one all the constraints and for each of them, the worst-case x(0).
    We can do this solving an LP
    V(t=0, i) = max_{x in X} D_i A x - e_i for i = 1,...,n
    if all these LPs have V < 0 there is no x(0) such that x(1) is outside X.
    The previous implies that all the time-evolution x(t) will lie in X (see Gilbert and Tan).
    In case one of the LPs gives a V > 0, we iterate and consider
    V(t=1, i) = max_{x in X, x in A X} D_i A^2 x - e_i for i = 1,...,n
    where A X := {x | D A x <= e}.
    If now all V < 0, then O_inf = X U AX, otherwise we iterate until convergence
    V(t, i) = max_{x in X, x in A X, ..., x in A^t X} D_i A^(t+1) x - e_i for i = 1,...,n
    Once at convergence O_Inf = X U A X U ... U A^t X.

    Arguments
    ---------
    A : numpy.ndarray
        State transition matrix.
    D : numpy.ndarray
        Left hand side of the constraints.
    e : numpy.ndarray
        Left hand side of the constraints.
    verbose : bool
        If True prints at each iteration the convergence parameters.

    Returns
    -------
    D_inf : numpy.ndarray
        Left hand side of the maximal constraint-admissible (positive) ivariant.
    e_inf : numpy.ndarray
        Right hand side of the maximal constraint-admissible (positive) ivariant.
    t : int
        Determinedness index.
    '''

    # ensure convergence of the algorithm
    eig_max = np.max(np.absolute(np.linalg.eig(A)[0]))
    if eig_max > 1.:
        raise ValueError('Unstable system, cannot derive maximal constraint-admissible set.')
    if np.min(e) < 0.:
        raise ValueError('The origin is not in the constraint set, cannot derive maximal constraint-admissible set.')

    # initialize mcais
    D_inf = copy(D)
    e_inf = copy(e)
    [nc, nx] = D.shape

    # loop over time
    t = 1
    convergence = False
    while not convergence:

        # solve one LP per facet
        J = D.dot(np.linalg.matrix_power(A,t))
        lp = BoundedQP()
        x = lp.add_variables(nx)
        lp.add_constraints(D_inf.dot(x), le, e_inf)
        residuals = []

        for i in range(D.shape[0]):
            lp.setObjective(- J[i].dot(x))
            lp.optimize()
            residuals.append(- lp.primal_objective() - e[i])

        # print status of the algorithm
        if verbose:
            print(f'Time horizon: {t}.', end=' ')
            print(f'Convergence index: {max(residuals)}.', end=' ')
            print(f'Number of facets: {D_inf.shape[0]}.', end='\r')

        # convergence check
        new_facets = [i for i, r in enumerate(residuals) if r > 0.]
        if len(new_facets) == 0:
            convergence = True
        else:

            # add (only non-redundant!) facets
            D_inf = np.vstack((D_inf, J[new_facets]))
            e_inf = np.concatenate((e_inf, e[new_facets]))
            t += 1

   # remove redundant facets
    if verbose:
        print('\nMaximal constraint-admissible invariant set found.', end=' ')
        print('Removing redundant facets ...', end=' ')

    D_inf, e_inf = remove_redundant_inequalities(D_inf, e_inf)
    if verbose:
        print(f'minimal facets are {D_inf.shape[0]}.')

    return D_inf, e_inf

def remove_redundant_inequalities(E, f, tol=1.e-7):
    '''
    Computes the indices of the facets that generate a minimal representation of the polyhedron solving an LP for each facet of the redundant representation.
    (See "Fukuda - Frequently asked questions in polyhedral computation" Sec.2.21.)
    In case of equalities, first the problem is projected in the nullspace of the equalities.
    Arguments
    ----------
    tol : float
        Minimum distance of a redundant facet from the interior of the polyhedron to be considered as such.
    Returns
    ----------
    minimal_facets : list of int
        List of indices of the non-redundant inequalities A x <= b (None if the polyhedron in empty).
    '''

    # initialize list of non-redundant facets
    [nc, nx] = E.shape
    minimal_facets = list(range(nc))

    # initialize lp
    lp = BoundedQP()
    x = lp.add_variables(nx)
    lp.add_constraints(E.dot(x), le, f, name='c')

    # check each facet
    for i in range(nc):

        # remove redundant facets and relax ith inequality
        f_add = np.zeros(f.size)
        f_add[i] += 1.
        lp.set_constraint_rhs('c', f + f_add)
        lp.setObjective(-E[i].dot(x))
        lp.optimize()

        # remove redundant facets from the list
        if  - lp.primal_objective() - f[i] < tol:
            minimal_facets.remove(i)

    return E[minimal_facets], f[minimal_facets]
