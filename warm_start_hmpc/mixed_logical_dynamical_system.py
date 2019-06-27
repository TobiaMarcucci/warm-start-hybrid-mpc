# external imports
import numpy as np
import sympy as sp

class MixedLogicalDynamicalSystem(object):
    '''
    Discrete-time Mixed Logical Dynamical (MLD) system in the form
    x(t+1) = A x(t) + B u(t)
    F x(t) + G u(t) <= h
    where:
    - x(t) continuous states at time t
    - u(t) continuous and binary inputs at time t
    '''

    def __init__(self, nub, dynamics, constraints):
        '''
        Initializes the MLD system.

        Parameters
        ----------
        nub : int
            Number of binary inputs.
        dynamics : list of np.array
            Matrices A, B from the dynamics.
        constraints : list of np.array
            Matrices F, G, h from the inequality constraints.
        '''

        # store sizes of the system
        self.nx = A.shape[1]
        self.nu = B.shape[1]
        self.nub = nub
        self.nuc = self.nu - nub

        # store matrices
        [self.A, self.B] = dynamics
        [self.F, self.G, self.h] = constraints

        # selection matrix for the binaries
        self.V = np.hstack((np.zeros((nub, self.nuc)), np.eye(nub)))

        # check size of the input matrices
        self._check_input_sizes()

    def _check_input_sizes(self):
        '''
        Checks the size of the matrices passed as inputs in the initialization of the class.
        '''

        # check dynamics
        assert self.A.shape[0] == self.A.shape[1]
        assert self.B.shape[0] == self.nx

        # check constraints
        assert self.F.shape == [self.h.size, self.nx]
        assert self.G.shape == [self.h.size, self.nu]

    @staticmethod
    def from_symbolic(variables, nub, dynamics, constraints):
        '''
        Instatiates a MixedLogicalDynamicalSystem starting from the symbolic value of the dynamics and the constraints.

        Parameters
        ----------
        nub : int
            Number of binary inputs.
        variables : dict of sympy matrix filled with sympy symbols
            Symbolic variables of the system, keys: 'xc', 'xb', 'uc', 'ub'.
        dynamics : sympy matrix of sympy symbolic linear expressions
            Symbolic value of the next state (linear function of xc, xb, uc, ub).
        constraints : sympy matrix of sympy symbolic affine expressions
            Symbolic constraints in the form constraints <= 0.
            Hence, constraints = F x + G u - l.

        Returns
        -------
        instance of MixedLogicalDynamicalSystem
            MLD system extracted from the symbolic expressions.
        '''

        # collect variables
        v_labels = ['x', 'u']
        v = sp.Matrix([variables[l] for l in v_labels])
        blocks = [variables[l].shape[0] for l in v_labels]

        # dynamics
        lhs, offest = sym2matrices(v, dynamics)
        A, B = unpack_bmat(lhs, blocks, 'h')
        assert np.allclose(offest, np.zeros(offest.size))
        dynamics = [A, B]

        # constraints
        lhs, offest = sym2matrices(v, constraints)
        F, G = unpack_bmat(lhs, blocks, 'h')
        constraints = [F, G, -offest]

        # construct MLD system
        return MixedLogicalDynamicalSystem(nub, dynamics, constraints)


def sym2matrices(x, expr):
    '''
    Extracts from the symbolic affine expression the matrices such that expr(x) = A x + b.

    Arguments
    ---------
    x : sympy matrix of sympy symbols
        Variables of the affine expression.
    expr : sympy matrix of sympy symbolic affine expressions
        Left hand side of the inequality constraint.

    Returns
    -------
    lhs : np.array
        Jacobian of the affine expression.
    offset : np.array
        Offset term of the affine expression.
    '''

    # left hand side
    lhs = np.array(expr.jacobian(x)).astype(np.float64)

    # offset term
    offset = np.array(expr.subs({xi:0 for xi in x})).astype(np.float64).flatten()
    
    return lhs, offset

def unpack_bmat(A, indices, direction):
    '''
    Unpacks a matrix in blocks.

    Arguments
    ---------
    A : np.array
        Matrix to be unpacked.
    indices : list of int
        Set of indices at which the matrix has to be cut.
    direction : string
        'h' to unpack horizontally and 'v' to unpack vertically.

    Returns
    -------
    blocks : list of np.array
        Blocks extracted from the matrix A.
    '''

    # initialize blocks
    blocks = []

    # unpack
    i = 0
    for j in indices:

        # horizontally
        if direction == 'h':
            blocks.append(A[:,i:i+j])

        # vertically
        elif direction == 'v':
            blocks.append(A[i:i+j,:])

        # raise error if uknown key
        else:
            raise ValueError('unknown direction ' + direction)

        # increase index by j
        i += j

    return blocks