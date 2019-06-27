# external imports
import numpy as np
import sympy as sp

# internal inputs
from warm_start_hmpc.utils import sym2mat, unpack_bmat

class MLDSystem(object):
    '''
    Discrete-time Mixed Logical Dynamical (MLD) system in the form
    x(t+1) = A x(t) + B u(t)
    F x(t) + G u(t) <= h
    where:
    x(t) continuous states at time t,
    u(t) continuous and binary inputs at time t.
    '''

    def __init__(self, dynamics, constraints, nub):
        '''
        Initializes the MLD system.

        Parameters
        ----------
        dynamics : list of np.array
            Matrices A, B from the dynamics.
        constraints : list of np.array
            Matrices F, G, h from the inequality constraints.
        nub : int
            Number of binary inputs.
        '''

        # store matrices
        [self.A, self.B] = dynamics
        [self.F, self.G, self.h] = constraints

        # store sizes of the system
        self.nx = self.A.shape[1]
        self.nu = self.B.shape[1]
        self.nub = nub
        self.nuc = self.nu - nub

        # selection matrix for the binaries
        self.V = np.hstack((np.zeros((nub, self.nuc)), np.eye(nub)))

        # check size of the input matrices
        self._check_input_sizes()

    def _check_input_sizes(self):
        '''
        Checks the size of the matrices passed as inputs in the initialization of the class.
        '''

        # check dynamics
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError('Nonsquare A matrix.')
        if self.B.shape[0] != self.nx:
            raise ValueError('A and B matrices have incompatible size.')

        # check constraints
        if self.F.shape != (self.h.size, self.nx):
            raise ValueError('Matrix F has incompatible size.')
        if self.G.shape != (self.h.size, self.nu):
            raise ValueError('Matrix G has incompatible size.')

    @staticmethod
    def from_symbolic(dynamics, constraints, x, u, nub):
        '''
        Instatiates a MLDSystem starting from the symbolic value of the dynamics and the constraints.

        Parameters
        ----------
        dynamics : sympy matrix of sympy symbolic linear expressions
            Symbolic value of the next state (linear function of x and u).
        constraints : sympy matrix of sympy symbolic affine expressions
            Symbolic constraints in the form constraints <= 0.
            Hence, constraints = F x + G u - h.
        x : sympy matrix
            Symbolic state of the system.
        u : sympy matrix
            Symbolic input of the system.
        nub : int
            Number of binary inputs.

        Returns
        -------
        instance of MLDSystem
            MLD system extracted from the symbolic expressions.
        '''

        # collect variables
        v = sp.Matrix([x, u])
        blocks = [x.shape[0], u.shape[0]]

        # dynamics
        lhs, offest = sym2mat(v, dynamics)
        A, B = unpack_bmat(lhs, blocks, 'h')
        dynamics = [A, B]
        if not np.allclose(offest, np.zeros(offest.size)):
            raise ValueError('The dynamics seems to be affine and not linear.')

        # constraints
        lhs, offest = sym2mat(v, constraints)
        F, G = unpack_bmat(lhs, blocks, 'h')
        constraints = [F, G, -offest]

        # construct MLD system
        return MLDSystem(dynamics, constraints, nub)