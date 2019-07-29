# external imports
import numpy as np
import sympy as sp
from scipy.linalg import block_diag

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

    @staticmethod
    def from_pwa(dynamics, domains):
        '''
        Translates a piecewise-affine (PWA) system in a mixed logical dynamical (MLD) system.
        Uses the convex hull method.
        Continuous auxiliary variables are ordered as (x1, ..., xI, u1, ..., uI).
        Binary auxiliary variables are (mu1, ..., muI).

        Parameters
        ----------
        dynamics : list of np.array
            Matrices Ai, Bi, ci from the PWA dynamics.
        domains : list of np.array
            Matrices Fi, Gi, hi from the inequality constraints.
        '''

        # system size
        nm = len(dynamics)
        nx = dynamics[0][0].shape[0]
        nu = dynamics[0][1].shape[1]
        nc = sum(d[0].shape[0] for d in domains)

        # reshape 1d vectors
        # change sign to the rhs of the domains to simplify the code below
        for i in range(nm):
            dynamics[i][2] = dynamics[i][2].reshape((dynamics[i][2].size, 1))
            domains[i][2] = - domains[i][2].reshape((domains[i][2].size, 1))

        # MLD dynamics
        A = np.zeros((nx, nx))
        B = np.hstack([np.zeros((nx, 1))] + [d[k] for k in range(3) for d in dynamics])

        # MLD constraints
        F0 = np.zeros((nc, nx))
        G0 = np.hstack((
            np.zeros((nc, nu)),
            np.hstack([block_diag(*[d[k] for d in domains]) for k in range(3)])
            ))
        h0 = np.zeros(nc)

        # reconstruct x
        Fx = np.eye(nx)
        Gx = np.hstack((
            np.zeros((nx, nu)),
            np.hstack([-np.eye(nx)]*nm),
            np.zeros((nx, (nu+1)*nm)),
            ))
        hx = np.zeros(nx)

        # reconstruct u
        Fu = np.zeros((nu, nx))
        Gu = np.hstack((
            np.eye(nu),
            np.zeros((nu, nx*nm)),
            np.hstack([-np.eye(nu)]*nm),
            np.zeros((nu, nm))
            ))
        hu = np.zeros(nu)

        # reconstruct mu
        Fmu = np.zeros((1, nx))
        Gmu = np.hstack((
            np.zeros((1, nu+(nx+nu)*nm)),
            np.ones((1, nm))
            ))
        hmu = np.ones(1)

        # assemble mld constraints
        F = np.vstack((F0, Fx, -Fx, Fu, -Fu, Fmu, -Fmu,))
        G = np.vstack((G0, Gx, -Gx, Gu, -Gu, Gmu, -Gmu))
        h = np.concatenate((h0, hx, -hx, hu, -hu, hmu, -hmu))

        return MLDSystem([A, B], [F, G, h], nm)

    @staticmethod
    def from_symbolic_pwa(dynamics_sym, domains_sym, x, u):
        '''
        Parameters
        ----------
        dynamics_sym : list of sympy matrices of sympy symbolic linear expressions
            Symbolic value of the next state for each mode (affine function of x and u).
        domains : list of sympy matrices of sympy symbolic linear expressions
            Domain in the (x,u) space of each mode, expressed as domains[i] <= 0.
        '''

        # collect variables
        v = sp.Matrix([x, u])
        blocks = [x.shape[0], u.shape[0]]

        # dynamics
        dynamics = []
        for d in dynamics_sym:
            lhs, c = sym2mat(v, d)
            A, B = unpack_bmat(lhs, blocks, 'h')
            dynamics.append([A, B, c])

        # domains
        domains = []
        for d in domains_sym:
            lhs, h = sym2mat(v, d)
            F, G = unpack_bmat(lhs, blocks, 'h')
            domains.append([F, G, -h])

        # construct MLD system
        return MLDSystem.from_pwa(dynamics, domains)