# external imports
import numpy as np
from sympy import symbols, Matrix, sin, cos
from sympy.utilities import lambdify
from copy import copy

# internal imports
from warm_start_hmpc.mld_system import MLDSystem
from warm_start_hmpc.utils import sym2mat, unpack_bmat
# from warm_start_hmpc.mcais import mcais, solve_dare
from warm_start_hmpc.bounded_qp import BoundedQP

class CartPoleWithWalls(object):

    def __init__(self):

        # physical parameters
        self.mc = 1. # mass cart
        self.mp = 1. # mass pole
        self.l  = 1. # length pole
        self.d  = .5 # distance walls from origin
        self.eps = .5 # coefficient of restitution
        self.h = 1./20. # nominal integration step
        self.g = Matrix([0., -10]) # gravity acceleration

        # bounds
        self.x_max = np.array([self.d, np.pi/8., 2., 1.])
        self.x_min = - self.x_max
        self.fc_max = 2.
        self.fc_min = - self.fc_max

        # dimensions
        self.nq = 2
        self.nv = 2
        self.nx = self.nq + self.nv
        self.nu = 1
        self.nf = 3

        # symbolic state
        [self.qc, self.qp, self.vc, self.vp] = symbols('q_c q_p v_c v_p')
        self.q = Matrix([self.qc, self.qp])
        self.v = Matrix([self.vc, self.vp])
        self.x = Matrix([self.q, self.v])

        # symbolic forces
        [self.fc, self.fl, self.fr] = Matrix([symbols('f_c f_l f_r')])
        self.f = Matrix([self.fc, self.fl, self.fr])

        # binary indicators (= 1 if contact)
        [self.cl, self.cr] = symbols('c_l c_r')

        # symbolic integration step
        self.h_sym = symbols('h')

        # nominal parameters
        self.nominal = {
            self.h_sym: self.h,
            **{xi: 0. for xi in self.x},
            **{fi: 0. for fi in self.f},
            }

        # systems model
        self._kinematics()
        self._dynamics()

        # compute big Ms for the MLD constraints
        [self.M_phi, self.M_phi_dot, self.M_fw] = self._bigMs()

        # construct MLD system
        c = Matrix([self.cl, self.cr])
        self.mld = MLDSystem.from_symbolic(
          self.x_next_lin,
          Matrix(self._mld_constraints()),
          self.x,
          Matrix([self.f, c]),
          c.shape[0]
          )

    def _kinematics(self):

        # positions
        self.pc = Matrix([self.qc])
        self.pp = Matrix([
            self.qc - self.l*sin(self.qp),
            self.l*cos(self.qp)
            ])

        # velocities
        self.vc = self.pc.jacobian(self.q)*self.v
        self.vp = self.pp.jacobian(self.q)*self.v

        # gap functions
        self.phi = {
            'l': Matrix([self.d + self.pp[0]]),
            'r': Matrix([self.d - self.pp[0]])
        }
        self.phi_fun = {k: lambdify([self.q], v[0,0], 'numpy') for k, v in self.phi.items()}
        self.phi_lin = {k: self.linearize(self.phi[k], [self.q]) for k in self.phi.keys()}

        # jacobians
        self.Phi = {k: v.jacobian(self.q) for k, v in self.phi.items()}
        self.Phi_nom = {k: v.subs(self.nominal) for k, v in self.Phi.items()}

    def _dynamics(self):

        # Lagrangian function
        potential = - self.mp*(self.g.T*self.pp)
        kinetic = .5 * self.mc * (self.vc.T * self.vc) + \
                  .5 * self.mp * (self.vp.T * self.vp)
        lagrangian = kinetic - potential

        # mass matrix
        M = lagrangian.jacobian(self.v).T.jacobian(self.v)
        M_inv = M.inv()

        # Coriolis, centrifugal, and gravitational terms
        tau = - lagrangian.jacobian(self.v).T.jacobian(self.q) * self.v + \
                lagrangian.jacobian(self.q).T

        # generalized forces
        Qc = self.pc.jacobian(self.q).T * self.fc
        Qw = self.pp[0,:].jacobian(self.q).T * (self.fl - self.fr)

        # nonlinear dynamics
        v_next = self.v + M_inv * (self.h_sym * (tau + Qc) + Qw)
        q_next = self.q + self.h_sym * v_next
        x_next = Matrix([q_next, v_next])
        self.x_next_fun = lambdify([self.x, self.f, self.h_sym], x_next, 'numpy')

        # linearized dynamics
        self.x_next_lin = self.linearize(x_next, [self.x, self.f])
        self.q_next_lin = self.x_next_lin[:self.nq, :]
        self.v_next_lin = self.x_next_lin[self.nq:, :]

        # contact forces with wall
        v_next_0 = v_next.subs({self.fl: 0., self.fr: 0.})
        M_reduced = {k: (v * M_inv * v.T).inv() for k, v in self.Phi.items()}
        f_solved = {k: - M_reduced[k] * self.Phi[k] * (self.eps * self.v + v_next_0) for k in self.phi.keys()}
        self.f_solved_fun = {k: lambdify([self.x, self.fc, self.h_sym], v[0,0], 'numpy') for k, v in f_solved.items()}

    def _bigMs(self):

        M_phi = 2. * self.d + self.x_max[2] + .2

        M_phi_dot = (1+self.eps)*(self.x_max[2] - self.l * self.x_min[3])

        M_fw = 30.

        return M_phi, M_phi_dot, M_fw

    def _mld_constraints(self):

        # collect all the MLD constraints
        constraints = []

        # state bounds
        constraints.append(self.x - self.x_max.reshape(self.nx, 1))
        constraints.append(self.x_min.reshape(self.nx, 1) - self.x)

        # input bounds
        constraints.append(Matrix([self.fc - self.fc_max]))
        constraints.append(Matrix([self.fc_min - self.fc]))

        # organize forces and inidcators
        f = {'l': self.fl, 'r': self.fr}
        c = {'l': self.cl, 'r': self.cr}

        # indicator binary equal one => collision at next time step
        q_next_lin_0 = self.q_next_lin.subs({self.fl: 0., self.fr: 0.})
        self.phi_next_lin = {k: self.Phi_nom[k] * q_next_lin_0 + self.phi_lin[k].subs(self.nominal) for k in self.phi.keys()}
        for k, v in self.phi_next_lin.items():
            indicator = Matrix([self.M_phi * (1 - c[k])])
            constraints.append(v - indicator)
            indicator = Matrix([10. * c[k]])
            constraints.append(- v - indicator)

        # collision at next time step => restitution in the gap velocity
        dv = self.v_next_lin + self.eps*self.v
        for k in self.phi.keys():
            indicator = Matrix([self.M_phi_dot * (1 - c[k])])
            constraints.append(- self.Phi_nom[k] * dv - indicator)
            constraints.append(self.Phi_nom[k] * dv - indicator)

        # collision at next time step => positive contact force
        for k in self.phi.keys():
            constraints.append(Matrix([- f[k]]))
            constraints.append(Matrix([f[k] - self.M_fw * c[k]]))

        # redundant bounds to make the constraints bounded
        # the second could be made much stronger as cl + cr <= 1
        for v in c.values():
            constraints.append(Matrix([- v]))
            constraints.append(Matrix([v - 1.]))

        return constraints


    def linearize(self, expression, variables):

        # offset term
        linearization = expression.subs(self.nominal)

        # linear terms
        for v in variables:
            linearization += expression.jacobian(v).subs(self.nominal) * v

        return linearization

    def simulate(self, x0, u, dt, h_des=.001):

        # round desired discretization step
        T = int(dt/h_des)
        h = dt/T

        # state trajectory
        x = [x0]
        for t in range(T):

            # simulate without contact forces
            f = np.array([u, 0., 0.])
            x_next = self.x_next_fun(x[-1], f, h).flatten()
            q_next = x_next[:self.nq]

            # get contact forces if penetration
            for i, k in enumerate(self.phi.keys()):
                if self.phi_fun[k](q_next) <= 0.:
                    f[i+1] = self.f_solved_fun[k](x[-1], u, h)

            # simulate with contact forces
            x.append(self.x_next_fun(x[-1], f, h).flatten())
        
        return np.array(x), h


    def project_in_feasible_set(self, x):

        return np.minimum(np.maximum(x, self.x_min), self.x_max)