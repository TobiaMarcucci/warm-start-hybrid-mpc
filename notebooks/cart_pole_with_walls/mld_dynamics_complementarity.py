# external imports
import numpy as np
from sympy import symbols, Matrix, sin, cos
from sympy.utilities import lambdify
from operator import le, ge, eq

# internal imports
from warm_start_hmpc.mld_system import MLDSystem
from warm_start_hmpc.bounded_qp import BoundedQP

class CartPoleWithWalls(object):

    def __init__(self):

        # physical parameters
        self.mc = 1. # mass cart
        self.mp = 1. # mass pole
        self.l  = 1. # length pole
        self.d  = .5 # distance walls from origin
        self.eps = .5 # coefficient of restitution
        self.h = 1./10. # nominal integration step
        self.g = Matrix([0., -10]) # gravity acceleration

        # bounds
        self.x_max = np.array([self.d, np.pi/10., 1., 1.])
        self.x_min = - self.x_max
        self.fc_max = 1.2
        self.fc_min = - self.fc_max

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
        c = Matrix([self.cl, self.cr])
        u = Matrix([self.f, c])

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
        self.M_phi_next = self._bigM_phi_next()
        self.M_phi_dot = self._bigM_phi_dot()
        self.M_fw = self._bigM_fw()

        # construct MLD system
        mld_constraints = Matrix(self._mld_constraints())
        self.mld = MLDSystem.from_symbolic(self.x_next_lin, mld_constraints, self.x, u, c.shape[0])

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
        self.x_next = Matrix([q_next, v_next])
        self.x_next_fun = lambdify([self.x, self.f, self.h_sym], self.x_next, 'numpy')

        # linearized dynamics
        self.x_next_lin = self.linearize(self.x_next, [self.x, self.f])
        self.q_next_lin = self.x_next_lin[:self.q.shape[0], :]
        self.v_next_lin = self.x_next_lin[self.q.shape[0]:, :]

        # linearized distance with the wall at the next time step
        q_next_lin_0 = self.q_next_lin.subs({self.fl: 0., self.fr: 0.})
        self.phi_next_lin = {k: self.Phi_nom[k] * q_next_lin_0 + self.phi_lin[k].subs(self.nominal) for k in self.phi.keys()}

        # contact forces with wall
        v_next_0 = v_next.subs({self.fl: 0., self.fr: 0.})
        self.f_explicit_lin = {}
        self.f_explicit_fun = {}
        for k in self.phi.keys():
            M_reduced = (self.Phi[k] * M_inv * self.Phi[k].T).inv()
            f_explicit = - M_reduced * self.Phi[k] * (self.eps * self.v + v_next_0)
            self.f_explicit_lin[k] = self.linearize(f_explicit, [self.x, self.f])
            self.f_explicit_fun[k] = lambdify([self.x, self.fc, self.h_sym], f_explicit[0,0], 'numpy')

    def _setup_lp_bigM(self):

        # initialize
        lp = BoundedQP()

        # optimization variables
        x = lp.add_variables(self.x.shape[0], name='x')
        fc = lp.add_variables(1, name='fc')

        # bounds on current state and input
        lp.add_constraints(x, le, self.x_max)
        lp.add_constraints(x, ge, self.x_min)
        lp.add_constraints(fc, le, np.array([self.fc_max]))
        lp.add_constraints(fc, ge, np.array([self.fc_min]))

        # # bounds on the next state
        # x_next_lin = np.array(self.x_next_lin.jacobian(self.x)).dot(x) + \
        #              np.array(self.x_next_lin.jacobian(Matrix([self.fc]))).dot(fc)
        # lp.add_constraints(x_next_lin, le, self.x_max)
        # lp.add_constraints(x_next_lin, ge, self.x_min)

        # not currently in contact
        for k, phi in self.phi_lin.items():
            phi_lin = np.array(phi.jacobian(self.x)).dot(x) + phi.subs(self.nominal)[0,0]
            lp.add_constraints(phi_lin, ge, np.zeros(1))

        return lp, x, fc


    def _bigM_phi_next(self):

        # setup
        lp, x, fc =self._setup_lp_bigM()

        # loop for left and right
        M = {}
        for k, v in self.phi_next_lin.items():
            phi_next_lin = np.array(v.jacobian(self.x)).dot(x) + \
                           np.array(v.jacobian(Matrix([self.fc]))).dot(fc) + \
                           v.subs(self.nominal)[0,0]

            # lower bound
            lp.setObjective(phi_next_lin[0])
            lp.optimize()
            lb = lp.primal_objective()

            # upper bound
            lp.setObjective(- phi_next_lin[0])
            lp.optimize()
            ub = - lp.primal_objective()

            # collect big Ms
            M[k] = [lb, ub]

        return M

    def _bigM_phi_dot(self):

        # setup
        lp, x, fc = self._setup_lp_bigM()
        v_next_lin = np.array(self.v_next_lin.jacobian(self.x)).dot(x) + \
                     np.array(self.v_next_lin.jacobian(Matrix([self.fc]))).dot(fc)

        # loop for left and right
        M = {}
        for k, v in self.Phi_nom.items():
            v = np.array(v)

            # lower bound
            lp.setObjective(v.dot(v_next_lin + self.eps*x[self.q.shape[0]:])[0])
            lp.optimize()
            lb = lp.primal_objective()

            # upper bound
            lp.setObjective(- v.dot(v_next_lin + self.eps*x[self.q.shape[0]:])[0])
            lp.optimize()
            ub = - lp.primal_objective()

            # collect big Ms
            M[k] = [lb, ub]

        return M

    def _bigM_fw(self):

        # setup
        lp, x, fc = self._setup_lp_bigM()

        # loop for left and right
        M = {}
        for k in self.phi.keys():

            # contact force
            f_lin = np.array(self.f_explicit_lin[k].jacobian(self.x)).dot(x) + \
                    np.array(self.f_explicit_lin[k].jacobian(Matrix([self.fc]))).dot(fc)
            
            # upper bound
            lp.setObjective(- f_lin[0])
            lp.optimize()
            M[k] = - lp.primal_objective()

        return M

    def _mld_constraints(self):

        # collect all the MLD constraints
        constraints = []

        # state bounds
        constraints.append(self.x - self.x_max.reshape(self.x.shape[0], 1))
        constraints.append(self.x_min.reshape(self.x.shape[0], 1) - self.x)

        # input bounds
        constraints.append(Matrix([self.fc - self.fc_max]))
        constraints.append(Matrix([self.fc_min - self.fc]))

        # organize forces and inidcators
        f = {'l': self.fl, 'r': self.fr}
        c = {'l': self.cl, 'r': self.cr}

        # indicator equal one <=> penetration at next time step if no force
        for k, v in self.phi_next_lin.items():

            # lower bound
            indicator = Matrix([self.M_phi_next[k][0] * c[k]])
            constraints.append(- v + indicator)

            # upper bound
            indicator = Matrix([self.M_phi_next[k][1] * (1 - c[k])])
            constraints.append(v - indicator)
            
        # indicator equal one <=> restitution in the gap velocity
        dv = self.v_next_lin + self.eps*self.v
        for k in self.phi.keys():

            # lower bound
            indicator = Matrix([self.M_phi_dot[k][0] * (1 - c[k])])
            constraints.append(- self.Phi_nom[k] * dv + indicator)

            # upper bound
            indicator = Matrix([self.M_phi_dot[k][1] * (1 - c[k])])
            constraints.append(self.Phi_nom[k] * dv - indicator)

        # indicator equal one <=> positive contact force
        for k in self.phi.keys():

            # upper bound
            constraints.append(Matrix([- f[k]]))
            constraints.append(Matrix([f[k] - self.M_fw[k] * c[k]]))

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
            q_next = x_next[:self.q.shape[0]]

            # get contact forces if penetration
            for i, k in enumerate(self.phi.keys()):
                if self.phi_fun[k](q_next) <= 0.:
                    f[i+1] = self.f_explicit_fun[k](x[-1], u, h)

            # simulate with contact forces
            x.append(self.x_next_fun(x[-1], f, h).flatten())
        
        return np.array(x), h


    def project_in_feasible_set(self, x):

        return np.minimum(np.maximum(x, self.x_min), self.x_max)