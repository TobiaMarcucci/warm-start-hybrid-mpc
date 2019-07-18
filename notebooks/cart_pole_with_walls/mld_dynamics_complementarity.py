# external imports
import numpy as np
from sympy import symbols, Matrix, sin, cos
from sympy.utilities import lambdify
from copy import copy

# internal imports
from warm_start_hmpc.mld_system import MLDSystem
from warm_start_hmpc.utils import sym2mat, unpack_bmat
from warm_start_hmpc.mcais import mcais, solve_dare
from warm_start_hmpc.bounded_qp import BoundedQP

'''
numeric parameters
'''

mc = 1. # mass cart
mp = 1. # mass pole
l  = 1. # length pole
d  = .5 # distance walls from origin
eps = .5 # coefficient of restitution
g = Matrix([0., -10]) # gravity acceleration
h_nom = 1./20. # nominal integration step

'''
symbolic variables
'''

# state
[qc, qp, vc, vp] = symbols('q_c q_p v_c v_p')
q = Matrix([qc, qp])
v = Matrix([vc, vp])
x = Matrix([q, v])

# inputs
[fc, fl, fr] = Matrix([symbols('f_c f_l f_r')])
f = Matrix([fc, fl, fr])

# symbolic integration step
h = symbols('h')

'''
equations of motion
'''

# kinematics
pc = Matrix([qc])
pp = Matrix([qc-l*sin(qp), l*cos(qp)])
vc = pc.jacobian(q)*v
vp = pp.jacobian(q)*v

# Lagrangian
U = - mp*(g.T*pp)
T = .5*mc*(vc.T*vc) + .5*mp*(vp.T*vp)
T.simplify()
L = T - U

# mass matrix
M = L.jacobian(v).T.jacobian(v)
M_inv = M.inv()

# Coriolis, centrifugal, and gravitational terms
tau = - L.jacobian(v).T.jacobian(q)*v + L.jacobian(q).T

# generalized forces
Qc = pc.jacobian(q).T * fc
Qw = pp[0:1,:].jacobian(q).T * (fl - fr)
# Q = Qc + Qw

# nonlinear dynamics
# v_dot = M_inv*(tau + Q)
v_next = v + M_inv * (h * (tau + Qc) + Qw)
q_next = q + h * v_next
x_next = Matrix([q_next, v_next])
x_next_fun = lambdify([x, f, h], x_next, 'numpy')

'''
Linearization of the dynamics
'''

# equilibrium point
subs = {
    h: h_nom,
    **{xi: 0. for xi in x},
    **{fi: 0. for fi in f},
    }

# linear dynamics
x_next_lin = x_next.jacobian(x).subs(subs) * x  + \
             x_next.jacobian(f).subs(subs) * f
q_next_lin = x_next_lin[:q.shape[0], :]
v_next_lin = x_next_lin[q.shape[0]:, :]

'''
Gap functions
'''

# gaps
phi_l = Matrix([d + pp[0]])
phi_r = Matrix([d - pp[0]])

# functions
phi_l_fun = lambdify([q], phi_l, 'numpy')
phi_r_fun = lambdify([q], phi_r, 'numpy')

# jacobians
Phi_l = phi_l.jacobian(q)
Phi_r = phi_r.jacobian(q)
Phi_l_nom = Phi_l.subs(subs)
Phi_r_nom = Phi_r.subs(subs)

# linearizations
phi_l_lin = Phi_l.subs(subs) * q + phi_l.subs(subs)
phi_r_lin = Phi_r.subs(subs) * q + phi_r.subs(subs)

'''
Contact forces
'''

# left
# fl_solved_lin is = mp [0, gh, -(1+e), (1+e)l]
M_l = (Phi_l * M_inv * Phi_l.T).inv()
fl_solved = - M_l * ((1+eps) * Phi_l * v + h * Phi_l * M_inv * (tau + Qc))
fl_solved_fun = lambdify([x, fc, h], fl_solved[0,0], 'numpy')

# right
# fr_solved_lin is = mp [0, -gh, 1+e, -(1+e)l]
M_r = (Phi_r * M_inv * Phi_r.T).inv()
fr_solved = - M_r * ((1+eps) * Phi_r * v + h * Phi_r * M_inv*(tau + Qc))
fr_solved_fun = lambdify([x, fc, h], fr_solved[0,0], 'numpy')

'''
Bounds
'''

# state bounds
x_max = np.array([d, np.pi/8., 2., 1.])
x_min = - x_max
x_upper_bound = x - x_max.reshape(x_max.size, 1)
x_lower_bound = x_min.reshape(x_min.size, 1) - x

# input bounds
fc_max = np.array([[2.]])
fc_min = - fc_max
fc_upper_bound = Matrix([fc]) - fc_max
fc_lower_bound = fc_min - Matrix([fc])

# domain
constraints = [
    x_upper_bound,
    x_lower_bound,
    fc_upper_bound,
    fc_lower_bound
    ]

'''
Complementarity constraints
'''

# binary indicators (= 1 if contact)
[cl, cr] = symbols('c_l c_r')
c = Matrix([cl, cr])

# big-M gap
bigM_gap = 2. * d + x_max[2] + .2
phi_l_next = Phi_l_nom * q_next_lin + phi_l_lin.subs(subs)
phi_r_next = Phi_r_nom * q_next_lin + phi_r_lin.subs(subs)
print(phi_r_next)
constraints.extend([
	# phi_l_lin.subs({qi: q_next_lin[i] for i, qi in enumerate(q)}) - \
	phi_l_next.subs({fl: 0., fr: 0.}) - Matrix([bigM_gap * (1 - cl)]),
	# phi_r_lin.subs({qi: q_next_lin[i] for i, qi in enumerate(q)}) - \
	phi_r_next.subs({fl: 0., fr: 0.}) - Matrix([bigM_gap * (1 - cr)])
    ])

# big-M gap velocity
bigM_gap_vel = (1+eps)*(x_max[2] - l * x_min[3])
constraints.extend([
	- Phi_l_nom * (v_next_lin + eps*v) - Matrix([bigM_gap_vel * (1 - cl)]),
	- Phi_r_nom * (v_next_lin + eps*v) - Matrix([bigM_gap_vel * (1 - cr)]),
	Phi_l_nom * (v_next_lin + eps*v) - Matrix([bigM_gap_vel * (1 - cl)]),
	Phi_r_nom * (v_next_lin + eps*v) - Matrix([bigM_gap_vel * (1 - cr)])
	])

# big-M force
fl_solved_lin = fl_solved.jacobian(x).subs(subs)
bigM_f = (fl_solved_lin * Matrix([0, x_max[1], x_min[2], x_max[3]]))[0,0]
constraints.extend([
	Matrix([- fl]),
	Matrix([- fr]),
	Matrix([fl - bigM_f * cl]),
	Matrix([fr - bigM_f * cr])
	])

'''
Redundant bounds to make the constraints bounded
'''
constraints.extend([
	Matrix([- cl]),
	Matrix([- cr]),
	Matrix([cl - 1.]),
	Matrix([cr - 1.]),
	# Matrix([cr + cl - 1]) show be much stronger
	])

'''
Mixed Logical Dynamical system
'''

# construct MLD system
mld = MLDSystem.from_symbolic(
	x_next_lin,
	Matrix(constraints),
	x,
	Matrix([f, c]),
	c.shape[0]
	)

'''
Objective function
'''

# weight matrices
C = np.diag([1.,1.,1.,1.]) * h_nom
D = np.array([[1.]]).T * h_nom

# LQR cost to go
P, K = solve_dare(mld.A, mld.B[:,:1], C.dot(C), D.dot(D))
C_T = np.linalg.cholesky(P).T
D = np.hstack((D, np.zeros((1, mld.nu-1))))
objective = [C, D, C_T]

# mcais terminal set
A_cl = mld.A + mld.B[:,:1].dot(K)
lhs_cl = mld.F + mld.G[:,:1].dot(K)
terminal_set = mcais(A_cl, lhs_cl, mld.h, verbose=True)

# simulation
def simulate(x, dt, u=np.zeros(1), h_des=.001):
    T = int(dt/h_des)
    h = dt/T
    x_list = [x]
    for t in range(T):
    	f = np.array([u[0], 0., 0.])
    	x_next = x_next_fun(x_list[-1], f, h).flatten()

    	# contact forces
    	q_next = x_next[:2]
    	if phi_l_fun(q_next) <= 0.:
    		f[1] = fl_solved_fun(x_list[-1], u[0], h)
    	if phi_r_fun(q_next) <= 0.:
    		f[2] = fr_solved_fun(x_list[-1], u[0], h)

    	x_list.append(x_next_fun(x_list[-1], f, h).flatten())
    
    return np.array(x_list), h


def project_in_feasible_set(x):
    return np.minimum(np.maximum(x, x_min), x_max)