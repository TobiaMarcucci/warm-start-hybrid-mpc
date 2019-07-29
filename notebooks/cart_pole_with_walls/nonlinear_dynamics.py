# external imports
import numpy as np
from sympy import symbols, Matrix, sin, cos
from sympy.utilities import lambdify

'''
numeric parameters
'''

mc = 1. # mass cart
mp = 1. # mass pole
l  = 1. # length pole
d  = .5 # distance walls from origin
stiffness  = 100. # stiffness contacts
damping = 10. # damping contacts
g  = 10. # gravity acceleration
g_vec = Matrix([0., -g])

'''
symbolic variables
'''

# state
[qc, qp, qcd, qpd] = symbols('qc qp qcd qpd')
q = Matrix([qc, qp])
qd = Matrix([qcd, qpd])
x = Matrix([q, qd])

# input
fc = symbols('fc')
fc_vec = Matrix([fc, 0])

# contact forces
[fl, fr] = symbols('fl fr')
fl_vec = Matrix([fl, 0])
fr_vec = Matrix([-fr, 0])

# all the external forces
f = Matrix([fc, fl, fr])

'''
equations of motion
'''

# kinematics
pc = Matrix([qc, 0])
pp = Matrix([qc-l*sin(qp), l*cos(qp)])
vc = pc.jacobian(q)*qd
vp = pp.jacobian(q)*qd

# lagrangian
potential_energy = - mp*(g_vec.T*pp)
kinetic_energy = .5*mc*(vc.T*vc) + .5*mp*(vp.T*vp)
kinetic_energy.simplify()
lagrangian = kinetic_energy - potential_energy

# mass matrix
M = lagrangian.jacobian(qd).T.jacobian(qd)

# Coriolis, centrifugal, and gravitational terms
c = lagrangian.jacobian(qd).T.jacobian(q)*qd - lagrangian.jacobian(q).T

# generalized forces
Q = pc.jacobian(q).T*fc_vec + \
    pp.jacobian(q).T*fl_vec + \
    pp.jacobian(q).T*fr_vec

# nonlinear dynamics
rhs = Matrix([qd, M.inv()*(- c + Q)])
rhs.simplify()
rhs_fun = lambdify([x, f], rhs, 'numpy')

# equibirium point
subs = {
    **{xi: 0. for i, xi in enumerate(x)},
    **{fi: 0. for i, fi in enumerate(f)}
}

# linearized dynamics
A = rhs.jacobian(x).subs(subs)
B = rhs.jacobian(f).subs(subs)
# A.simplify()
# B.simplify()

# '''
# simulator
# '''

# # gap functions
# gap = {
#     'l': pp[0] + d,
#     'r': d - pp[0]
#     }
# gap_fun = {k: lambdify([x], v, 'numpy') for k, v in gap.items()}

# # contact model
# contact_model = {
#     'l': - stiffness*gap['l'] - damping*vp[0],
#     'r': - stiffness*gap['r'] + damping*vp[0]
#     }
# contact_model_fun = {k: lambdify([x], v, 'numpy') for k, v in contact_model.items()}

# # state derivative
# def x_dot(x, fc):
#     forces = [fc]
#     for wall in ['l','r']:
#         if gap_fun[wall](x) > 0. or contact_model_fun[wall](x) < 0.:
#             forces.append(0.)
#         else:
#             forces.append(contact_model_fun[wall](x))
#     return rhs_fun(x, forces).flatten()

# # simulator
# def simulate(x, dt, fc=0., h_des=.001):
#     T = int(dt/h_des)
#     h = dt/T
#     x_list = [x]
#     for t in range(T):
#         x_list.append(x_list[-1] + h*x_dot(x_list[-1], fc))
#     return np.array(x_list)