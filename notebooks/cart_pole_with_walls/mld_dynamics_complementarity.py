# external imports
import numpy as np
from sympy import symbols, Matrix, sin, cos
from sympy.utilities import lambdify
from copy import copy

# internal imports
from warm_start_hmpc.mld_system import MLDSystem

'''
numeric parameters
'''

mc = 1. # mass cart
mp = 1. # mass pole
l  = 1. # length pole
d  = .5 # distance walls from origin
eps = .5 # coefficient of restitution
g  = 10. # gravity acceleration
h_nom = .05 # nominal integration step
g_vec = Matrix([0., -g])

'''
symbolic variables
'''

# state
[qc, qp, vc, vp] = symbols('qc qp vc vp')
q = Matrix([qc, qp])
v = Matrix([vc, vp])
x = Matrix([q, v])

# input
u = Matrix([symbols('u')])
f = Matrix([u, Matrix([0])])

# symbolic integration step
h = symbols('h')

'''
equations of motion
'''

# kinematics
pc = Matrix([qc, 0])
pp = Matrix([qc-l*sin(qp), l*cos(qp)])
vc = pc.jacobian(q)*v
vp = pp.jacobian(q)*v

# Lagrangian
U = - mp*(g_vec.T*pp)
T = .5*mc*(vc.T*vc) + .5*mp*(vp.T*vp)
T.simplify()
L = T - U

# mass matrix
M = L.jacobian(v).T.jacobian(v)
M_inv = M.inv()

# Coriolis, centrifugal, and gravitational terms
tau = - L.jacobian(v).T.jacobian(q)*v + L.jacobian(q).T

# generalized forces
Q = pc.jacobian(q).T * f

'''
Bounds
'''

# state bounds
x_max = np.array([d, np.pi/8., 2., 1.])
x_min = - x_max
state_upper_bound = x - x_max.reshape(x_max.size, 1)
state_lower_bound = x_min.reshape(x_min.size, 1) - x

# input bounds
u_max = np.array([2.])
u_min = - u_max
input_upper_bound = u - u_max.reshape(u_max.size, 1)
input_lower_bound = u_min.reshape(u_max.size, 1) - u

# domain
bounds = [
    state_upper_bound,
    state_lower_bound,
    input_upper_bound,
    input_lower_bound
    ]

'''
No contact
'''

# equilibrium point
subs = {
    h: h_nom,
    **{xi: 0. for xi in x},
    **{ui: 0. for ui in u},
    }

# nonlinear dynamics
v_dot = M.inv()*(tau + Q)
# v_dot = v_dot.subs({qi: qi + h*v[i] for i, qi in enumerate(q)})
v_dot.simplify()
v_next = v + h * v_dot
v_next.simplify()
q_next = q + h * v_next
q_next.simplify()
x_next = Matrix([q_next, v_next])
nonlinear_dynamics = [lambdify([x, u, h], x_next, 'numpy')]

# linearized dynamics
x_next_lin = x_next.jacobian(x).subs(subs) * x + \
             x_next.jacobian(u).subs(subs) * u + \
             x_next.subs(subs)
linearized_dynamics = [x_next_lin]

# domains
gaps = [Matrix([d + pp[0]]), Matrix([d - pp[0]])]
gaps_next = []
nonlinear_domain = copy(bounds)
linearized_domain = copy(bounds)
for gap in gaps:

    # nonlinear domain
    gap_next = gap.subs({qi: q_next[i] for i, qi in enumerate(q)})
    gaps_next.append(gap_next)
    nonlinear_domain.append(- gap_next)

    # linearized domain
    gap_next_lin = gap_next.jacobian(x).subs(subs) * x + \
                   gap_next.jacobian(u).subs(subs) * u + \
                   gap_next.subs(subs)
    linearized_domain.append(- gap_next_lin)

# initilize lists for PWA system
nonlinear_domains = [lambdify([x, u, h], Matrix(nonlinear_domain), 'numpy')]
linearized_domains = [Matrix(linearized_domain)]

# '''
# Contact with walls
# '''

for i, gap in enumerate(gaps):

    # nonlinear dynamics
    Phi = gap.jacobian(q)
    L = - (1+eps) * M_inv * Phi.T * (Phi*M_inv*Phi.T).inv()
    v_next = v + L * Phi * v
    q_next = q + h * v_next
    x_next = Matrix([q_next, v_next])
    nonlinear_dynamics.append(lambdify([x, u, h], x_next, 'numpy'))

    # linearized dynamics
    x_next_lin = x_next.jacobian(x).subs(subs) * x + x_next.subs(subs)
    linearized_dynamics.append(x_next_lin)
    print(x_next_lin)

    # nonlinear domain
    nonlinear_domain = Matrix(bounds + [gaps_next[i]])
    nonlinear_domains.append(lambdify([x, u, h], nonlinear_domain, 'numpy'))

    # linearized domain
    gap_next_lin = gaps_next[i].jacobian(x).subs(subs) * x + \
                   gaps_next[i].jacobian(u).subs(subs) * u + \
                   gaps_next[i].subs(subs)
    linearized_domain = copy(bounds) + [gap_next_lin]
    linearized_domains.append(Matrix(linearized_domain))

# construct MLD system
mld = MLDSystem.from_symbolic_pwa(linearized_dynamics, linearized_domains, x, u)

# simulation
def simulate(x, dt, u=np.zeros(1), h_des=.01):
    T = int(dt/h_des)
    h = dt/T
    x_list = [x]
    for t in range(T):
        domain = None
        for i, d in enumerate(nonlinear_domains):
            if all(d(x_list[-1], u, h) < 0.):
                domain = i
                break
        x_list.append(nonlinear_dynamics[i](x_list[-1], u, h).flatten())
    return np.array(x_list), h