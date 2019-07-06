# external imports
import numpy as np
import sympy as sp
from scipy.linalg import solve_discrete_are

# internal imports
from warm_start_hmpc.mld_system import MLDSystem
from warm_start_hmpc.controller import HybridModelPredictiveController


### NUMERIC PARAMETERS

# mass cart
mc = 1.

# mass pole
mp = 1.   

# length pole
l  = 1.   

# distance walls from origin
d  = .5   

# stiffness contacts
k  = 100. 

# damping contacts
nu = 30.  

# gravity acceleration
g  = 10.  

# integration time step
h  = .05  


### SYMBOLIC SYSTEM VARIABLES

# symbolic states
x = sp.Matrix(sp.symbols('q t qd td'))

# symbolic inputs
u = sp.Matrix([sp.symbols('u')])

# symbolic forces (left, right)
f = sp.Matrix([sp.symbols('f')])

# symbolic auxiliary binaries (left, right)
b = sp.Matrix(sp.symbols('el dam'))

# gather inputs and auxiliaries
inputs = sp.Matrix([u, f, b])


### DYNAMICS

# time derivatives
x2_dot = x[1]*g*mp/mc + u[0]/mc
x3_dot = x[1]*g*(mc+mp)/(l*mc) + u[0]/(l*mc) + f[0]/(l*mp)

# discretized dynamics
dynamics = sp.Matrix([
    x[0] + h * x[2],
    x[1] + h * x[3],
    x[2] + h * x2_dot,
    x[3] + h * x3_dot
])


### CONSTRAINTS

# state bounds
x_max = np.array([d, np.pi/8., 2., 1.])
x_min = - x_max

# input bounds
u_max = np.array([2.])
u_min = - u_max

# state constraints
state_upper_bound = x - x_max.reshape(x_max.size, 1)
state_lower_bound = x_min.reshape(x_min.size, 1) - x

# input constraints
input_upper_bound = u - u_max.reshape(u_max.size, 1)
input_lower_bound = u_min.reshape(u_min.size, 1) - u

# auxiliary dictionaries for remaining constraints
p = x[0] - l*x[1] - d # penetration
p_dot = x[2] - l*x[3] # relative velocity
p_min = x_min[0] - l*x_max[1] - d # min penetration
p_max = x_max[0] - l*x_min[1] - d # max penetration
p_dot_min = x_min[2] - l*x_max[3] # min relative velocity
p_dot_max = x_max[2] - l*x_min[3] # max relative velocity
f_min = k*p_min + nu*p_dot_min # min force
f_max = k*p_max + nu*p_dot_max # max force

# contact constraints big-M method
contacts = []

# el = 1 if p >= 0, el = 0 if otherwise
contacts.append(sp.Matrix([p_min * (1. - b[0]) - p]))
contacts.append(sp.Matrix([p - p_max * b[0]]))

# dam = 1 if k p + nu p_dot >= 0, dam = 0 if otherwise
contacts.append(sp.Matrix([f_min * (1. - b[1]) - k*p - nu*p_dot]))
contacts.append(sp.Matrix([k*p + nu*p_dot - f_max * b[1]]))

# el = 0 or dam = 0 implies f = 0
contacts.append(sp.Matrix([- f[0]]))
contacts.append(sp.Matrix([f[0] - f_max*b[0]]))
contacts.append(sp.Matrix([f[0] - f_max*b[1]]))

# el = dam = 1 implies f = k p + nu p_dot
contacts.append(sp.Matrix([k*p + nu*p_dot + nu*p_dot_max*(b[0]-1.) - f[0]]))
contacts.append(sp.Matrix([f[0] - k*p - nu*p_dot - f_min*(b[1]-1.)]))

# gather constraints
constraints = sp.Matrix([
    state_upper_bound,
    state_lower_bound,
    input_upper_bound,
    input_lower_bound,
    sp.Matrix(contacts)
])


### CONSTRUCT MLD SYSTEM

mld = MLDSystem.from_symbolic(dynamics, constraints, x, inputs, b.shape[0])

### CONTROLLER

# horizon
T = 40

# weight matrices
C = np.eye(mld.nx)
D = np.vstack([1.]+[0.]*(mld.nu - 1)).T
C_T_scaling = 1.1
C_T = C * C_T_scaling
objective = [C, D, C_T]

# terminal constraints
F_T = np.vstack((np.eye(mld.nx), - np.eye(mld.nx)))
h_T = np.concatenate((x_max, x_max))/1.1
terminal_set = [F_T, h_T]

# hybrid controller
controller = HybridModelPredictiveController(mld, T, objective, terminal_set)

### SOLVE OPTIMAL CONTROL PROBLEM

# initial state
x0 = np.array([0., 0., 1., 0.])

# solve MIQP
# solution, leaves = controller.feedforward(x0, draw_label='test')
solution, leaves = controller.feedforward(x0, printing_period=None)

# solve leaf subproblem
solution_leaves = [controller._solve_subproblem(leaf.identifier, x0) for leaf in leaves]
