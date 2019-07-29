# external imports
import numpy as np
from sympy import symbols, Matrix

# internal imports
from warm_start_hmpc.mld_system import MLDSystem
import nonlinear_dynamics as nld
from pympc.dynamics.discretization_methods import zero_order_hold

'''
numeric parameters
'''

mc = nld.mc
mp = nld.mp
l = nld.l
d = nld.d
stiffness = nld.stiffness
damping = nld.damping
g = nld.g
h  = .05 # integration time step

'''
symbolic variables
'''

x = nld.x # state
fc = nld.fc # force on the cart
b = Matrix(symbols('el_l dam_l el_r dam_r')) # symbolic auxiliary binaries (left, right)
u = Matrix([nld.fc, nld.fl, nld.fr, *b]) # gather inputs and auxiliaries

'''
linearized dynamics
'''

# explicit Euler
x_dot = nld.A * x + nld.B * nld.f
dynamics = x + h * x_dot

'''
linearized constraints
'''

# state bounds
x_max = np.array([d, np.pi/10., 1., 1.])
x_min = - x_max
state_upper_bound = x - x_max.reshape(x_max.size, 1)
state_lower_bound = x_min.reshape(x_min.size, 1) - x

# input bounds
fc_max = np.array([1.])
fc_min = - fc_max
input_upper_bound = Matrix([fc - fc_max])
input_lower_bound = Matrix([fc_min - fc])

# auxiliary dictionaries for remaining constraints
fw = { # wall forces
    'l': nld.fl,
    'r': nld.fr
}
el = { # inidicator for penetation
    'l': b[0],
    'r': b[2]
}
dam = { # inidicator for pulling
    'l': b[1],
    'r': b[3]
}
p = { # penetration
    'l': - d - x[0] + l*x[1],
    'r': x[0] - l*x[1] - d
}
p_dot = { # relative velocity
    'l': - x[2] + l*x[3],
    'r': x[2] - l*x[3]
}
p_min = { # min penetration
    'l': - d - x_max[0] + l*x_min[1],
    'r': x_min[0] - l*x_max[1] - d
}
p_max = { # max penetration
    'l': - d - x_min[0] + l*x_max[1],
    'r': x_max[0] - l*x_min[1] - d
}
p_dot_min = { # min relative velocity
    'l': - x_max[2] + l*x_min[3],
    'r': x_min[2] - l*x_max[3]
}
p_dot_max = { # max relative velocity
    'l': - x_min[2] + l*x_max[3],
    'r': x_max[2] - l*x_min[3]
}
fw_min = { # min force
    k: stiffness*p_min[k]+damping*p_dot_min[k] for k in ['l','r']
}
fw_max = { # max force
    k: stiffness*p_max[k] + damping*p_dot_max[k] for k in ['l','r']
}

# contact constraints big-M method
contacts = []
for k in ['l', 'r']:
    
    # el = 1 if p >= 0, el = 0 if otherwise
    contacts.append(Matrix([p_min[k] * (1. - el[k]) - p[k]]))
    contacts.append(Matrix([p[k] - p_max[k] * el[k]]))
    
    # dam = 1 if k p + nu p_dot >= 0, dam = 0 if otherwise
    contacts.append(Matrix([fw_min[k] * (1. - dam[k]) - stiffness*p[k] - damping*p_dot[k]]))
    contacts.append(Matrix([stiffness*p[k] + damping*p_dot[k] - fw_max[k] * dam[k]]))
    
    # el = 0 or dam = 0 implies f = 0
    contacts.append(Matrix([- fw[k]]))
    contacts.append(Matrix([fw[k] - fw_max[k]*el[k]]))
    contacts.append(Matrix([fw[k] - fw_max[k]*dam[k]]))
    
    # el = dam = 1 implies f = k p + nu p_dot
    contacts.append(Matrix([stiffness*p[k] + damping*p_dot[k] + damping*p_dot_max[k]*(el[k]-1.) - fw[k]]))
    contacts.append(Matrix([fw[k] - stiffness*p[k] - damping*p_dot[k] - fw_min[k]*(dam[k]-1.)]))

# gather constraints
constraints = Matrix([
    state_upper_bound,
    state_lower_bound,
    input_upper_bound,
    input_lower_bound,
    Matrix(contacts)
])

'''
mld system
'''

mld = MLDSystem.from_symbolic(dynamics, constraints, x, u, b.shape[0])