# external imports
import numpy as np
import sympy as sp

# internal imports
from warm_start_hmpc.mld_system import MLDSystem


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
f = sp.Matrix(sp.symbols('f_l f_r'))

# symbolic auxiliary binaries (left, right)
b = sp.Matrix(sp.symbols('el_l dam_l el_r dam_r'))

# gather inputs and auxiliaries
inputs = sp.Matrix([u, f, b])


### DYNAMICS

# time derivatives
x2_dot = x[1]*g*mp/mc + u[0]/mc
x3_dot = x[1]*g*(mc+mp)/(l*mc) + u[0]/(l*mc) + (f[1]-f[0])/(l*mp)

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
f = { # force
    'l': f[0],
    'r': f[1]
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
f_min = { # min force
    'l': k*p_min['l'] + nu*p_dot_min['l'],
    'r': k*p_min['r'] + nu*p_dot_min['r']
}
f_max = { # max force
    'l': k*p_max['l'] + nu*p_dot_max['l'],
    'r': k*p_max['r'] + nu*p_dot_max['r']
}

# contact constraints big-M method
contacts = []
for lr in ['l', 'r']:
    
    # el = 1 if p >= 0, el = 0 if otherwise
    contacts.append(sp.Matrix([p_min[lr] * (1. - el[lr]) - p[lr]]))
    contacts.append(sp.Matrix([p[lr] - p_max[lr] * el[lr]]))
    
    # dam = 1 if k p + nu p_dot >= 0, dam = 0 if otherwise
    contacts.append(sp.Matrix([f_min[lr] * (1. - dam[lr]) - k*p[lr] - nu*p_dot[lr]]))
    contacts.append(sp.Matrix([k*p[lr] + nu*p_dot[lr] - f_max[lr] * dam[lr]]))
    
    # el = 0 or dam = 0 implies f = 0
    contacts.append(sp.Matrix([- f[lr]]))
    contacts.append(sp.Matrix([f[lr] - f_max[lr]*el[lr]]))
    contacts.append(sp.Matrix([f[lr] - f_max[lr]*dam[lr]]))
    
    # el = dam = 1 implies f = k p + nu p_dot
    contacts.append(sp.Matrix([k*p[lr] + nu*p_dot[lr] + nu*p_dot_max[lr]*(el[lr]-1.) - f[lr]]))
    contacts.append(sp.Matrix([f[lr] - k*p[lr] - nu*p_dot[lr] - f_min[lr]*(dam[lr]-1.)]))

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