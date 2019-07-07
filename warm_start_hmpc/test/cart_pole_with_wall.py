# external imports
import numpy as np
import sympy as sp
from scipy.linalg import solve_discrete_are

# internal imports
from warm_start_hmpc.mld_system import MLDSystem
from warm_start_hmpc.controller import HybridModelPredictiveController


### NUMERIC PARAMETERS

mc = 1. # mass cart
mp = 1. # mass pole
l  = 1. # length pole
d  = .5 # distance walls from origin
k  = 100.# stiffness contacts
nu = 30. # damping contacts
g  = 10. # gravity acceleration
h  = .05 # integration time step


### SYMBOLIC SYSTEM VARIABLES

x = sp.Matrix(sp.symbols('q t qd td')) # symbolic states
u = sp.Matrix([sp.symbols('u')]) # symbolic inputs
f = sp.Matrix([sp.symbols('f')]) # symbolic force
b = sp.Matrix(sp.symbols('el dam')) # symbolic auxiliary binaries
inputs = sp.Matrix([u, f, b]) # gather inputs and auxiliaries


### DYNAMICS

x2_dot = x[1]*g*mp/mc + u[0]/mc
x3_dot = x[1]*g*(mc+mp)/(l*mc) + u[0]/(l*mc) + f[0]/(l*mp)
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
state_upper_bound = x - x_max.reshape(x_max.size, 1)
state_lower_bound = x_min.reshape(x_min.size, 1) - x

# input bounds
u_max = np.array([2.])
u_min = - u_max
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


### MLD SYSTEM AND CONTROLLER

# mld dynamics
mld = MLDSystem.from_symbolic(dynamics, constraints, x, inputs, b.shape[0])

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
h_T = np.concatenate((x_max, x_max)) / 1.1
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

# generate warm start
np.random.seed(1)
uc0 = solution.variables['uc'][0]
ub0 = solution.variables['ub'][0]
e0 = np.random.randn(mld.nx) * .01
warm_start = controller.construct_warm_start(leaves, x0, uc0, ub0, e0)

# next state
u0 = np.concatenate((uc0, ub0))
x1 = mld.A.dot(x0) + mld.B.dot(u0) + e0


### HELPER FUNCTIONS FOR THE TESTS

def pairwise_disjoint_cover_of_unit_cube(nodes, n):
    np.random.seed(1)

    # generate n random vertices of the unit cube
    is_cover = True
    for _ in range(n):
        ub = np.random.randint(0, 2, mld.nub*T)
        included = 0

        # among the nodes there must be one and only one which containts the vertex
        for node in nodes:
            ub_lb, ub_ub = controller._get_bound_binaries(node.identifier)
            ub_lb = np.concatenate(ub_lb)
            ub_ub = np.concatenate(ub_ub)
            if np.min(ub-ub_lb) >= 0. and np.min(ub_ub-ub) >= 0.:
                included += 1

        # if not covered by a single set, test must fail
        if included != 1:
            is_cover = False
            break

    return is_cover

def plug_in_primal_constraints(variables, identifier):

    # rename primal varibales
    x = variables['x']
    uc = variables['uc']
    ub = variables['ub']
    u = [np.concatenate((uc[t], ub[t])) for t in range(T)]

    # get bounds for this identifier
    ub_lb, ub_ub = controller._get_bound_binaries(identifier)

    # initial state constraint
    zero_terms = []
    zero_terms.append(x0 - x[0])

    # MLD dynamics
    x_next = x0
    for t in range(T):
        x_next = mld.A.dot(x_next) + mld.B.dot(u[t])
        zero_terms.append(x_next - x[t+1])

    # MLD constraints
    nonnegative_terms = []
    for t in range(T):
        nonnegative_terms.append(mld.h - mld.F.dot(x[t]) - mld.G.dot(u[t]))

    # binaries constraint
    for t in range(T):
        nonnegative_terms.append(ub[t] - ub_lb[t])
        nonnegative_terms.append(ub_ub[t] - ub[t])

    # terminal constraints
    nonnegative_terms.append(terminal_set[1] - terminal_set[0].dot(x[T]))
    
    return np.concatenate(zero_terms), np.concatenate(nonnegative_terms)

def plug_in_dual_constraints(variables):

    # rename dual varibales
    rho = variables['rho']
    lam = variables['lam']
    sigma = variables['sigma']
    mu = variables['mu']
    nu_lb = variables['nu_lb']
    nu_ub = variables['nu_ub']

    # dual terminal conditions
    zero_terms = []
    zero_terms.append(controller.C_T.T.dot(rho[T]) + lam[T])

    # dual dynamics time T-1
    zero_terms.append(controller.C.T.dot(rho[T-1]) + lam[T-1] \
	                - mld.A.T.dot(lam[T]) \
	                + controller.F_Tm1.T.dot(mu[T-1]))

    # dual dynamics at time t
    for t in range(T-1):
        zero_terms.append(controller.C.T.dot(rho[t]) + lam[t] \
	                    - mld.A.T.dot(lam[t+1]) \
	                    + mld.F.T.dot(mu[t]))

    # dual constraints at time T-1
    zero_terms.append(controller.D.T.dot(sigma[T-1]) \
	                - mld.B.T.dot(lam[T]) \
	                + controller.G_Tm1.T.dot(mu[T-1]) \
	                + mld.V.T.dot(nu_ub[T-1] - nu_lb[T-1]))
    # dual cosntraints at time t
    for t in range(T-1):
        zero_terms.append(controller.D.T.dot(sigma[t]) \
	                    - mld.B.T.dot(lam[t+1]) \
	                    + mld.G.T.dot(mu[t]) \
	                    + mld.V.T.dot(nu_ub[t] - nu_lb[t]))

    # nonnegativity
    nonnegative_terms = mu + nu_lb + nu_ub

    return np.concatenate(zero_terms), np.concatenate(nonnegative_terms)

def plug_in_dual_objective(variables, identifier):

    # evaluate quadratic terms
    objective = 0.
    for k in ['rho', 'sigma']:
        objective -= sum(np.linalg.norm(vt)**2 for vt in variables[k]) / 4.

    # cost initial conditions
    objective -= variables['lam'][0].dot(x1)

    # cost bounds on binaries
    ub_lb, ub_ub = controller._get_bound_binaries(identifier)
    objective += sum(ub_lb[t].dot(vt) for t, vt in enumerate(variables['nu_lb']))
    objective -= sum(ub_ub[t].dot(vt) for t, vt in enumerate(variables['nu_ub']))

    # cost mld inequalities
    objective -= sum(mld.h.dot(vt) for vt in variables['mu'][:-1])
    objective -= controller.h_Tm1.dot(variables['mu'][-1])

    return objective