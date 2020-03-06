# external imports
import numpy as np
import matplotlib.pyplot as plt

# internal imports
from controller import controller
from mld_dynamics import mld, h, x_min, x_max, fc_min, fc_max, d, l

# initial push towards the right wall
x0 = np.array([0., 0., 1., 0.])

# set up simulation
x = [x0]
u = []
warm_start = None

# simulate system in closed loop
T_sim = 50
for t in range(T_sim):
    print(f'Time step {t}', end='\r')
    
    # solve miqp
    solution, leaves = controller.feedforward(
        x[-1],
        warm_start=warm_start,
        printing_period=None
    )[:2]
    
    # reorganize solution
    uc0 = solution.variables['uc'][0]
    ub0 = solution.variables['ub'][0]
    u0 = np.concatenate((uc0, ub0))
    
    # generate warm start
    warm_start = controller.construct_warm_start(
        leaves,
        x[-1],
        uc0,
        ub0,
        np.zeros(4)
    )[0]
    
    # retrieve closed-loop trajectory
    x.append(solution.variables['x'][1])
    u.append(uc0[0])

# plot settings
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 12})
plt.rc('text.latex', preamble=r'\usepackage{stackengine}')

# plot of the input sequence
fig, axes = plt.subplots(2, sharex=True, figsize=(6,3.5))
t = range(T_sim+1)
axes[0].step( # lower bound
    t,
    [fc_min[0]] * (T_sim + 1),
    label=r'Force bound \stackunder[1.2pt]{$u$}{\rule{.8ex}{.075ex}}$_1$',
    color='#d62728',
    linestyle='--'
    )
axes[0].step( # optimal control
    t,
    [u[0]] + u,
    label=r'Force on cart $u_1$',
    color='#1f77b4'
    )
handles, labels = axes[0].get_legend_handles_labels()
order = [1,0] # swap order in the legend
axes[0].legend([handles[i] for i in order],[labels[i] for i in order], loc='right')
axes[0].set_xlim((0, T_sim))
yticks = np.linspace(-1, .4, 8)
axes[0].set_yticks(yticks)

# tip of the pole linearized position
C = np.array([[1., -l, 0., 0.]]) # extracts the position of the pole from the state
y = [C.dot(xt) for xt in x]
axes[1].step(
    t,
    [d] * (T_sim + 1),
    label=r'Right wall $d$',
    color='#2ca02c',
    linestyle='--'
    )
axes[1].plot(t, y, label=r'Pole tip $x_1-lx_2$', color='#1f77b4')
axes[1].set_xlabel(r'Time step $\tau$')
handles, labels = axes[1].get_legend_handles_labels()
order = [1,0] # swap order in the legend
axes[1].legend([handles[i] for i in order],[labels[i] for i in order], loc='right')
yticks = np.linspace(0, .6, 7)
axes[1].set_yticks(yticks)

# misc settings
xticks = range(0, 51, 5)
plt.xticks(xticks)
axes[0].grid(True, color=np.ones(3) * .85)
axes[1].grid(True, color=np.ones(3) * .85)

# show and save
plt.show()
fig.savefig('trajectory.pdf', bbox_inches='tight')
