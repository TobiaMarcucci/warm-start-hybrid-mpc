# external imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# internal imports
from plot_utils import plot_statistics, fexp, fman

# number of time steps per simulation
N_sim = 50

# model error standard deviation
e_sd = [0., 0.001, 0.003, 0.010]

# load data
nodes_cs = {}
nodes_ws = {}
len_ws = {}
for e in e_sd:
    nodes_cs[e] = list(np.load('data/nodes_cs_sd_{:.3f}.npy'.format(e), allow_pickle=True))
    nodes_ws[e] = [t[1:] for t in np.load('data/nodes_ws_sd_{:.3f}.npy'.format(e), allow_pickle=True)]
    len_ws[e] = [t[:-1] for t in np.load('data/nodes_len_ws_sd_{:.3f}.npy'.format(e), allow_pickle=True)]    

# remove infeasible trajecotories
for e in e_sd:
    nodes_cs[e] = [t for t in nodes_cs[e] if len(t) == N_sim]
    nodes_ws[e] = [t for t in nodes_ws[e] if len(t) == N_sim-1]
    len_ws[e] = [t for t in len_ws[e] if len(t) == N_sim-1]

# set latex params
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 12})

# initialize figure
fig = plt.figure(figsize=(6, 10))
gridspec.GridSpec(20,1)

# sizes of the subplots
heights = [175, 175, 350, 700]
rowspans = [2, 2, 4, 8]
xticks = range(0, N_sim+1, 5)

# nominal case
plt.subplot2grid((20, 1), (0,0), rowspan=rowspans[0], colspan=1)
plt.title(r'Model-error standard deviation $\sigma_i = 0$')
plt.plot(range(1,len(len_ws[0.][0])+1), len_ws[0.][0], color=np.ones(3)*.6)#'#2ca02c')
plt.plot(range(len(nodes_cs[0.][0])), nodes_cs[0.][0], color='#1f77b4')
plt.plot(range(1,len(nodes_ws[0.][0])+1), nodes_ws[0.][0], color='#ff7f0e')
plt.grid(True, color=np.ones(3)*.85)
plt.xlim(0, N_sim)
plt.ylim(0, heights[0])
plt.xticks(xticks, ['']*len(xticks))

# with disturbances
for i, e in enumerate(e_sd[1:]):
    i += 1
    plt.subplot2grid(
        (20, 1),
        (sum(rowspans[:i]) + i,0),
        rowspan=rowspans[i],
        colspan=1
    )
    plot_statistics(plt, len_ws[e], N_sim, t0=1, color=np.ones(3)*.6)#'#2ca02c')
    plot_statistics(plt, nodes_cs[e], N_sim, color='#1f77b4')
    plot_statistics(plt, nodes_ws[e], N_sim, t0=1, color='#ff7f0e')
    plt.grid(True, color=np.ones(3)*.85)
    plt.xlim(0, N_sim)
    plt.ylim(0, heights[i])
    plt.xticks(xticks, ['']*len(xticks))
    coef = float(fman(e_sd[i]))
    exp = fexp(e_sd[i])
    if coef == 1.:
        plt.title(r'$\sigma_i = 10^{%.0f} \bar x_i$'%exp)
    else:
        plt.title(r'$\sigma_i = %.0f \cdot 10^{%.0f} \bar x_i$'%(coef,exp))

# x axis
plt.xlabel(r'Time step $\tau$')
plt.xticks(xticks, xticks)
    
# legend
plt.scatter(-10., -10., color='#ff7f0e', label='QP solves w/ warm start')
plt.scatter(-10., -10., color='#1f77b4', label='QP solves w/ cold start')
plt.scatter(-10., -10., color=np.ones(3)*.6, label=r'Initial-cover cardinality')#'#2ca02c')
plt.plot(-10., -10., color='k', label='min, max')
plt.plot(-10., -10., color='k', linestyle='--', label='80th percentile')
plt.plot(-10., -10., color='k', linestyle=':', label='90th percentile')
handles, labels = plt.gca().get_legend_handles_labels()
order = [3,4,5,0,1,2] # 3 columns
lgd = plt.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    loc='upper center',
    bbox_to_anchor=(0.477, -0.15),
    ncol=2
)

# save
fig.savefig('nodes.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
