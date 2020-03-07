# external imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import log10


# internal imports
from plot_utils import plot_statistics, fexp, fman

# number of time steps per simulation
N_sim = 50

# model error standard deviation
e_sd = [0., 0.001, 0.003, 0.010]

# load data
times_cs = {}
times_ws = {}
times_ws_constr = {}
times_grb = {}
times_grb_fair = {}
for e in e_sd:
    times_cs[e] = list(np.load('data/times_cs_sd_{:.3f}.npy'.format(e), allow_pickle=True))
    times_ws[e] = [t[1:] for t in np.load('data/times_ws_sd_{:.3f}.npy'.format(e), allow_pickle=True)]
    times_grb[e] = list(np.load('data/times_grb_sd_{:.3f}.npy'.format(e), allow_pickle=True))
    times_ws_constr[e] = [t[:-1] for t in np.load('data/times_ws_constr_sd_{:.3f}.npy'.format(e), allow_pickle=True)]    
    times_grb_fair[e] = list(np.load('data/times_grb_fair_sd_{:.3f}.npy'.format(e), allow_pickle=True))

# remove infeasible trajecotories
for e in e_sd:
    times_cs[e] = [t for t in times_cs[e] if len(t) == N_sim]
    times_ws[e] = [t for t in times_ws[e] if len(t) == N_sim-1]
    times_grb[e] = [t for t in times_grb[e] if len(t) == N_sim]
    times_ws_constr[e] = [t for t in times_ws_constr[e] if len(t) == N_sim-1]
    times_grb_fair[e] = [t for t in times_grb_fair[e] if len(t) == N_sim]

# reformat y-axis ticks
def tick_format(value, tick_number):
    return r'$10^{%d} \ \mathrm{s}$' % int(log10(value))

# set latex params
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 12})

# initialize figure
fig = plt.figure(figsize=(6, 10))
gridspec.GridSpec(20, 1)

# sizes of the subplots
heights = [2., 2., 2., 2.]
rowspans = [4, 4, 4, 4]
xticks = range(0, N_sim + 1, 5)
yticks = [0.0001, 0.001, 0.01, 0.1, 1.]

# nominal case
plt.subplot2grid((20, 1), (0,0), rowspan=rowspans[0], colspan=1)
plt.title(r'Model-error standard deviation $\sigma_i = 0$')
plt.plot(range(1, len(times_ws_constr[0.][0])+1), times_ws_constr[0.][0], color=np.ones(3)*.6)
plt.plot(range(len(times_grb[0.][0])), times_grb[0.][0], color='#2ca02c')
plt.plot(range(len(times_cs[0.][0])), times_cs[0.][0], color='#1f77b4')
plt.plot(range(1, len(times_ws[0.][0])+1), times_ws[0.][0], color='#ff7f0e')
plt.grid(True, color=np.ones(3)*.85)
plt.xlim(0, N_sim)
plt.xticks(xticks, ['']*len(xticks))
plt.yscale('log')
plt.yticks(yticks)
plt.ylim(1e-4, heights[0])
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(tick_format))

# with disturbances
for i, e in enumerate(e_sd[1:]):
    i += 1
    plt.subplot2grid(
        (20, 1),
        (sum(rowspans[:i]) + i,0),
        rowspan=rowspans[i],
        colspan=1
    )
    plot_statistics(plt, times_ws_constr[e], N_sim, t0=1, color=np.ones(3)*.6)
    plot_statistics(plt, times_grb[e], N_sim, color='#2ca02c')
    plot_statistics(plt, times_cs[e], N_sim, color='#1f77b4')
    plot_statistics(plt, times_ws[e], N_sim, t0=1, color='#ff7f0e')
    plt.grid(True, color=np.ones(3)*.85)
    plt.xlim(0, N_sim)
    plt.yscale('log')
    plt.yticks(yticks)
    plt.ylim(1e-4, heights[i])
    plt.xticks(xticks, ['']*len(xticks))
    coef = float(fman(e_sd[i]))
    exp = fexp(e_sd[i])
    if coef == 1.:
        plt.title(r'$\sigma_i = 10^{%.0f} \bar x_i$'%exp)
    else:
        plt.title(r'$\sigma_i = %.0f \cdot 10^{%.0f} \bar x_i$'%(coef,exp))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(tick_format))
    
# x axis common to all subplots
plt.xlabel(r'Time step $\tau$')
plt.xticks(xticks, xticks)

# legend
scatter = lambda c, l: plt.scatter(-10., -10., color=c, label=l)
plot = lambda c, l, ls: plt.plot(-10., -10., color=c, label=l, linestyle=ls)
scatter('#ff7f0e', 'Solve time w/ warm start')
scatter('#1f77b4', 'Solve time w/ cold start')
scatter('#2ca02c', 'Solve time w/ Gurobi')
scatter(np.ones(3) * .6, 'Warm-start construction time')
plot('k', 'min, max', '-')
plot('k', '80th percentile', '--')
plot('k', '90th percentile', ':')
handles, labels = plt.gca().get_legend_handles_labels()
order = [3,4,5,6,0,1,2] # 3 columns
lgd = plt.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    loc='upper center',
    bbox_to_anchor=(0.477, -0.30),
    ncol=2
)

# save
fig.savefig('times.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
