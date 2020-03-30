# external imports
import numpy as np
from decimal import Decimal

def get_statistics(data, len_data):
    mins = []
    p80 = []
    p90 = []
    maxs = []
    for t in range(len_data):
        n_t = [float(n[t]) for n in data]
        mins.append(min(n_t))
        p80.append(np.percentile(n_t, 80))
        p90.append(np.percentile(n_t, 90))
        maxs.append(max(n_t))
    return mins, p80, p90, maxs

def plot_statistics(ax, data, N_sim, t0=0, **kwargs):
    mins, p80, p90, maxs = get_statistics(data, N_sim-t0)
    ax.plot(range(t0, N_sim), mins, **kwargs)
    ax.plot(range(t0, N_sim), maxs, **kwargs)
    ax.plot(range(t0, N_sim), p80, linestyle='--', **kwargs)
    ax.plot(range(t0, N_sim), p90, linestyle=':', **kwargs)
    ax.fill_between(range(t0, N_sim), mins, maxs, alpha=.1, **kwargs)

def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

def fman(number):
    return Decimal(number).scaleb(-fexp(number)).normalize()
