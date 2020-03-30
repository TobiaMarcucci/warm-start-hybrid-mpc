# external imports
import numpy as np

# internal imports
from mld_dynamics import mld, h
from warm_start_hmpc.controller import HybridModelPredictiveController
from warm_start_hmpc.mcais import mcais, solve_dare

# controller horizon
T = 20

# weight matrices
Q = np.eye(mld.nx) * h
R = np.vstack([1.]+[0.]*(mld.nu - 1)).T * h

# LQR terminal cost
Bu = mld.B[:, :1]
Ru = R[:, :1]
P, K = solve_dare(mld.A, Bu, Q.dot(Q), Ru.dot(Ru))
Q_T = np.linalg.cholesky(P).T
objective = [Q, R, Q_T]

# MCAIS terminal set
A_cl = mld.A + Bu.dot(K)
lhs = mld.F + mld.G[:, :1].dot(K)
rhs = mld.h
terminal_set = mcais(A_cl, lhs, rhs, verbose=True)

# hybrid controller
controller = HybridModelPredictiveController(mld, T, objective, terminal_set)
