# external imports
import numpy as np

def simulate_closed_loop(cp, T_sim, warm_start=True):

    # set up simulation
    x_predicted = []
    x_true = [x0]
    x_sim = []
    u = []
    e = []
    V = []
    warm_start = None

    for t in range(T_sim):
        print('Time step %d.'%t, end='\r')
        
        # solve miqp
        solution, leaves = controller.feedforward(
            cp.project_in_feasible_set(x_true[-1]),
            warm_start=warm_start,
            printing_period=None
        )
        
        # reorganize solution
        uc0 = solution.variables['uc'][0]
        ub0 = solution.variables['ub'][0]
        u0 = np.concatenate((uc0, ub0))

        # simulate with nonlinear model
        x_sim.extend(cp.simulate(x_true[-1], uc0[0], cp.h)[0])
        x_sim_t = cp.simulate(x_true[-1], uc0[0:1], cp.h)[0]
        x1 = x_sim_t[-1]
        e0 = x1 - cp.mld.A.dot(x_true[-1]) - cp.mld.B.dot(u0)
        print('predicted state:', solution.variables['x'][1])
        print('true state:', x1)
        print('modeling error:', e0)
        visualize(x1)
        
        # generate warm start
        warm_start = controller.construct_warm_start(
            leaves,
            cp.project_in_feasible_set(x_true[-1]),
            uc0,
            ub0,
            e0
        )
        
        # retrieve closed-loop trajectory
        x_predicted.append(solution.variables['x'][1])
        x_true.append(x1)
        x_sim.extend(x_sim_t)
        u.append(u0)
        e.append(e0)
        V.append(solution.objective)