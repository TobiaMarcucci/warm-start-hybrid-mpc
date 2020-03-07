# external imports
import numpy as np
import matplotlib.pyplot as plt

# internal imports
from controller import controller
from mld_dynamics import mld, h, x_min, x_max, fc_min, fc_max, d, l

# function that retunrs how many successful trajectories
# there are in a list
def number_success(trajectories, horizon):
    return sum(1 for t in trajectories if len(t) == horizon)

# initial push towards the right wall
x0 = np.array([0., 0., 1., 0.])

# error standard deviation
e_sd = 0.# 0.001, 0.003, 0.01

# legth of each simulation
N_sim = 50

# number of simulations
N_samples = 1

# initial counter for the simulations
# if there are already stored data this allows to continue
# the collection from where it stopped last time
# set this value to the number of trajectories you already have
# included unsuccesful ones
i0 = 0

# list of the data to be collected
nodes_keys = ['cs', 'ws', 'len_ws', 'grb', 'grb_fair']
times_keys = ['cs', 'ws', 'ws_constr', 'grb', 'grb_fair']

# initialize nodes, times, and errors to empty lists if
# we do not want to load previous data
if i0 == 0:
    nodes = {k: [] for k in nodes_keys}
    times = {k: [] for k in times_keys}
    errors = []

# continue collecting data loading old ones
else:
    nodes = {k: list(np.load('data/nodes_'+k+'_sd_{:.3f}.npy'.format(e_sd), allow_pickle=True)) for k in nodes_keys}
    times = {k: list(np.load('data/times_'+k+'_sd_{:.3f}.npy'.format(e_sd), allow_pickle=True)) for k in times_keys}
    errors = list(np.load('data/errors_sd_{:.3f}.npy'.format(e_sd), allow_pickle=True))
 
# needed to print the log
if i0 == 0:
    open_mode = 'w'
else:
    open_mode = 'a'
with open('data/solve_log_sd_{:.3f}.log'.format(e_sd), open_mode) as f:

    # prints the title to the log file
    # this prints to the log file
    if i0 == 0:
        f.write('Error standard deviation {:.3f}\n\n'.format(e_sd)) 
        f.flush()

    # loop until N_samples success has been collected
    # note that for big disturbances the regulation problem
    # can become infeasible and the simulation is discarded
    i = i0
    while number_success(nodes['cs'], N_sim) < N_samples:

        # select different random seeed for each sim
        np.random.seed(i)

        # initialize simulation in the log file
        n_sim = number_success(nodes['cs'], N_sim)
        f.write(f'\n\nSimulation {i}\n')
        f.write(f'Simulation success {n_sim}\n\n')
        f.flush() # forces printing to log

        # flag that determines if the simulation had success
        simulation_success = True

        # data of the current simulation
        nodes_i = {k: [] for k in nodes_keys}
        times_i = {k: [] for k in times_keys}
        errors_i = []

        # simulate for N_sim steps
        x_sim = [x0]
        ws = None
        for t in range(N_sim):
            print((i,t), end='\r')
            f.write('Time step %d '%t)
            f.flush()

            # solve with cold start
            try:
                solution_cs, leaves_cs, nodes_cs, time_cs = controller.feedforward(
                    x_sim[-1],
                    printing_period=None,
                    gurobi_options={'Method': 1} # dual simplex
                )
            except:
                simulation_success = False
                f.write('\nUnseccessful simulation, cold-started BB broke!')
                break
            nodes_i['cs'].append(nodes_cs)
            times_i['cs'].append(time_cs)
            f.write('(cs: {}, {:.3f}) '.format(nodes_cs, time_cs))
            f.flush()

            # solve with warm start
            try:
                solution_ws, leaves_ws, nodes_ws, time_ws = controller.feedforward(
                    x_sim[-1],
                    warm_start=ws,
                    printing_period=None,
                    gurobi_options={'Method': 1} # dual simplex
                )
            except:
                simulation_success = False
                f.write('\nUnseccessful simulation, warm-started BB broke!')
                break
            nodes_i['ws'].append(nodes_ws)
            times_i['ws'].append(time_ws)
            f.write('(ws: {}, {:.3f}) '.format(nodes_ws, time_ws))
            f.flush()
            
            # solve with gurobi
            try:
                x_grb, cost_grb, nodes_grb, time_grb = controller.feedforward_gurobi(
                    x_sim[-1],
                    {'OutputFlag': 0, 'MIPGap': 0}
                )
            except:
                simulation_success = False
                f.write('\nUnseccessful simulation, Gurobi broke!')
                break
            nodes_i['grb'].append(nodes_grb)
            times_i['grb'].append(time_grb)
            f.write('(grb: {}, {:.3f}) '.format(nodes_grb, time_grb))
            f.flush()
            
            # solve with gurobi fair
            # gurobi parameters are set to be "fair" with our branch and bound
            try:
                x_grb_fair, cost_grb_fair, nodes_grb_fair, time_grb_fair = controller.feedforward_gurobi(
                    x_sim[-1],
                    {'OutputFlag': 0, 'MIPGap': 0, 'Presolve': 0, 'Heuristics': 0, 'Threads': 1}
                )
            except:
                simulation_success = False
                f.write('\nUnseccessful simulation, Gurobi fair broke!')
                break
            nodes_i['grb_fair'].append(nodes_grb_fair)
            times_i['grb_fair'].append(time_grb_fair)
            f.write('(grb_fair: {}, {:.3f}) '.format(nodes_grb_fair, time_grb_fair))
            f.flush()

            # break if unfeasible
            # but still store the solution.
            if solution_cs is None:
                break

            # check that all the optimal cost coincide
            # (no infeasible at this point)
            assert np.isclose(solution_cs.objective, solution_ws.objective)
            assert np.isclose(solution_cs.objective, cost_grb)
            assert np.isclose(solution_cs.objective, cost_grb_fair)

            # generate random error of specified norm
            e_t = e_sd * np.multiply(np.random.randn(mld.nx), x_max)
            errors_i.append(e_t)

            # generate warm start
            ws, time_ws_constr = controller.construct_warm_start(
                leaves_ws,
                solution_ws.variables['x'][0],
                solution_cs.variables['uc'][0],
                solution_cs.variables['ub'][0],
                e_t
            )
            nodes_i['len_ws'].append(len(ws))
            times_i['ws_constr'].append(time_ws_constr)
            f.write('(ws info: {}, {:.3f}) '.format(len(ws), time_ws_constr))
            f.flush()

            # next state
            x_sim.append(solution_ws.variables['x'][1] + e_t)
            f.write('(e: {:.3f}, {})\n'.format(np.linalg.norm(e_t), e_t))
            f.flush()

        # store data if simulation is not infeasible or broken
        if simulation_success:
            for k in nodes_keys:
                nodes[k].append(nodes_i[k])
                np.save('data/nodes_'+k+'_sd_{:.3f}.npy'.format(e_sd), nodes[k])
            for k in times_keys:
                times[k].append(times_i[k])
                np.save('data/times_'+k+'_sd_{:.3f}.npy'.format(e_sd), times[k])
            errors.append(errors_i)
            np.save('data/errors_sd_{:.3f}.npy'.format(e_sd), errors)

        # increase counter even if the simulation is not successful
        i += 1
