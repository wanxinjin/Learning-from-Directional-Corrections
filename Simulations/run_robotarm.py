#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/LFC')
sys.path.append(os.getcwd()+'/JinEnv')
sys.path.append(os.getcwd()+'/lib')
from LFC import LFC
from JinEnv import JinEnv
import numpy as np
from casadi import *
import matplotlib.pyplot as plt
import scipy.io as sio

# loading the environment
env = JinEnv.RobotArm()
env.initDyn(l1=1, m1=1, l2=1, m2=1, g=0)
# define the cost function: weights and features
features = vcat([(env.X[0])**2, env.X[0], (env.X[1])**2, env.X[1], dot(env.U,env.U) ])
weights = SX.sym('weights', features.shape)
# define the final cost
final_cost = 100*((env.X[0]-pi/2)**2+(env.X[1])**2+(env.X[2])**2+(env.X[3])**2)




# load the oc solver object
oc = LFC.OCSys()
oc.setStateVariable(env.X)
oc.setControlVariable(env.U)
dt = 0.2
dyn = env.X + dt * env.f
oc.setDyn(dyn)
oc.setPathCost(features=features, weights=weights)
oc.setFinalCost(final_cost)

# set the initial condition and horizon
init_x = [0,0,0,0]
horizon = 50
true_weights = np.array([1,1,1,1,1])
opt_sol = oc.ocSolver(ini_state=init_x, horizon=horizon, weights=true_weights)
# env.play_animation(l1=1,l2=1,dt=dt,state_traj=opt_sol['state_traj_opt'])

# initialize the MVE solver
mve = LFC.MVE()
mve.initSearchRegion(x_lb=[0,0,0,0,0], x_ub=[4,4,4,4,4])

# compute the maximum iteration time
epsilon = 1e-1
R = 1
r = 5
K = r*log(R/epsilon)/(-log(1-1/r))
print('maximum iteration:', K)

# generate the initial weight guess
mve_center, mve_C, = mve.mveSolver()
current_guess = mve_center
weights_trace = [current_guess]
corrections_trace = []
correction_time_trace = []
for k in range(int(K)):
    # generate the optimal trajectory based on current weights guess
    opt_sol = oc.ocSolver(ini_state=init_x, horizon=horizon, weights=current_guess)

    # use a dummy human to generate the input data
    correction_time = np.random.randint(0, horizon)
    mat_gradient1, mat_gradient2 = oc.getRecoveryMat(opt_sol=opt_sol)
    correction = -sign((np.matmul(mat_gradient1,true_weights)+mat_gradient2)[correction_time*oc.n_control:(correction_time+1)*oc.n_control])
    corrections_trace += [correction.full()]
    correction_time_trace += [correction_time]

    # generate the hyperplane from the correction information
    hyperplane_a, hyperplane_b = oc.getHyperplane(opt_sol=opt_sol, correction=correction, correction_time=correction_time)

    # add the hyperplane and generate the next weights guess
    mve.addHyperplane(hyperplane_a, -hyperplane_b)
    mve_center, mve_C, = mve.mveSolver()
    # if the MVE solver is down
    if mve_center is None:
        break
    # prepare for the next update
    current_guess = mve_center
    weights_trace += [current_guess]

    print('iter:', k, '--- current guess:', current_guess,)

# # save the reuslts
# print(corrections_trace)
# results = {'weights_trace':weights_trace,
#             'true_weights':true_weights,
#             'correction_time_trace':correction_time_trace,
#             'corrections_trace':corrections_trace}
# sio.savemat('data/robotarm2_results.mat', results)



# plot the results
error_trace = np.linalg.norm(np.array(weights_trace)-true_weights,axis=1)**2
params = {'axes.labelsize': 25,
            'axes.titlesize': 25,
            'xtick.labelsize':20,
            'ytick.labelsize':20,
            'legend.fontsize':16}
plt.rcParams.update(params)

fig = plt.figure(0,figsize=(8,3.5))
ax = fig.subplots()
ax.plot(error_trace,linewidth=4)
ax.set_facecolor('#E6E6E6')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel(r'$e_{\theta}$')
ax.grid()
ax.set_position(pos=[0.11,0.20,0.85,0.78])

plt.show()