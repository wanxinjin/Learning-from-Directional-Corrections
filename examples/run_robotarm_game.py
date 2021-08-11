#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/LFC')
sys.path.append(os.getcwd()+'/JinEnv')
sys.path.append(os.getcwd()+'/lib')
import LFC
import JinEnv
import numpy as np
from casadi import *
import matplotlib.pyplot as plt
import scipy.io as sio


# loading the environment
env = JinEnv.RobotArm()
env.initDyn(l1=1, m1=1, l2=1, m2=1, g=0)
# define the cost function: weights and features
features = vcat([(env.X[0])**2, env.X[0], (env.X[1])**2, env.X[1], dot(env.U,env.U) ])
weights = SX.sym('weights',features.shape)
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
ini_state = [-pi/2,0,0,0]
horizon = 50
# true_weights = np.array([0,0,0,0,1.])
# opt_sol = oc.ocSolver(ini_state=ini_state, horizon=horizon, weights=true_weights)
# env.play_animation(l1=1, l2=1, dt=dt, state_traj=opt_sol['state_traj_opt'])


# initialize the MVE solver
mve = LFC.MVE()
mve.initSearchRegion(x_lb=[0,-3,0,-3,0], x_ub=[1,3,1,3,0.5])
# mve.initSearchRegion(x_lb=[-3,-3,-3,-3,0], x_ub=[3,3,3,3,0.5])

# generate the initial weight guess
mve_center, mve_C, = mve.mveSolver()
current_guess = mve_center
weights_trace = [current_guess]
corrections_trace = []
correction_time_trace = []
# max learning iteration
for k in range(int(10)):
    # generate the optimal trajectory based on current weights guess
    opt_sol = oc.ocSolver(ini_state=ini_state, horizon=horizon, weights=current_guess)
    state_traj = opt_sol['state_traj_opt']
    # plot the execution and accept the human correction from GUI interface
    human_interface = env.human_interface(l1=1, l2=1, state_traj=state_traj, obstacles=True)
    if not human_interface:
        current_guess = mve_center
        weights_trace += [current_guess]
    else:
        correction, correction_time = env.interface_interpretation(human_interface,horizon)
        # generate the hyperplane from the correction information
        hyperplane_a, hyperplane_b = oc.getHyperplane(opt_sol=opt_sol, correction=correction, correction_time=correction_time)
        # add the hyperplane and generate the next weights guess
        mve.addHyperplane(hyperplane_a, -hyperplane_b)
        mve_center, mve_C, = mve.mveSolver()
        # if the MVE solver is down
        if mve_center is None:
            break
        current_guess = mve_center
        weights_trace += [current_guess]

    print('iter:', k, )

