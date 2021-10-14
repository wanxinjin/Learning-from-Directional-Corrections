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

def compute_features(traj_u, init_x, oc):
    """
    Define a utility function to compute features.
    """
    current_x = init_x
    sum_features = 0.0
    for u in traj_u:
        sum_features += oc.feature_fn(current_x, u).full().flatten()
        current_x = oc.dyn_fn(current_x, u).full().flatten()
    return sum_features

def compute_matrix_intended_traj(time_horizon: float, time_step: float):
    """
    Compute the M matrix used to generate the intended trajectory.
    """
    horizon = int(time_horizon / time_step)
    M1 = np.eye(horizon)
    M2 = -1 * np.eye(horizon - 1)
    M2 = np.hstack((np.zeros((horizon - 1, 1)), M2))
    M2 = np.vstack((M2, np.zeros((1, horizon)),))
    M = M1 + M2
    M = M + np.transpose(M)
    inv_M = np.linalg.inv(M)
    return inv_M

# maximum iteration number
iter_num = 30
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

# set the initial condition and time horizon
ini_state = [-pi/2, 0, 0, 0]
time_horizon = 10  # seconds
num_steps_horizon = int(time_horizon / dt)
# true_weights = np.array([0, 0, 0, 0, 1.])
# true_weights = [0.5, -1.49335936, 0.5, -1.71562772, 0.25]
# opt_sol = oc.ocSolver(ini_state=ini_state, horizon=num_steps_horizon, time_step=dt, weights=true_weights)
# env.play_animation(l1=1, l2=1, dt=dt, state_traj=opt_sol['state_traj_opt'])

# initialize the parameter
parameter_lb = np.array([0, -3, 0, -3, 0])
parameter_ub = np.array([1, 3, 1, 3, 0.5])
initial_parameter = 0.5 * (parameter_lb + parameter_ub)

# generate the initial weight guess
current_guess = initial_parameter
# print("Current guess: ", current_guess)
weights_trace = [current_guess]
corrections_trace = []
correction_time_trace = []

# compute the M matrix used to generate the intended trajectory
inv_M = compute_matrix_intended_traj(time_horizon=time_horizon, time_step=dt)

# max learning iteration
for k in range(iter_num):
    # generate the optimal trajectory based on current weights guess
    opt_sol = oc.ocSolver(ini_state=ini_state, horizon=num_steps_horizon, time_step=dt, weights=current_guess)
    state_traj = opt_sol['state_traj_opt']
    input_traj = opt_sol['control_traj_opt']
    # plot the execution and accept the human correction from GUI interface
    human_interface = env.human_interface(l1=1, l2=1, state_traj=state_traj, obstacles=True)

    # initialize the corrections over the whole horizon
    corrections_all = np.zeros(input_traj.shape)

    if not human_interface:
        weights_trace += [current_guess]
        print("No human corrections. Repeat the previous one.")
    else:
        correction, correction_time = env.interface_interpretation(human_interface, num_steps_horizon)
        # load all the corrections into corrections_all
        for idx in range(len(correction_time)):
            time_index = correction_time[idx]
            corrections_all[time_index] = 0.005 * correction[idx]
        input_traj_intended = input_traj + np.matmul(inv_M, corrections_all)
        # solve the feature vector
        old_features = compute_features(input_traj, ini_state, oc)
        new_features = compute_features(input_traj_intended, ini_state, oc)
        # update the parameters
        step_length = np.array([1E-8, 5E-3, 1E-8, 5E-3, 1E-8])
        current_guess = current_guess - np.multiply(step_length, (new_features - old_features))
        # current_guess = current_guess - 1E-3 * (new_features - old_features)
        # print("Current guess: ", current_guess)
        weights_trace += [current_guess]

    print('iter:', k, )
