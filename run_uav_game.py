from LFC import LFC
from JinEnv import JinEnv
import numpy as np
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as plt
import time


env=JinEnv.Quadrotor()
env.initDyn(Jx=0.5,Jy=0.5,Jz=1,mass=1,l=1,c=0.1)
# define the cost function: weights and features
features=vcat([(env.X[0])**2, env.X[0], (env.X[1])**2, env.X[1], (env.X[2])**2, env.X[2], dot(env.U,env.U)])
weights=SX.sym('weights',features.shape)
# define the final cost function
goal_r_I = np.array([8, 8, 0])
env.initFinalCost(goal_r_I=goal_r_I)


# load the oc solver object
oc=LFC.OCSys()
oc.setStateVariable(env.X)
oc.setControlVariable(env.U)
dt=0.1
dyn=env.X+dt*env.f
oc.setDyn(dyn)
oc.setPathCost(features=features,weights=weights)
oc.setFinalCost(10*env.final_cost)

# set the initial condition and horizon
ini_r_I = [-8, -8, 5.]
ini_v_I = [0, 0, 0]
ini_q = JinEnv.toQuaternion(0, [0, 0, 1])

ini_w = [0.0, 0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w
horizon=50


# initialize sthe MVE solver
mve=LFC.MVE()
mve.initSearchRegion(x_lb=[0,-8,0,-8,0,-8,0], x_ub=[1,8,1,8,1,8,0.5])
# generate the initial weight guess
mve_center,mve_C,=mve.mveSolver()
current_guess=mve_center
weights_trace=[current_guess]
corrections_trace=[]
correction_time_trace=[]
# max learning iteration
for k in range(int(20)):
    # generate the optimal trajectory based on current weights guess
    opt_sol=oc.ocSolver(ini_state=ini_state,horizon=horizon,weights=current_guess)
    state_traj = opt_sol['state_traj_opt']
    # plot the execution and accept the human correction from GUI interface
    human_interface=env.human_interface(state_traj,obstacles=True)
    if not human_interface:
        current_guess = mve_center
        weights_trace += [current_guess]
        # corrections_trace+=[100*np.ones(oc.n_control)]
        # correction_time_trace+=[100]
    else:
        correction, correction_time=env.interface_interpretation(human_interface,horizon)
        # corrections_trace+=[correction]
        # correction_time_trace+=[correction_time]
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

# # save the reuslts
# print(corrections_trace)
# results={'weights_trace':weights_trace,
#          'correction_time_trace':correction_time_trace,
#          'corrections_trace':corrections_trace}
# sio.savemat('data/uav_results.mat', results)
