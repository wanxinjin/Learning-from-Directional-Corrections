#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/LFC')
sys.path.append(os.getcwd()+'/JinEnv')
sys.path.append(os.getcwd()+'/lib')
import LFC
import JinEnv
from casadi import *
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import time
import transforms3d
from dataclasses import dataclass, field
from QuadStates import QuadStates
from QuadPara import QuadPara


class QuadAlgorithm(object):
    QuadPara: QuadPara # the dataclass QuadPara including the quadrotor parameters
    ini_state: list # initial states for in a 1D list, [posi, velo, quaternion, angular_velo]
    weights_trace: list # 2D list for weights trajectory during the iteration, each sub-list is a weight vector/list
    corrections_trace: list
    correction_time_trace: list


    def __init__(self, QuadParaInput: QuadPara):
        """
        constructor

        """
        self.QuadPara = QuadParaInput


    def settings(self, QuadDesiredStates: QuadStates):
        """
        Do the settings and defined the goal states.
        Rerun this function everytime the initial condition or goal states change.
        """

        # load environment
        self.env = JinEnv.Quadrotor()
        self.env.initDyn(Jx=self.QuadPara.inertial_x, Jy=self.QuadPara.inertial_y, Jz=self.QuadPara.inertial_z, \
            mass=self.QuadPara.mass, l=self.QuadPara.l, c=self.QuadPara.c)

        # define the cost function: weights and features
        features = vcat([(self.env.X[0])**2, self.env.X[0], \
            (self.env.X[1])**2, self.env.X[1], \
                (self.env.X[2])**2, self.env.X[2], dot(self.env.U,self.env.U)])
        weights = SX.sym('weights', features.shape)
        # define the final cost function
        self.env.initFinalCost(QuadDesiredStates)

        # load the oc solver object
        self.oc = LFC.OCSys()
        self.oc.setStateVariable(self.env.X)
        self.oc.setControlVariable(self.env.U)
        dt = 0.1
        dyn = self.env.X + dt * self.env.f
        self.oc.setDyn(dyn)
        self.oc.setPathCost(features=features, weights=weights)
        self.oc.setFinalCost(10 * self.env.final_cost)

        # initialize sthe MVE solver
        self.mve = LFC.MVE()
        self.mve.initSearchRegion(x_lb=[0,-8,0,-8,0,-8,0], x_ub=[1,8,1,8,1,8,0.5])
        self.weights_trace = []
        self.corrections_trace = []
        self.correction_time_trace = []


    def run(self, QuadInitialCondition: QuadStates, QuadDesiredStates: QuadStates, iter_num: int, horizon: float, save_flag: bool):
        """
        Run the algorithm.
        """

        t0 = time.time()
        print("Algorithm is running now.")
        # set the goal states
        self.settings(QuadDesiredStates)
        # set initial condition
        self.ini_state = QuadInitialCondition.position + QuadInitialCondition.velocity + \
            QuadInitialCondition.attitude_quaternion + QuadInitialCondition.angular_velocity

        # generate the initial weight guess
        mve_center, mve_C = self.mve.mveSolver()
        current_guess = mve_center

        # iter_num is the maximum iteration number
        for k in range(iter_num):
            # generate the optimal trajectory based on current weights guess
            opt_sol = self.oc.ocSolver(ini_state=self.ini_state, horizon=horizon, weights=current_guess)
            state_traj = opt_sol['state_traj_opt']
            
            # plot the execution and accept the human correction from GUI interface
            human_interface = self.env.human_interface(state_traj, obstacles=True)

            t0 = time.time()
            if not human_interface:
                self.weights_trace.append(current_guess)
                print("No human corrections. Repeat the previous one.")
            else:
                correction, correction_time = self.env.interface_interpretation(human_interface, horizon)
                self.corrections_trace.append(correction)
                self.correction_time_trace.append(correction_time)

                # generate the hyperplane from the correction information
                hyperplane_a, hyperplane_b = self.oc.getHyperplane(opt_sol=opt_sol, correction=correction, correction_time=correction_time)
                
                # add the hyperplane and generate the next weights guess
                self.mve.addHyperplane(hyperplane_a, -hyperplane_b)
                mve_center, mve_C, = self.mve.mveSolver()

                # if the MVE solver is done
                if mve_center is None:
                    break
                current_guess = mve_center
                self.weights_trace.append(current_guess)

            t1 = time.time()
            print("iter:", k, ", time used [sec]: ", math.floor((t1-t0)*1000)/1000.0)

        # save the reuslts
        if save_flag:
            time_prefix = time.strftime("%Y%m%d%H%M%S")
            results = {'weights_trace': self.weights_trace,
                        'correction_time_trace': self.correction_time_trace,
                        'corrections_trace': self.corrections_trace}

            # save the results as mat files
            name_prefix_mat = os.getcwd() + '/data/uav_results_random_' + time_prefix
            sio.savemat(name_prefix_mat + '.mat', {'results': results})
