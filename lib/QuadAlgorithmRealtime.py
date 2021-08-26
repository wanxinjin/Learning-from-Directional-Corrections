#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/LFC')
sys.path.append(os.getcwd()+'/lib')
import LFC
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import json
import math
import time
import transforms3d
from pynput import keyboard
from dataclasses import dataclass, field
from QuadStates import QuadStates
from QuadPara import QuadPara
from QuadrotorRealtime import QuadrotorRealtime


def remove_traj_ref_lib(Directory):
    """
    Delete all the files under a specific directory

    Input:
        Directory: A string for the directory

    Output:
        void

    Usage:
        Directory_delete = '/home/blah/RTD_Mambo_interface/traj_csv_files/*'
        remove_traj_ref_lib(Directory_delete)
    """
    files = glob.glob(Directory)
    for f in files:
        os.remove(f)


class QuadAlgorithmRealtime:
    QuadPara: QuadPara  # the dataclass QuadPara including the quadrotor parameters
    ini_state: list  # initial states for in a 1D list, [posi, velo, quaternion, angular_velo]
    weights_trace: list  # 2D list for weights trajectory during the iteration, each sub-list is a weight vector/list
    corrections_trace: list
    correction_time_trace: list
    config_data: dict  # a dictionary object for configurations of Mambo-Tracking-Interface
    time_step: float  # the length of time step for optimal controller
    time_scale: float  # scaling the time trajectory
    case_num: int  # the case number for demo

    def __init__(self, QuadParaInput: QuadPara, time_step=0.1, time_scale=1, case_num=1):
        """
        constructor
        """
        self.QuadPara = QuadParaInput
        self.time_step = time_step
        self.time_scale = time_scale
        self.case_num = case_num

        # load the configuration as a dictionary
        json_file = open(os.getcwd() + "/experiments/config_aimslab.json")
        self.config_data = json.load(json_file)

        # remove all the existing files in the trajectory directory
        directory_delete = os.path.expanduser("~") + "/Mambo-Tracking-Interface" + self.config_data["DIRECTORY_TRAJ"] + "*"
        remove_traj_ref_lib(directory_delete)

    def settings(self, QuadDesiredStates: QuadStates):
        """
        Do the settings and defined the goal states.
        Rerun this function everytime the initial condition or goal states change.
        """
        # load environment
        self.env = QuadrotorRealtime(time_step=self.time_step, time_scale=self.time_scale, case_num=self.case_num)
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
        dyn = self.env.X + self.time_step * self.env.f
        self.oc.setDyn(dyn)
        self.oc.setPathCost(features=features, weights=weights)
        self.oc.setFinalCost(10 * self.env.final_cost)

        # initialize the MVE solver
        self.mve = LFC.MVE()
        self.mve.initSearchRegion(x_lb=[0,-8,0,-8,0,-4,0], x_ub=[1,8,1,8,1,4,2])
        self.weights_trace = []
        self.corrections_trace = []
        self.correction_time_trace = []

    def run(self, QuadInitialCondition: QuadStates, QuadDesiredStates: QuadStates, iter_num: int, time_horizon: float, save_flag: bool):
        """
        Run the algorithm.

        Input:
            time_horizon: the time horizon for optimal controller [seconds]
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
            num_steps_horizon = int(time_horizon / self.time_step)
            opt_sol = self.oc.ocSolver(ini_state=self.ini_state, horizon=num_steps_horizon, weights=current_guess, time_step=self.time_step)
            
            # state_traj is a time_step by states numpy 2d array, each row is [positions *3, velocities *3, quaternion *4, angular velocities *3]
            state_traj = opt_sol['state_traj_opt']
            # time_traj is a numpy 1d array for timestamps
            time_traj = opt_sol['time'] * self.time_scale

            # save the trajectory
            traj_csv = np.vstack((time_traj, state_traj[:, 0:6].transpose()))
            if save_flag:
                filename_csv = os.path.expanduser("~") + "/Mambo-Tracking-Interface" + \
                    self.config_data["DIRECTORY_TRAJ"] + time.strftime("%Y%m%d%H%M%S") + ".csv"
                np.savetxt(filename_csv, traj_csv, delimiter=",")

            # plot the execution and accept the human correction from GUI interface
            human_interface = self.env.human_interface(state_traj, obstacles=True)

            t0 = time.time()
            if not human_interface:
                self.weights_trace.append(current_guess)
                print("No human corrections. Repeat the previous one.")
            else:
                correction, correction_time = self.env.interface_interpretation(human_interface, num_steps_horizon)
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

            print("Press n to next iteration")
            while True:
                with keyboard.Events() as events:
                    event = events.get(3)
                    if event is not None:
                        if type(event) is keyboard.Events.Press:
                            if event.key == keyboard.KeyCode(char='n'):
                                break
