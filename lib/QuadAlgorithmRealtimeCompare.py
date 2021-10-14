#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/LFC')
sys.path.append(os.getcwd()+'/lib')
import LFC
from casadi import *
import numpy as np
import glob
import json
import math
import time
from pynput import keyboard
from QuadStates import QuadStates
from QuadPara import QuadPara
from QuadrotorRealtime import QuadrotorRealtime


def remove_traj_ref_lib(directory: str):
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
    files = glob.glob(directory)
    for f in files:
        os.remove(f)


class QuadAlgorithmRealtimeCompare:
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

        try:
            # remove all the existing files in the trajectory directory
            directory_delete = os.path.expanduser("~") + "/Mambo-Tracking-Interface" + \
                            self.config_data["DIRECTORY_TRAJ"] + "*"
            remove_traj_ref_lib(directory_delete)
        except:
            print("Failed to deleter files in: ", directory_delete)

    def settings(self, QuadDesiredStates: QuadStates):
        """
        Do the settings and defined the goal states.
        Rerun this function everytime the initial condition or goal states change.
        """
        # load environment
        self.env = QuadrotorRealtime(time_step=self.time_step, time_scale=self.time_scale, case_num=self.case_num)
        self.env.initDyn(Jx=self.QuadPara.inertial_x, Jy=self.QuadPara.inertial_y, Jz=self.QuadPara.inertial_z,
                         mass=self.QuadPara.mass, l=self.QuadPara.l, c=self.QuadPara.c)

        # define the cost function: weights and features
        features = vcat([(self.env.X[0])**2, self.env.X[0],
                        (self.env.X[1])**2, self.env.X[1],
                        (self.env.X[2])**2, self.env.X[2], dot(self.env.U, self.env.U)])
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

        # initialize the parameter
        parameter_lb = np.array([0, -8, 0, -8, 0, -4, 0])
        parameter_ub = np.array([1, 8, 1, 8, 1, 4, 2])
        self.weights_trace = []
        self.corrections_trace = []
        self.correction_time_trace = []
        self.initial_parameter = 0.5 * (parameter_lb + parameter_ub)

    def run(self, QuadInitialCondition: QuadStates, QuadDesiredStates: QuadStates, iter_num: int,
            time_horizon: float, save_flag: bool):
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

        # set the initial parameter guess
        current_guess = self.initial_parameter
        # print("Current guess: ", current_guess)

        # compute the M matrix used to generate the intended trajectory
        inv_M = self.compute_matrix_intended_traj(time_horizon=time_horizon)

        # iter_num is the maximum iteration number
        for k in range(iter_num):
            # generate the optimal trajectory based on current weights guess
            num_steps_horizon = int(time_horizon / self.time_step)
            opt_sol = self.oc.ocSolver(ini_state=self.ini_state, horizon=num_steps_horizon,
                                       weights=current_guess, time_step=self.time_step)
            
            # state_traj is a time_step by states numpy 2d array, each row is
            # [positions *3, velocities *3, quaternion *4, angular velocities *3]
            state_traj = opt_sol['state_traj_opt']
            # time_traj is a numpy 1d array for timestamps
            time_traj = opt_sol['time'] * self.time_scale
            # input_traj is a trajectory of inputs
            input_traj = opt_sol['control_traj_opt']

            # save the trajectory
            traj_csv = np.vstack((time_traj, state_traj[:, 0:6].transpose()))
            if save_flag:
                filename_csv = os.path.expanduser("~") + "/Mambo-Tracking-Interface" + \
                    self.config_data["DIRECTORY_TRAJ"] + time.strftime("%Y%m%d%H%M%S") + ".csv"
                np.savetxt(filename_csv, traj_csv, delimiter=",")

            # plot the execution and accept the human correction from GUI interface
            human_interface = self.env.human_interface(state_traj, obstacles=True)

            # initialize the corrections over the whole horizon
            corrections_all = np.zeros(input_traj.shape)

            t0 = time.time()
            if not human_interface:
                self.weights_trace.append(current_guess)
                print("No human corrections. Repeat the previous one.")
            else:
                correction, correction_time = self.env.interface_interpretation(human_interface, num_steps_horizon)
                self.corrections_trace.append(correction)
                self.correction_time_trace.append(correction_time)

                # load all the corrections into corrections_all
                for idx in range(len(correction_time)):
                    time_index = correction_time[idx]
                    corrections_all[time_index] = 0.002 * correction[idx]
                input_traj_intended = input_traj + np.matmul(inv_M, corrections_all)
                # solve the feature vector
                old_features = self.compute_features(input_traj, self.ini_state)
                new_features = self.compute_features(input_traj_intended, self.ini_state)
                # update the parameters
                step_length = np.array([1E-8, 1E-3, 1E-8, 5E-2, 1E-8, 3E-2, 1E-2])
                current_guess = current_guess - np.multiply(step_length, (new_features - old_features))
                # current_guess = current_guess - 1E-3 * (new_features - old_features)
                # print("Current guess: ", current_guess)
                self.weights_trace.append(current_guess)

            t1 = time.time()
            print("iter:", k, ", time used [sec]: ", math.floor((t1-t0)*1000)/1000.0)

            print("Press n to next iteration")
            while True:
                with keyboard.Events() as events:
                    event = events.get(10)
                    if event is not None:
                        if type(event) is keyboard.Events.Press:
                            if event.key == keyboard.KeyCode(char='n'):
                                break

    def compute_features(self, traj_u, init_x):
        """
        Define a utility function to compute features.
        """
        current_x = init_x
        sum_features = 0.0
        for u in traj_u:
            sum_features += self.oc.feature_fn(current_x, u).full().flatten()
            current_x = self.oc.dyn_fn(current_x, u).full().flatten()
        return sum_features

    def compute_matrix_intended_traj(self, time_horizon: float):
        """
        Compute the M matrix used to generate the intended trajectory.
        """
        horizon = int(time_horizon / self.time_step)
        M1 = np.eye(horizon)
        M2 = -1 * np.eye(horizon - 1)
        M2 = np.hstack((np.zeros((horizon - 1, 1)), M2))
        M2 = np.vstack((M2, np.zeros((1, horizon)),))
        M = M1 + M2
        M = M + np.transpose(M)
        inv_M = np.linalg.inv(M)
        return inv_M
