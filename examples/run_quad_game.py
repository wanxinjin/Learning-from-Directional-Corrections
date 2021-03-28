#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/LFC')
sys.path.append(os.getcwd()+'/JinEnv')
sys.path.append(os.getcwd()+'/lib')
import numpy as np
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as pltdad
import time
import transforms3d
from dataclasses import dataclass, field
from QuadAlgorithm import QuadAlgorithm
from QuadStates import QuadStates
from QuadPara import QuadPara


if __name__ == '__main__':

    # define the quadrotor dynamics parameters
    QuadParaInput = QuadPara(inertial_list=[1.0, 1.0, 1.0], mass=1.0, l=1.0, c=0.02)

    # define the initial condition
    R = np.array([[1,0,0],[0,1,0],[0,0,1]]) # rotation matrix in numpy 2D array
    QuadInitialCondition = QuadStates()
    QuadInitialCondition.position = [-8, -8, 2]
    QuadInitialCondition.velocity = [0, 0, 0]
    QuadInitialCondition.attitude_quaternion = transforms3d.quaternions.mat2quat(R).tolist()
    QuadInitialCondition.angular_velocity = [0, 0, 0]

    # define the desired goal
    R = np.array([[1,0,0],[0,1,0],[0,0,1]]) # rotation matrix in numpy 2D array
    QuadDesiredStates = QuadStates()
    QuadDesiredStates.position = [8, 2, 4]
    QuadDesiredStates.velocity = [0, 0, 0]
    QuadDesiredStates.attitude_quaternion = transforms3d.quaternions.mat2quat(R).tolist()
    QuadDesiredStates.angular_velocity = [0, 0, 0]

    # create the quadrotor algorithm solver
    Solver = QuadAlgorithm(QuadParaInput)

    # solve itd
    Solver.run(QuadInitialCondition, QuadDesiredStates, \
        iter_num=50, horizon=40, save_flag=False)
    