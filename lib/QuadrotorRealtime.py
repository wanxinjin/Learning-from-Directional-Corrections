#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/JinEnv')
from JinEnv import Quadrotor
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch
import math
import time
from pynput import keyboard
from dataclasses import dataclass, field
from QuadStates import QuadStates


class QuadrotorRealtime(Quadrotor):
    def human_interface(self, state_traj, obstacles=False):
        plt.rcParams['keymap.save'] = []
        # fig = plt.figure()
        plt.close()
        fig = plt.figure(figsize=(8,5))
        # main view
        gs = fig.add_gridspec(2, 4)
        ax = fig.add_subplot(gs[:,0:3], projection='3d')
        ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
        ax.set_zlabel('Z (m)', fontsize=10, labelpad=5)
        ax.set_zlim(0, 3)
        ax.set_ylim(-1.8, 1.6)
        ax.set_xlim(-2.4, 2.4)
        ax.set_title('6-DoF Quadrobot Game', pad=00, fontsize=15)
        ax.view_init(elev=25., azim=-70)
        # top view
        ax_top = fig.add_subplot(gs[0,3])
        ax_top.set_xlabel('X (m)', fontsize=10, labelpad=4)
        ax_top.set_ylabel('Y (m)', fontsize=10, labelpad=10)
        ax_top.set_ylim(-1.8, 1.6)
        ax_top.set_xlim(-2.4, 2.4)
        ax_top.set_title('XOY Plane', y=1, fontsize=15)

        # front view
        ax_front = fig.add_subplot(gs[1,3])
        ax_front.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax_front.set_ylabel('Z (m)', fontsize=10, labelpad=15)
        ax_front.set_ylim(0, 3)
        ax_front.set_xlim(-2.4, 2.4)
        ax_front.set_title('XOZ Plane', y=0.85,  fontsize=15)

        plt.subplots_adjust(left=0, right=0.90, wspace=-0.0, hspace=0.1)

        # data
        position = self.get_quadrotor_position(self.l, state_traj)
        sim_horizon = np.size(position, 0)

        # initial for main view
        line_traj, = ax.plot3D(position[:1, 0], position[:1, 1], position[:1, 2])
        line_quadrotor, = ax.plot3D(position[:1, 0], position[:1, 1], position[:1, 2], marker="o")

        # initial for top view
        line_traj_top, = ax_top.plot(position[:1, 0], position[:1, 1])
        line_quadrotor_top, = ax_top.plot(position[:1, 0], position[:1, 1], marker="o")

        # initial for front view
        line_traj_front, = ax_front.plot(position[:1, 0], position[:1, 2])
        line_quadrotor_front, = ax_front.plot(position[:1, 0], position[:1, 2], marker="o")

        # allowed key press
        directions = [keyboard.Key.up, keyboard.Key.down,
                      keyboard.KeyCode(char='w'), keyboard.KeyCode(char='s'),
                      keyboard.KeyCode(char='a'), keyboard.KeyCode(char='d')]

        with keyboard.Events() as events:
            # plot and ainimation
            human_interactions = []
            for num in range(sim_horizon):
                line_traj.set_data(position[:num, 0], position[:num, 1])
                line_traj.set_3d_properties(position[:num, 2])
                line_traj_top.set_data(position[:num, 0], position[:num, 1])
                line_traj_front.set_data(position[:num, 0], position[:num, 2])

                line_quadrotor.set_data(position[num, 0], position[num, 1])
                line_quadrotor.set_3d_properties(position[num, 2])
                line_quadrotor_top.set_data(position[num, 0], position[num, 1])
                line_quadrotor_front.set_data(position[num, 0], position[num, 2])

                # detect the human inputs
                inputs = []
                event = events.get(0.005)
                if event is not None:
                    if type(event) is keyboard.Events.Press and event.key in directions:
                        inputs.append(event.key)
                event = events.get(0.005)
                if event is not None: 
                        inputs.append(event.key)
                if len(inputs) != 0:
                    inputs.append(num)
                    human_interactions.append(inputs)

                    if inputs[-2] == keyboard.Key.up:
                        purpose_str = " Human wants to move upwards"
                    elif inputs[-2] == keyboard.Key.down:
                        purpose_str = " Human wants to move downwards"
                    elif inputs[-2] == keyboard.KeyCode(char='w'):
                        purpose_str = " Human wants to move faster"
                    elif inputs[-2] == keyboard.KeyCode(char='s'):
                        purpose_str = " Human wants to move slower"
                    elif inputs[-2] == keyboard.KeyCode(char='a'):
                        purpose_str = " Human wants to move leftwards"
                    else:
                        purpose_str = " Human wants to move rightwards"

                    print('Human action captured:', inputs[-2], purpose_str)

                plt.pause(0.01)
            return human_interactions
