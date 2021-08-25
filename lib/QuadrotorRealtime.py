#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/JinEnv')
from JinEnv import Quadrotor
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import time
from pynput import keyboard
from dataclasses import dataclass, field
from QuadStates import QuadStates


# def cuboid_data2(o, size=(1, 1, 1)):
#     """
#     A helper function to plot 3D cube.
#     """
#     X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
#          [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
#          [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
#          [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
#          [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
#          [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
#     X = np.array(X).astype(float)
#     for i in range(3):
#         X[:, :, i] *= size[i]
#     X += np.array(o)
#     return X

# def plotCubeAt2(positions, sizes=None, colors=None, **kwargs):
#     """
#     A helper function to plot 3D cube.
#     """
#     if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
#     if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
#     g = []
#     for p, s, c in zip(positions, sizes, colors):
#         g.append(cuboid_data2(p, size=s))
#     return Poly3DCollection(np.concatenate(g),  
#                             facecolors=np.repeat(colors,6), **kwargs)

class QuadrotorRealtime(Quadrotor):
    time_step: float  # the length of time step for optimal controller
    time_scale: float  # scaling the time trajectory

    def __init__(self, time_step=0.1, time_scale=1, project_name='my uav'):
        super().__init__(project_name)
        self.time_step = time_step
        self.time_scale = time_scale

    def human_interface(self, state_traj, obstacles=False):
        plt.rcParams['keymap.save'] = []
        plt.close()
        fig = plt.figure(figsize=(11, 8))
        # main view
        ax = fig.add_subplot(2, 2, (1,3), projection='3d')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_zlim(0, 3)
        ax.set_ylim(-1.8, 1.6)
        ax.set_xlim(-2.4, 2.4)
        ax.set_title('Quadrotor Game')
        ax.view_init(elev=25., azim=-70)
        # top view
        ax_top = fig.add_subplot(2, 2, 2)
        ax_top.axis("equal")
        ax_top.set_xlabel('X (m)')
        ax_top.set_ylabel('Y (m)')
        ax_top.set_ylim(-1.8, 1.6)
        ax_top.set_xlim(-2.4, 2.4)
        ax_top.set_title('XOY Plane', x=0.75)

        # front view
        ax_front = fig.add_subplot(2, 2, 4)
        ax_front.axis("equal")
        ax_front.set_xlabel('X (m)')
        ax_front.set_ylabel('Z (m)')
        ax_front.set_ylim(0, 3)
        ax_front.set_xlim(-2.4, 2.4)
        ax_front.set_title('XOZ Plane', x=0.75)

        # data
        position = self.get_quadrotor_position(self.l, state_traj)
        sim_horizon = np.size(position, 0)

        # obstacles
        positions = [(-3,5,-2),(1,7,1)]
        sizes = [(4,5,3), (3,3,7)]
        colors = ["crimson","limegreen"]
        # pc = plotCubeAt2(positions,sizes,colors=colors, edgecolor="k")



        obs_01_top = patches.Rectangle((-0.48, -1.5), 0.96, 1.0, linewidth=1, edgecolor="red", facecolor="red")
        obs_01_front = patches.Rectangle((-0.48, 0), 0.96, 0.8, linewidth=1, edgecolor="red", facecolor="red")

        # initial for main view
        line_traj, = ax.plot3D(position[:1, 0], position[:1, 1], position[:1, 2])
        line_quadrotor, = ax.plot3D(position[:1, 0], position[:1, 1], position[:1, 2], marker="o")

        # initial for top view
        line_traj_top, = ax_top.plot(position[:1, 0], position[:1, 1])
        line_quadrotor_top, = ax_top.plot(position[:1, 0], position[:1, 1], marker="o")
        ax_top.add_patch(obs_01_top)

        # initial for front view
        line_traj_front, = ax_front.plot(position[:1, 0], position[:1, 2])
        line_quadrotor_front, = ax_front.plot(position[:1, 0], position[:1, 2], marker="o")
        ax_front.add_patch(obs_01_front)

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

                ax_top.add_patch(obs_01_top)
                ax_front.add_patch(obs_01_front)

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
                        purpose_str = " Human wants to move along +Y axis"
                    elif inputs[-2] == keyboard.KeyCode(char='s'):
                        purpose_str = " Human wants to move -Y axis"
                    elif inputs[-2] == keyboard.KeyCode(char='a'):
                        purpose_str = " Human wants to move -X axis"
                    else:
                        purpose_str = " Human wants to move +X axis"
                        inputs[-2] = keyboard.KeyCode(char='d')

                    print('Human action captured:', inputs[-2], purpose_str)

                # plt.pause(self.time_step * self.time_scale)
                plt.pause(self.time_step * 1)
            return human_interactions
