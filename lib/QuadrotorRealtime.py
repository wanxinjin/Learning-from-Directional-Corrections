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


def cuboid_data2(o, size=(1, 1, 1)):
    """
    A helper function to plot 3D cube.
    """
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    X += np.array(o)
    return X

def plotCubeAt2(positions, sizes=None, colors=None, **kwargs):
    """
    A helper function to plot 3D cube.
    """
    if not isinstance(colors, (list, np.ndarray)): colors = ["C0"] * len(positions)
    if not isinstance(sizes, (list, np.ndarray)): sizes = [(1,1,1)] * len(positions)
    g = []
    for p, s, c in zip(positions, sizes, colors):
        g.append(cuboid_data2(p, size=s))
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=np.repeat(colors,6), **kwargs)

class QuadrotorRealtime(Quadrotor):
    time_step: float  # the length of time step for optimal controller
    time_scale: float  # scaling the time trajectory
    case_num: int  # the case number for demo

    def __init__(self, time_step=0.1, time_scale=1, case_num=1, project_name='my uav'):
        super().__init__(project_name)
        self.time_step = time_step
        self.time_scale = time_scale
        self.case_num = case_num

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
        ax.set_title('Quadrotor Trajeectory')
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
        if self.case_num == 1:
            obs_01_3d = plotCubeAt2(positions=[(-0.48, -1.5, 0.0)], sizes=[(0.96, 1.0, 0.8)], colors=["red"], edgecolor="k", alpha=0.5)
            obs_01_top = patches.Rectangle((-0.48, -1.5), 0.96, 1.0, linewidth=1, edgecolor="red", facecolor="red", alpha=0.5)
            obs_01_front = patches.Rectangle((-0.48, 0), 0.96, 0.8, linewidth=1, edgecolor="red", facecolor="red", alpha=0.5)
            # plot obstacles
            ax.add_collection3d(obs_01_3d)
            ax_top.add_patch(obs_01_top)
            ax_front.add_patch(obs_01_front)
        elif self.case_num == 2:
            x0 = -0.30
            y0 = -0.80
            z0 = 0.76
            x1 = -0.20
            y1 = 0.48
            z1 = 2.00
            dx = 0.1
            dy = 0.1
            dz = 0.1
            # plot the obstacle on main view
            window_01_3d = plotCubeAt2(positions=[(x0, y0, z0)], sizes=[(dx, y1-y0, dz)], colors=["red"], alpha=0.3)
            window_02_3d = plotCubeAt2(positions=[(x0, y0, z0)], sizes=[(dx, dy, z1-z0)], colors=["red"], alpha=0.3)
            window_03_3d = plotCubeAt2(positions=[(x0, y1-dy, z0)], sizes=[(dx, dy, z1-z0)], colors=["red"], alpha=0.3)
            window_04_3d = plotCubeAt2(positions=[(x0, y0, z1-dz)], sizes=[(dx, y1-y0, dz)], colors=["red"], alpha=0.3)
            ax.add_collection3d(window_01_3d)
            ax.add_collection3d(window_02_3d)
            ax.add_collection3d(window_03_3d)
            ax.add_collection3d(window_04_3d)
            # plot the obstacle on top view (XOY Plane)
            window_01_top = patches.Rectangle((x0, y0), dx, y1-y0, linewidth=1, edgecolor="red", facecolor="red", alpha=0.3)
            window_02_top = patches.Rectangle((x0, y0), dx, dy, linewidth=1, edgecolor="red", facecolor="red", alpha=0.5)
            window_03_top = patches.Rectangle((x0, y1-dy), dx, dy, linewidth=1, edgecolor="red", facecolor="red", alpha=0.5)
            ax_top.add_patch(window_01_top)
            ax_top.add_patch(window_02_top)
            ax_top.add_patch(window_03_top)
            # plot the obstacle on front view (XOZ Plane)
            window_01_front = patches.Rectangle((x0, z0), dx, z1-z0, linewidth=1, edgecolor="red", facecolor="red", alpha=0.3)
            window_02_front = patches.Rectangle((x0, z0), dx, dz, linewidth=1, edgecolor="red", facecolor="red", alpha=0.5)
            window_03_front = patches.Rectangle((x0, z1-dz), dx, dz, linewidth=1, edgecolor="red", facecolor="red", alpha=0.5)
            ax_front.add_patch(window_01_front)
            ax_front.add_patch(window_02_front)
            ax_front.add_patch(window_03_front)

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
                # plt.pause(self.time_step * 1)
                plt.pause(0.01)
            return human_interactions
