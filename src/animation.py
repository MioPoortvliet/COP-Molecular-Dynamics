# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:56:24 2021

@author: Jonah Post
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib as mpl


class Animation:
    def __init__(self, positions, box_size, dimension, frameskip=1):
        """doccstring"""
        # This call forces the plot in an interactive window
        mpl.use('Qt5Agg')
        self.frameskip = frameskip

        self.time_index=0
        self.positions = positions
        self.dimension = dimension

        if type(box_size) in (int, float):
            self.box_size = np.repeat(box_size, dimension)
        else:
            assert len(box_size) == dimension
            self.box_size = box_size
        
        self.fig = plt.figure()

        if self.dimension == 2:
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.ax.set_aspect('equal')
            self.scat = self.ax.scatter(self.positions[0,::,0],self.positions[0,::,1])
            self.ax.set_xlim([0, self.box_size[0]])
            self.ax.set_ylim([0, self.box_size[1]])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')

        elif self.dimension == 3:
            self.ax = p3.Axes3D(self.fig)

            self.scat = self.ax.scatter(self.positions[0,::,0],self.positions[0,::,1],self.positions[0,::,2])
            self.ax.set_xlim3d([0, self.box_size[0]])
            self.ax.set_ylim3d([0, self.box_size[1]])
            self.ax.set_zlim3d([0, self.box_size[2]])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')


    def run(self):
        if self.dimension == 2:
            self.anim = animation.FuncAnimation(self.fig, self.update2d)
        elif self.dimension == 3:
            self.anim = animation.FuncAnimation(self.fig, self.update3d)

        plt.show()


    def update2d(self, i):
        self.time_index += self.frameskip
        self.scat._offsets = (self.positions[self.time_index,::,0], self.positions[self.time_index,::,1])


    def update3d(self, i):
        self.time_index += self.frameskip
        self.scat._offsets3d = (self.positions[self.time_index,::,0], self.positions[self.time_index,::,1], self.positions[self.time_index,::,2])
        