# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:56:24 2021

@author: Jonah Post
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
class animation:
    def __init__(self, positions, box_size, dimension):
        self.time_index=0
        self.positions = positions
        self.dimension = dimension
        
        if len(box_size) is not dimension:
            self.box_size = np.repeat(box_size, dimension)
        else:
            self.box_size = box_size
        
        self.fig = plt.figure()
        if self.dimension ==3:
            self.ax = p3.Axes3D(self.fig)
            self.scat = self.ax.scatter(positions[0])
            self.ax.set_xlim3d([-1.*self.box_size[0], self.box_size[0]])
            self.ax.set_ylim3d([-1.*self.box_size[1], self.box_size[1]])
            self.ax.set_zlim3d([-1.*self.box_size[2], self.box_size[2]])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
        elif self.dimension ==2:
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.scat = self.ax.scatter(positions[0])
            self.ax.set_xlim([-1.*self.box_size[0], self.box_size[0]])
            self.ax.set_ylim([-1.*self.box_size[1], self.box_size[1]])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            
    def run(self):
        if self.dimension == 2:
            self.animation = animation.FuncAnimation(self.fig, self.update2d)
        elif self.dimension == 3:
            self.animation = animation.FuncAnimation(self.fig, self.update3d)
    def update2d(self):
        self.time_index =+1
        self.scat._offsets = (self.positions[self.time_index])
    def update3d(self):
        self.time_index =+1
        self.scat._offsets3d = (self.positions[self.time_index])
        