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
            self.ax = self.fig.add_subplot(111)
            self.ax.set_aspect('equal')
            self.scat = self.ax.scatter(self.positions[0,::,0], self.positions[0,::,1])
            self.time_text = self.ax.text(.02, .02, '', transform=self.ax.transAxes, color="black")
        elif self.dimension == 3:
            self.ax = self.fig.gca(projection='3d')
            self.ax.auto_scale_xyz([0, self.box_size[0]],[0, self.box_size[1]],[0, 5*self.box_size[2]])
            self.scat = self.ax.scatter(self.positions[0,::,0],self.positions[0,::,1],self.positions[0,::,2])
            self.time_text = self.ax.text2D(.02, .02, '', transform=self.ax.transAxes, color="black")
            self.ax.set_zlim3d([0, self.box_size[2]])
            self.ax.set_zlabel('Z')
        
        self.ax.set_xlim([0, self.box_size[0]])
        self.ax.set_ylim([0, self.box_size[1]])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Molecular dynamics')

    def run(self):
        if self.dimension == 2:
            self.anim = animation.FuncAnimation(self.fig, self.update2d)
        elif self.dimension == 3:
            self.anim = animation.FuncAnimation(self.fig, self.update3d)

        plt.show()


    def update2d(self, i):
        self.time_index += self.frameskip
        self.scat._offsets = self.positions[self.time_index,::,::]
        self.time_text.set_text('frame = %.1f' % self.time_index)


    def update3d(self, i):
        self.time_index += self.frameskip
        self.scat._offsets3d = (self.positions[self.time_index,::,0], self.positions[self.time_index,::,1], self.positions[self.time_index,::,2])
        self.time_text.set_text('frame = %.1f' % self.time_index)
""""
    maybe we should consider running the aimation during the simulation.
    Then we do not need to save all data, and we can save time/space for larger simulations
    Now it takes enourmous times to do a simulation for 5e5 steps for example
    Or we could just save the data with frameskip intervals.
"""