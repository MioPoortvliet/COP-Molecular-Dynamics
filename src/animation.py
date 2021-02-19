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
import matplotlib.gridspec as gridspec


class Animation:
    def __init__(self, positions, kinetic_energy, potential_energy, box_size, dimension, frameskip=1):
        """doccstring"""
        # This call forces the plot in an interactive window
        mpl.use('Qt5Agg')
        self.frameskip = frameskip

        self.time_index=0
        self.positions = positions
        self.dimension = dimension
        
        self.ke = np.sum(kinetic_energy, axis=-1)
        self.pe = np.sum(potential_energy, axis=-1)
        self.te = self.ke + self.pe

        if type(box_size) in (int, float):
            self.box_size = np.repeat(box_size, dimension)
        else:
            assert len(box_size) == dimension
            self.box_size = box_size
        
        self.fig = plt.figure()
        self.gs = gridspec.GridSpec(2,6)

        if self.dimension == 2:
            self.ax = self.fig.add_subplot(self.gs[:,:-2])
            self.ax.set_aspect('equal')
            self.scat = self.ax.scatter(self.positions[0,::,0], self.positions[0,::,1])
            self.time_text = self.ax.text(.02, .02, '', transform=self.ax.transAxes, color="black")
        elif self.dimension == 3:
            self.ax = self.fig.add_subplot(self.gs[:,:-2], projection='3d')
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
        
        
        self.ax_energy = self.fig.add_subplot(self.gs[0,-2:])
        self.ax_energy.set_title('Energy')
        self.ax_energy.set_xlim(0,len(self.ke))
        self.ax_energy.set_ylim((np.amin([self.ke,self.pe])),(np.amax([self.ke, self.pe])))
        self.line_ke, = self.ax_energy.plot([],[], label="ke")
        self.line_pe, = self.ax_energy.plot([],[], label="pe")
        self.line_te, = self.ax_energy.plot([],[], label="te")
        self.ax_energy.legend()
        
        self.ax_temp   = self.fig.add_subplot(self.gs[1,-2:])
        self.ax_temp.set_title('Temperature')

    def run(self):
        if self.dimension == 2:
            self.anim = animation.FuncAnimation(self.fig, self.update2d)
        elif self.dimension == 3:
            self.anim = animation.FuncAnimation(self.fig, self.update3d)
        
        self.gs.tight_layout(self.fig)
        plt.show()


    def update2d(self, i):
        self.time_index += self.frameskip
        self.scat._offsets = self.positions[self.time_index,::,::]
        self.update_energy(self.time_index)
        self.time_text.set_text('frame = %.1f' % self.time_index)


    def update3d(self, i):
        self.time_index += self.frameskip
        self.scat._offsets3d = (self.positions[self.time_index,::,0], self.positions[self.time_index,::,1], self.positions[self.time_index,::,2])
        self.update_energy(self.time_index)
        self.time_text.set_text('frame = %.1f' % self.time_index)

    def update_energy(self, i):
        self.line_ke.set_data(np.arange(i), self.ke[:i])
        self.line_pe.set_data(np.arange(i), self.pe[:i])
        self.line_te.set_data(np.arange(i), self.te[:i])
""""
    maybe we should consider running the aimation during the simulation.
    Then we do not need to save all data, and we can save time/space for larger simulations
    Now it takes enourmous times to do a simulation for 5e5 steps for example
    Or we could just save the data with frameskip intervals.
"""