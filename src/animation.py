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
from matplotlib import colors
from src.process_results import *


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
        
        self.pressure = np.zeros(shape=positions.shape[0])
        for i in np.arange(positions.shape[0]):
            self.pressure[i] = pressure_over_rho(positions[i,:].reshape((1,*positions.shape[1:]))) #* properties["unitless_density"]
        
        if type(box_size) in (int, float):
            self.box_size = np.repeat(box_size, dimension)
        else:
            assert len(box_size) == dimension
            self.box_size = box_size
        
        self.fig = plt.figure()
        self.gs = gridspec.GridSpec(4,6)

        if self.dimension == 2:
            self.ax = self.fig.add_subplot(self.gs[:,:-2])
            self.ax.set_aspect('equal')
            
            self.scats = [self.ax.scatter(self.positions[0,j,0],self.positions[0,j,1], s=2) for j,pos in enumerate(self.positions[0,::,0])]
            # self.scat = self.ax.scatter(self.positions[0,::,0], self.positions[0,::,1])
            
            self.time_text = self.ax.text(.02, .02, '', transform=self.ax.transAxes, color="black")
        elif self.dimension == 3:
            self.ax = self.fig.add_subplot(self.gs[:,:-2], projection='3d')
            self.ax.auto_scale_xyz([0, self.box_size[0]],[0, self.box_size[1]],[0, 5*self.box_size[2]])
            
            self.scats = [self.ax.scatter(self.positions[0,j,0],self.positions[0,j,1],self.positions[0,j,2], s=2) for j,pos in enumerate(self.positions[0,::,0])]
            # self.scat = self.ax.scatter(self.positions[0,::,0],self.positions[0,::,1],self.positions[0,::,2])
            
            self.time_text = self.ax.text2D(.02, .02, '', transform=self.ax.transAxes, color="black")
            self.ax.set_zlim3d([0, self.box_size[2]])
            self.ax.set_zlabel('Z')
        
        self.ax.set_xlim([0, self.box_size[0]])
        self.ax.set_ylim([0, self.box_size[1]])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Molecular dynamics')
        
        self.tail_length = 3
        
        self.ax_energy = self.fig.add_subplot(self.gs[0:2,-2:])
        self.ax_energy.set_title('Energy')
        self.ax_energy.set_xlim(0,len(self.ke))
        self.ax_energy.set_ylim((np.amin([self.ke,self.pe])),(np.amax([self.ke, self.pe])))
        self.line_ke, = self.ax_energy.plot([],[], label="ke")
        self.line_pe, = self.ax_energy.plot([],[], label="pe")
        self.line_te, = self.ax_energy.plot([],[], label="te")
        self.ax_energy.legend()
        
        self.ax_pressure   = self.fig.add_subplot(self.gs[2,-2:])
        self.ax_pressure.set_title('Pressure over density')
        self.ax_pressure.set_xlim(0,len(self.pressure))
        self.ax_pressure.set_ylim((np.amin(self.pressure)),(np.amax(self.pressure)))
        self.line_pressure, = self.ax_pressure.plot([],[], label="pressure")
    

    def run(self):
        if self.dimension == 2:
            self.anim = animation.FuncAnimation(self.fig, self.update2d)
        elif self.dimension == 3:
            self.anim = animation.FuncAnimation(self.fig, self.update3d)
        
        self.gs.tight_layout(self.fig)
        plt.show()


    def update2d(self, i):
        self.time_index += self.frameskip
        for j, self.scat in enumerate(self.scats):
            if self.time_index<self.tail_length*self.frameskip:
                self.scat._offsets = (self.positions[:self.time_index+1,j,0:2])
                self.scat.set_sizes(np.linspace(0,4,num=self.time_index, dtype=float))
            else:
                self.scat._offsets = (self.positions[self.time_index+1-self.tail_length*self.frameskip:self.time_index+1,j,0:2])
                self.scat.set_sizes(np.linspace(0,4,num=self.tail_length*self.frameskip, dtype=float))
        # self.scat._offsets = self.positions[self.time_index,::,::]
        self.update_energy(self.time_index)
        self.update_pressure(self.time_index)
        self.time_text.set_text('frame = %.1f' % self.time_index)

    
    def update3d(self, i):
        self.time_index += self.frameskip
        for j, self.scat in enumerate(self.scats):
            if self.time_index<self.tail_length*self.frameskip:
                self.scat._offsets3d = (self.positions[:self.time_index+1,j,0], self.positions[:self.time_index+1,j,1], self.positions[:self.time_index+1,j,2])
                self.scat.set_sizes(np.linspace(0,4,num=self.time_index, dtype=float))
            else:
                self.scat._offsets3d = (self.positions[self.time_index+1-self.tail_length*self.frameskip:self.time_index+1,j,0], self.positions[self.time_index+1-self.tail_length*self.frameskip:self.time_index+1,j,1], self.positions[self.time_index+1-self.tail_length*self.frameskip:self.time_index+1,j,2])
                self.scat.set_sizes(np.linspace(0,4,num=self.tail_length*self.frameskip, dtype=float))
        # self.scat._offsets3d = (self.positions[self.time_index,::,0], self.positions[self.time_index,::,1], self.positions[self.time_index,::,2])
        self.update_energy(self.time_index)
        self.update_pressure(self.time_index)
        self.time_text.set_text('frame = %.1f' % self.time_index)

    def update_energy(self, i):
        self.line_ke.set_data(np.arange(i), self.ke[:i])
        self.line_pe.set_data(np.arange(i), self.pe[:i])
        self.line_te.set_data(np.arange(i), self.te[:i])
    
    def update_pressure(self, i):
        self.line_pressure.set_data(np.arange(i), self.pressure[:i])
""""
    maybe we should consider running the aimation during the simulation.
    Then we do not need to save all data, and we can save time/space for larger simulations
    Now it takes enourmous times to do a simulation for 5e5 steps for example
    Or we could just save the data with frameskip intervals.
"""