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
        """
        Initializes the Animation Class. Stores and calculates all relevant physical quantities and sets up the figures for the animation.
        
        param positions: positions of every particle at all timesteps.
        type positions: np.ndarray
        param kinetic energy: kinetic energy per particle at every timestep
        type positions: np.ndarray
        param potential energy: potential energy per particle at every timestep
        type potential energy: np.ndarray
        param box_size: if int or float the length of a cubic the box. If array_like contains the length of the box in each direction.
        type box_size: int, float or np.ndarray
        param dimension: dimensionallity of the system, can be either 2 or 3.
        type dimension: int
        param frameskip: the number of frames to be skipped while running the animation.
        type frameskip: int
        """
        
        # This call forces the plot in an interactive window
        mpl.use('Qt5Agg')
        self.frameskip = frameskip

        self.frame_index=0
        self.positions = positions
        self.dimension = dimension
        
        self.kin_energy = np.sum(kinetic_energy, axis=-1)
        self.pot_energy = np.sum(potential_energy, axis=-1)
        self.tot_energy = self.kin_energy + self.pot_energy
        
        self.pressure = np.zeros(shape=positions.shape[0])
        for i in np.arange(positions.shape[0]):
            self.pressure[i] = pressure_over_rho(positions[i,:].reshape((1,*positions.shape[1:])))[0] #* properties["unitless_density"]
        
        if type(box_size) in (int, float):
            self.box_size = np.repeat(box_size, dimension)
        else:
            assert len(box_size) == dimension
            self.box_size = box_size
        
        self.fig = plt.figure()
        self.gs = gridspec.GridSpec(3,6)

        if self.dimension == 2:
            self.ax = self.fig.add_subplot(self.gs[:,:-2])
            self.ax.set_aspect('equal')
            self.scats = [self.ax.scatter(self.positions[0,j,0],self.positions[0,j,1], s=2) for j,pos in enumerate(self.positions[0,::,0])]
            self.time_text = self.ax.text(.02, .02, '', transform=self.ax.transAxes, color="black")
        elif self.dimension == 3:
            self.ax = self.fig.add_subplot(self.gs[:,:-2], projection='3d')
            self.ax.auto_scale_xyz([0, self.box_size[0]],[0, self.box_size[1]],[0, 5*self.box_size[2]])
            self.scats = [self.ax.scatter(self.positions[0,j,0],self.positions[0,j,1],self.positions[0,j,2], s=2) for j,pos in enumerate(self.positions[0,::,0])]
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
        self.ax_energy.set_xlim(0,len(self.kin_energy))
        self.ax_energy.set_ylim((np.amin([self.kin_energy,self.pot_energy])),(np.amax([self.kin_energy, self.pot_energy])))
        self.line_kin_energy, = self.ax_energy.plot([],[], label="kinetic energy")
        self.line_pot_energy, = self.ax_energy.plot([],[], label="potential energy")
        self.line_tot_energy, = self.ax_energy.plot([],[], label="total energy")
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
            self.anim = animation.FuncAnimation(self.fig, self.update3d,frames = np.arange(0, np.shape(self.tot_energy)[0],step=self.frameskip), repeat=False)
        self.gs.tight_layout(self.fig)
        plt.show()


    def update2d(self, i):
        self.frame_index=i
        for j, self.scat in enumerate(self.scats):
            if self.frame_index<self.tail_length*self.frameskip:
                self.scat._offsets = (self.positions[:self.frame_index+1,j,0:2])
                self.scat.set_sizes(np.linspace(0,4,num=self.frame_index, dtype=float))
            else:
                self.scat._offsets = (self.positions[self.frame_index+1-self.tail_length*self.frameskip:self.frame_index+1,j,0:2])
                self.scat.set_sizes(np.linspace(0,4,num=self.tail_length*self.frameskip, dtype=float))
        self.update_energy(self.frame_index)
        self.update_pressure(self.frame_index)
        self.time_text.set_text('frame = %.1f' % self.frame_index)

    
    def update3d(self, i):
        self.frame_index=i
        for j, self.scat in enumerate(self.scats):
            if self.frame_index<self.tail_length*self.frameskip:
                self.scat._offsets3d = (self.positions[:self.frame_index+1,j,0], self.positions[:self.frame_index+1,j,1], self.positions[:self.frame_index+1,j,2])
                self.scat.set_sizes(np.linspace(0,4,num=self.frame_index, dtype=float))
            else:
                self.scat._offsets3d = (self.positions[self.frame_index+1-self.tail_length*self.frameskip:self.frame_index+1,j,0], self.positions[self.frame_index+1-self.tail_length*self.frameskip:self.frame_index+1,j,1], self.positions[self.frame_index+1-self.tail_length*self.frameskip:self.frame_index+1,j,2])
                self.scat.set_sizes(np.linspace(0,4,num=self.tail_length*self.frameskip, dtype=float))
        self.update_energy(self.frame_index)
        self.update_pressure(self.frame_index)
        self.time_text.set_text('frame = %.1f' % self.frame_index)

    def update_energy(self, i):
        self.line_kin_energy.set_data(np.arange(i), self.kin_energy[:i])
        self.line_pot_energy.set_data(np.arange(i), self.pot_energy[:i])
        self.line_tot_energy.set_data(np.arange(i), self.tot_energy[:i])
    
    def update_pressure(self, i):
        self.line_pressure.set_data(np.arange(i), self.pressure[:i])
