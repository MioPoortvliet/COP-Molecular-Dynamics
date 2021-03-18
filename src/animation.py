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
from src.IO_utils import load_and_concat, load_json


class Animation:
    def __init__(self, fpath, frameskip=1):
        """
        Initializes the Animation Class. Stores and calculates all relevant physical quantities and sets up the figures for the animation.
        
        param positions: positions of every particle at all timesteps.
        type positions: np.ndarray
        param kinetic energy: kinetic energy per particle at every timestep
        type positions: np.ndarray
        param potential energy: potential energy per particle at every timestep
        type potential energy: np.ndarray
        param properties: 
        type properties: 
        param frameskip: the number of frames to be skipped while running the animation.
        type frameskip: int
        """

        properties = load_json(fpath)
        positions = load_and_concat(fpath, "positions")
        velocities = load_and_concat(fpath, "velocities")
        potential_energy = load_and_concat(fpath, "potential_energy")

        # This call forces the plot in an interactive window
        mpl.use('Qt5Agg')
        self.frameskip = frameskip

        self.frame_index=0
        self.positions = positions
        self.dimension = properties["dimension"]
        
        self.kin_energy = np.sum(.5 *properties["particle_mass"] *np.sum(velocities ** 2, axis=-1), axis=-1)
        self.pot_energy = np.sum(potential_energy, axis=-1)
        self.tot_energy = self.kin_energy + self.pot_energy
        
        self.pressure = np.zeros(shape=positions.shape[0])
        for i in np.arange(positions.shape[0]):
            self.pressure[i] = find_pressure(positions[i,:].reshape((1,*positions.shape[1:])), properties=properties)[0] #* properties["unitless_density"]
        
        if type(properties["box_size"]) in (int, float):
            self.box_size = np.repeat(properties["box_size"], self.dimension)
        else:
            assert len(properties["box_size"]) == self.dimension
            self.box_size = properties["box_size"]
        
        self.fig = plt.figure()
        self.gs = gridspec.GridSpec(3,6)

        if self.dimension == 2:
            self.ax = self.fig.add_subplot(self.gs[:,:-2])
            self.ax.set_aspect('equal')
            self.scats = [self.ax.scatter(self.positions[0,j,0],self.positions[0,j,1], s=2) for j,pos in enumerate(self.positions[0,::,0])]
            self.time_text = self.ax.text(.02, .02, '', transform=self.ax.transAxes, color="black")
            self.config_text = self.ax.text(.02, .95, '', transform=self.ax.transAxes, color="black")
        elif self.dimension == 3:
            self.ax = self.fig.add_subplot(self.gs[:,:-2], projection='3d')
            self.ax.auto_scale_xyz([0, self.box_size[0]],[0, self.box_size[1]],[0, 5*self.box_size[2]])
            self.scats = [self.ax.scatter(self.positions[0,j,0],self.positions[0,j,1],self.positions[0,j,2], s=2) for j,pos in enumerate(self.positions[0,::,0])]
            self.time_text = self.ax.text2D(.02, .02, '', transform=self.ax.transAxes, color="black")
            self.config_text = self.ax.text2D(.02, .95, '', transform=self.ax.transAxes, color="black")
            self.ax.set_zlim3d([0, self.box_size[2]])
            self.ax.set_zlabel('Z')
        
        self.time_per_frame = properties["time_step"]* np.sqrt((properties["sigma"]**2) * properties["particle_mass"] /(properties["epsilon_over_kb"] * properties["kb"]))
        self.config_text.set_text("Temperature = {temp:.1f} K\n".format(temp = properties["unitless_temperature"]*properties["epsilon_over_kb"])
                                  +
                                  r"Density = {density:.1f} kg m$^{{-3}}$".format(density = properties["unitless_density"]*properties["particle_mass"]/(properties["sigma"]**3)))
        
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
        self.ax_pressure.set_title('Pressure')
        self.ax_pressure.set_xlim(0,len(self.pressure))
        self.ax_pressure.set_ylim((np.amin(self.pressure)),(np.amax(self.pressure)))
        self.line_pressure, = self.ax_pressure.plot([],[], label="pressure")
    

    def run(self):
        if self.dimension == 2:
            self.anim = animation.FuncAnimation(self.fig, self.update2d, frames = np.arange(0, np.shape(self.tot_energy)[0],step=self.frameskip), repeat=False)
        elif self.dimension == 3:
            self.anim = animation.FuncAnimation(self.fig, self.update3d, frames = np.arange(0, np.shape(self.tot_energy)[0],step=self.frameskip), repeat=False)
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
        self.time_text.set_text('frame = {frame:.1f}\n time = {time:.1f}'.format(frame = self.frame_index, time = self.frame_index*self.time_per_frame))

    
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
        self.time_text.set_text('frame = {frame:.1f}\n time = {time:.3f}e-12 s'.format(frame = self.frame_index, time = self.frame_index*self.time_per_frame*(1e12)))

    def update_energy(self, i):
        self.line_kin_energy.set_data(np.arange(i), self.kin_energy[:i])
        self.line_pot_energy.set_data(np.arange(i), self.pot_energy[:i])
        self.line_tot_energy.set_data(np.arange(i), self.tot_energy[:i])
    
    def update_pressure(self, i):
        self.line_pressure.set_data(np.arange(i), self.pressure[:i])
