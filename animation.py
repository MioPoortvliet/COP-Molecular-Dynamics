# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:56:24 2021

@author: Jonah Post
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

fig = plt.figure()
ax = p3.Axes3D(fig)
scat = ax.scatter(positions[0])
ax.set_xlim3d([-1.*self.box_size[0], self.box_size[0]])
ax.set_xlabel('X')

ax.set_ylim3d([-1.*self.box_size[1], self.box_size[1]])
ax.set_ylabel('Y')

ax.set_zlim3d([-1.*self.box_size[2], self.box_size[2]])
ax.set_zlabel('Z')


def update(frame_number):
    time_index +=1
    scat._offsets3d = (positions[time_index])

animation = animation.FuncAnimation(fig, update, interval=interval)