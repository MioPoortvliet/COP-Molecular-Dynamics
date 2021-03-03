# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 21:06:56 2021

@author: Jonah Post
"""
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous

import numpy as np



class maxwell_gen(rv_continuous):
	def _pdf(self, velocity,temperature):
		return np.sqrt(1/(2*np.pi*temperature) )* np.exp(-velocity**2 / (2*temperature))
    
    

maxwellian = maxwell_gen(name = "maxwellian")
# NOTE: no normalized units
n=1000
temp=293.15
velx = np.array(maxwellian.rvs(temperature=temp, size=n))
vely = np.array(maxwellian.rvs(temperature=temp, size=n))
velz = np.array(maxwellian.rvs(temperature=temp, size=n))
vel = np.sqrt(velx**2 + vely**2 + velz**2)
print(vel)
plt.hist(vel, bins=100)
plt.xlabel("Velocity")
plt.ylabel("Counts")
plt.show()