# COP-Molecular-Dynamics

Authors: Jonah Post, Mio Poortvliet
##
COP-Molecular-Dynamics is a molecular dynamics simulation implemented in Python 3.8 as an assignment for the course 
Computational Physics, part of the Leiden University physics master's.

The simulation is carried out mostly by making smart use of Numpy arrays and uses some Numba here and there too.
####Dependencies
The simulation relies on three standard scientific packages that are not part of the Python base.
- Numpy
- Matplotlib
- Numba

All examples are verified to work on the latest Windows build (as of 2021-03-17, Python 3.8). **It is important that the root path is this folder when executing a file.**

##How to use
Below a quick-start guide is written talking you through how to use the simulation and simultaneously how it works.
###Initialization
First initialize the simulation by calling ```Simulation()``` from the file ```src/molecular_dynamics.py```. The simulation initializes a system of particles in fcc unit cells. The amount of unit cells along an axis is determined by ```unit_cells_along_axis``` (default is 3 in 3 dimensions making for 108 particles). This fcc lattice is created at a density of ```unitless_density``` (alternatively pass along ```density``` in SI units instead, the simulation will do a conversion). Then starting velocities are drawn from a maxwellian distribution at ```unitless_temperature``` (or alternatively ```temperature``` in SI units). This will run for ```steps``` time steps with a unitless timestep length of ```time_step``` (default is 1e-3).  

###Thermalizing the system
Upon calling ```run_sim()``` the system is first thermalized. This is done by running the simulation for 100 steps, computing the mean kinetic energy and comparing this to the expected energy gotten from the equipartition theorem. The velocities are rescaled appropriately if the relative difference is within ```treshold``` (pass this to ```Simulation()```, default is 0.1).

###Simulating
After thermalization the system is calculated forward in time by ```steps``` time steps, writing data every ```steps_between_writing``` steps. This is written to a folder in ```/data/``` named as the ISO time code of the time when ```Simulation()``` was called.

###Finishing up
All data is written to the folder in sequential files. The path of the folder is returned. Using the ```IO_utils.py``` function ```load_and_concat(fpath, 'positions')``` the positions array is read from the files. The same can be done for ```'velocities'``` and ```'potential_energy'```.

These arrays can be fed into the functions in ```process_results.py``` and ```plotting_utilities.py```.

All functions have type annotations and a comprehensive docstring.

###Examples
In the folder ```/examples/```, scripts making use of the ```/src/``` files are located. 
- ```energy.py``` plots the energy, velocity distribution and a 1D projection of the paths of the simulation. It also plots the correlation function. 
- ```rhoT-plot.py``` makes a plot with temperature on the horizontal axis and density on the vertical axis. The z-variable is the pressure. All these quantities are in terms of the unitless quantities used by the simulation.
- ```statistics_pressure_and_correlationfunction.py``` runs the simulation N times and produces a pressure along with an error estimation, along with a plot of the correlation function.
- ```animation.py``` runs the simulation and plots the positions in an interactive animated plot.

###Todo
- More tests
- Create a GUI to run the simulation from and process results in.