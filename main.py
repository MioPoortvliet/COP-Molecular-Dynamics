from src.molecular_dynamics import Simulation
from src.plotting_utilities import *

import numpy as np

def main():
	sim = Simulation(particles=10, dimension=2, box_size=(15, 15), time_step=0.0001, end_time=1)
	sim.run_sim()
	plot_positions(sim.positions, sim.end_time)
	print(np.mean(sim.velocities[-1,::,::]), np.std(sim.velocities[-1,::,::]))


if __name__ == "__main__":
	main()