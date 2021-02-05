from src.molecular_dynamics import Simulation
from src.plotting_utilities import *
from src.animation import Animation

import numpy as np

def main():
	sim = Simulation(particles=15, dimension=2, box_size=15, time_step=0.0001, end_time=0.01)
	sim.run_sim()
	plot_positions(sim.positions, sim.end_time)
	print(np.mean(sim.velocities[-1,::,::]), np.std(sim.velocities[-1,::,::]))
	ani = Animation(sim.positions, 15, 3)


if __name__ == "__main__":
	main()