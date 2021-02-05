from src.molecular_dynamics import Simulation
from src.plotting_utilities import *
from src.animation import Animation

import numpy as np

def main():
	dimensions = 3
	particles = 50
	box_size = 1

	sim = Simulation(particles, dimensions, box_size=box_size, time_step=0.000001, end_time=0.0001)
	sim.run_sim()
	plot_positions(sim.positions, sim.end_time)
	print(np.mean(sim.positions[-1,::,::]), np.std(sim.positions[-1,::,::]))
	ani = Animation(sim.positions, box_size=box_size, dimension=dimensions)
	ani.run()


if __name__ == "__main__":
	main()