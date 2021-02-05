from src.molecular_dynamics import Simulation
from src.plotting_utilities import *
from src.animation import Animation

import numpy as np

def main():
	dimensions = 3
	particles = 15

	sim = Simulation(particles, dimensions, box_size=30, time_step=0.0001, end_time=0.01)
	sim.run_sim()
	plot_positions(sim.positions, sim.end_time)
	print(np.mean(sim.positions[-1,::,::]), np.std(sim.positions[-1,::,::]))
	ani = Animation(sim.positions, particles, dimensions)
	ani.run()


if __name__ == "__main__":
	main()