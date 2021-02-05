from src.molecular_dynamics import Simulation
from src.plotting_utilities import *
from src.animation import Animation

import numpy as np

def main() -> None:
	dimensions = 3
	particles = 100
	box_size = 1e-4

	sim = Simulation(particles, dimensions, box_size=box_size, time_step=1e-5, end_time=5e-3)
	sim.run_sim()
	plot_positions(sim.positions, sim.velocities, sim.end_time)
	print(np.mean(sim.velocities[-1,::,::]), np.std(sim.velocities[-1,::,::]), 1e-4/1e-5)
	ani = Animation(sim.positions, box_size=box_size, dimension=dimensions, frameskip=10)
	ani.run()


if __name__ == "__main__":
	main()