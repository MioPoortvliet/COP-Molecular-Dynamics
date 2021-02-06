from src.molecular_dynamics import Simulation
from src.plotting_utilities import *
from src.animation import Animation

import numpy as np

def main() -> None:
	dimensions = 2
	particles = 20
	box_size = 5

	timestep = 1e-9

	sim = Simulation(particles, dimensions, box_size=box_size, time_step=timestep, end_time=timestep*5e3, vel_max=1)
	sim.run_sim()
	plot_positions(sim.positions, sim.velocities, sim.end_time)
	print(np.mean(sim.velocities[-1,::,::]), np.std(sim.velocities[-1,::,::]), 1e-4/1e-5)
	ani = Animation(sim.positions, box_size=box_size, dimension=dimensions, frameskip=50)
	ani.run()


if __name__ == "__main__":
	main()