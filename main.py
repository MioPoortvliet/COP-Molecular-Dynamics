from src.molecular_dynamics import Simulation
from src.plotting_utilities import *
from src.animation import Animation

import numpy as np

def main() -> None:
    dimensions = 2
    particles = 2
    box_size = 3e-9
    
    timestep = 1e-2
    
    sim = Simulation(particles, dimensions, box_size=box_size, time_step=timestep, end_time=1e-11, vel_max=0)
    sim.run_sim()
    plot_positions(sim.positions, sim.velocities, sim.end_time)
    plot_energies(.5*np.sum(sim.velocities**2, axis=-1), sim.potential_energy)
    #print(np.mean(sim.velocities[-1,::,::]), np.std(sim.velocities[-1,::,::]), 1e-4/1e-5)
    #ani = Animation(sim.positions, box_size=sim.box_size, dimension=dimensions, frameskip=100)
    #ani.run()


if __name__ == "__main__":
    main()