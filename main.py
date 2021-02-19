from src.molecular_dynamics import Simulation
from src.plotting_utilities import *
from src.animation import Animation
from src.IO_utils import *

import numpy as np

def main() -> None:
    dimensions = 2
    particles = 5
    box_size = 3e-9
    
    timestep = 1e-2
    
    sim = Simulation(particles, dimensions, box_size=box_size, time_step=timestep, end_time=3e-11, vel_max=0)
    sim.run_sim()

    positions = load_and_concat(sim.fpath, "positions")
    velocities = load_and_concat(sim.fpath, "velocities")
    potential_energy = load_and_concat(sim.fpath, "potential_energy")

    print(positions.shape)

    plot_positions(positions, velocities, sim.end_time)
    plot_energies(.5*np.sum(velocities**2, axis=-1), potential_energy)
    ani = Animation(positions,.5*np.sum(velocities**2, axis=-1), potential_energy, box_size=sim.box_size, dimension=dimensions, frameskip=10)
    ani.run()



if __name__ == "__main__":
    main()