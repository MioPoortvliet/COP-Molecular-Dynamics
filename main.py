from src.molecular_dynamics import Simulation
from src.plotting_utilities import *
from src.animation import Animation
from src.IO_utils import *

import numpy as np

def main() -> None:
    dimensions = 3
    particles = 4*3**dimensions
    box_size = 20e-9
    
    timestep = 1e-2
    
    sim = Simulation(particles, dimensions, box_size=box_size, time_step=timestep, end_time=2e-11)

    sim.run_sim()

    fpath = sim.fpath

    positions = load_and_concat(fpath, "positions")
    velocities = load_and_concat(fpath, "velocities")
    potential_energy = load_and_concat(fpath, "potential_energy")

    plot_positions(positions, velocities)
    plot_energies(.5*np.sum(velocities**2, axis=-1), potential_energy)
    ani = Animation(positions,.5*np.sum(velocities**2, axis=-1), potential_energy, box_size=box_size/3.405e-10, dimension=dimensions, frameskip=10)
    ani.run()



if __name__ == "__main__":
    main()