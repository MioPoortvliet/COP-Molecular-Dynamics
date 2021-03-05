from src.molecular_dynamics import Simulation
from src.plotting_utilities import *
from src.animation import Animation
from src.IO_utils import *
from utils import

import numpy as np

def main() -> None:
    unit_cells = 3

    density = 0.8 # unitless
    temperature = 1 # unitless
    
    timestep = 1e-2 # unitless
    end_time = 2e-11 # seconds
    
    #sim = Simulation(unit_cells_along_axis=unit_cells, unitless_density=density, unitless_temperature=temperature, time_step=timestep, end_time=end_time)

    #sim.run_sim()

    fpath = "data/solid/"#sim.fpath

    positions = load_and_concat(fpath, "positions")
    velocities = load_and_concat(fpath, "velocities")
    potential_energy = load_and_concat(fpath, "potential_energy")



    #plot_positions(positions, velocities)
    #plot_energies(.5*np.sum(velocities**2, axis=-1), potential_energy)
    #ani = Animation(positions,.5*np.sum(velocities**2, axis=-1), potential_energy, box_size=sim.box_size, dimension=3, frameskip=10)
    #ani.run()


if __name__ == "__main__":
    main()