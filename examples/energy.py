from src.plotting_utilities import plot_positions, plot_energies
from src.animation import Animation
from src.IO_utils import load_and_concat, load_json
from src.utils import N_runs, pressure_fpaths, correlation_function_fpaths

import numpy as np


def energy_analysis(fpath) -> None:
    properties = load_json(fpath)
    positions = load_and_concat(fpath, "positions")
    velocities = load_and_concat(fpath, "velocities")
    potential_energy = load_and_concat(fpath, "potential_energy")

    plot_positions(positions, velocities)
    plot_energies(.5 * np.sum(velocities ** 2, axis=-1), potential_energy)
    ani = Animation(positions, .5 * np.sum(velocities ** 2, axis=-1), potential_energy, box_size=properties["box_size"], dimension=3, frameskip=10)
    ani.run()


if __name__ == "__main__":
    fpath = N_runs(unitless_temperature=1, unitless_density=1, N=1)

    energy_analysis(fpath[0])