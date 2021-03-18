from src.plotting_utilities import plot_positions, plot_energies
from src.animation import Animation
from src.IO_utils import load_and_concat, load_json
from src.utils import N_runs, pressure_fpaths, correlation_function_fpaths

import numpy as np


def energy_analysis(fpath) -> None:
    plot_positions(fpath)
    plot_energies(fpath)
    ani = Animation(fpath, frameskip=10)
    ani.run()


if __name__ == "__main__":
    fpath = N_runs(unitless_temperature=1, unitless_density=1, N=1)

    energy_analysis(fpath[0])