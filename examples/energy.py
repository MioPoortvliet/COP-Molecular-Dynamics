"""
Plot the energy over time, as well as a projection of the positions and the correlation function.
Authors: Mio Poortvliet
"""
from src.plotting_utilities import plot_positions, plot_energies
from src.animation import Animation
from src.utils import N_runs, pressure_fpaths, correlation_function_fpaths


def energy_analysis(fpath) -> None:
    """
    Plot position, energy and correlation function of results in fpath. Then run an animation.
    :param fpath:
    :type fpath:
    :return:
    :rtype:
    """
    plot_positions(fpath)
    plot_energies(fpath)
    correlation_function_fpaths(fpath)
    #ani = Animation(fpath, frameskip=10)
    #ani.run()


if __name__ == "__main__":
    temps = [60, 120, 360]
    densities = [1680*1.2, 1680*0.8, 1680*0.3]

    for temp, density in zip(temps, densities):
        paths = N_runs(
            temperature=temp,
            density=density,
            N=1,
            treshold=0.01,
            steps=1000,
            steps_between_writing=20000,
            steps_for_thermalizing=1000,
            time_step = 1e-2
        )

        for path in paths:
            print(temp, density)
            energy_analysis(path)
            pressure_fpaths(path)
