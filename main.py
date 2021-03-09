from src.molecular_dynamics import Simulation
from src.plotting_utilities import *
from src.animation import Animation
from src.IO_utils import *
from src.process_results import *

import numpy as np


def run_simulation(unitless_density = 0.8, unitless_temperature = 3.0, timestep = 1e-2, steps = 1000, unit_cells=3, verbosity=1) -> None:
    sim = Simulation(
        unit_cells_along_axis=unit_cells,
        unitless_density=unitless_density,
        unitless_temperature=unitless_temperature,
        steps=steps,
        time_step=timestep,
        verbosity=verbosity
    )
    sim.run_sim()

    return sim.fpath


def from_existing_data(fpath) -> None:
    properties = load_json(fpath)
    positions = load_and_concat(fpath, "positions")
    velocities = load_and_concat(fpath, "velocities")
    potential_energy = load_and_concat(fpath, "potential_energy")

    plot_correlation_function(*correlation_function(positions, box_length=properties["box_size"]), properties)
    unitless_pressure = pressure_over_rho(positions) * properties["unitless_density"]

    plot_positions(positions, velocities)
    plot_energies(.5 * np.sum(velocities ** 2, axis=-1), potential_energy)
    #ani = Animation(positions, .5 * np.sum(velocities ** 2, axis=-1), potential_energy, box_size=properties["box_size"], dimension=3, frameskip=10)
    #ani.run()


def run_statistics(unitless_density, unitless_temperature, N):
    steps = 100

    unitless_pressure = np.zeros(N)
    correlation_function_data_list = []

    for i in range(N):
        fpath = run_simulation(
            unitless_density=unitless_density,
            unitless_temperature=unitless_temperature,
            steps=steps,
            verbosity=1
        )

        properties = load_json(fpath)
        positions = load_and_concat(fpath, "positions")

        correlation_function_data, distance = correlation_function(positions, box_length=properties["box_size"])
        correlation_function_data_list.append(correlation_function_data)

        unitless_pressure[i] = pressure_over_rho(positions) * unitless_density

    # Make an array from the list
    correlation_function_data_array = np.array(correlation_function_data_list)

    # Yeet the temporary variables
    del correlation_function_data_list, correlation_function_data

    # Plot and print results
    print(f"Unitless pressure: {np.mean(unitless_pressure)} +/- {np.std(unitless_pressure, ddof=1)}")
    plot_correlation_function(np.mean(correlation_function_data_array, axis=0), distance, properties)

    return np.mean(unitless_pressure)


def calc_pressure(unitless_density, unitless_temperature, steps=100) -> np.float:
    fpath = run_simulation(
        unitless_density=unitless_density,
        unitless_temperature=unitless_temperature,
        steps=steps,
        verbosity=2
    )

    positions = load_and_concat(fpath, "positions")

    return pressure_over_rho(positions) * unitless_density, fpath


def density_temp_plot(temprange, densityrange):
    paths = []
    pressure = np.zeros(shape=(temprange.size, densityrange.size))
    for i, T in enumerate(temprange):
        for j, rho in enumerate(densityrange):
            print(T, rho)
            pressure[i,j], path = calc_pressure(rho, T)
            paths.append(path)

    [del_dirs(path) for path in paths]

    #np.save("data/pressure/pressure.npy", pressure)
    #np.save("data/pressure/temprange.npy", temprange)
    #np.save("data/pressure/rhorange.npy", densityrange)


    #rhoT_plot(pressure, temprange, densityrange)


def load_density_data():
    pressure = np.load("data/pressure/pressure.npy")
    temprange = np.load("data/pressure/temprange.npy")
    densityrange = np.load("data/pressure/rhorange.npy")


    rhoT_plot(pressure, temprange, densityrange)



if __name__ == "__main__":
    #fpath = run_simulation()
    #from_existing_data(fpath)
    resolution = 2 # total resolution is this number squared!
    density_temp_plot(temprange=np.linspace(0, 3, resolution), densityrange=np.linspace(0.01, 1.4, resolution)[::-1])
    load_density_data()
