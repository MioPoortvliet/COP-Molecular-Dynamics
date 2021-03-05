from src.molecular_dynamics import Simulation
from src.plotting_utilities import *
from src.animation import Animation
from src.IO_utils import *
from src.process_results import *

import numpy as np


def main() -> None:
    unit_cells = 3

    density = 0.8 # unitless
    temperature = 3.0 # unitless
    
    timestep = 1e-2 # unitless
    steps = 1000
    
    sim = Simulation(unit_cells_along_axis=unit_cells, unitless_density=density, unitless_temperature=temperature, time_step=timestep, steps=steps)

    sim.run_sim()

    fpath = sim.fpath
    #fpath = "data/solid/"

    properties = load_json(fpath)
    positions = load_and_concat(fpath, "positions")
    velocities = load_and_concat(fpath, "velocities")
    potential_energy = load_and_concat(fpath, "potential_energy")

    plot_correlation_function(*correlation_function(positions, box_length=properties["box_size"]), properties)
    unitless_pressure = pressure_over_rho(positions)*density

    print(f"Unitless pressure: {unitless_pressure}")


    plot_positions(positions, velocities)
    plot_energies(.5*np.sum(velocities**2, axis=-1), potential_energy)
    #ani = Animation(positions,.5*np.sum(velocities**2, axis=-1), potential_energy, box_size=sim.box_size, dimension=3, frameskip=10)
    #ani.run()


def from_existing_data() -> None:
    fpath = "data/2021-03-05T17-22-20/"

    properties = load_json(fpath)
    positions = load_and_concat(fpath, "positions")
    velocities = load_and_concat(fpath, "velocities")
    potential_energy = load_and_concat(fpath, "potential_energy")

    plot_correlation_function(*correlation_function(positions, box_length=properties["box_size"]), properties)
    unitless_pressure = pressure_over_rho(positions) * properties["unitless_density"]

    print(f"Unitless pressure: {unitless_pressure}")

    plot_positions(positions, velocities)
    plot_energies(.5 * np.sum(velocities ** 2, axis=-1), potential_energy)
    #ani = Animation(positions, .5 * np.sum(velocities ** 2, axis=-1), potential_energy, box_size=properties["box_size"], dimension=3, frameskip=10)
    #ani.run()


def run_statistics(unitless_density, unitless_temperature, N):
    unit_cells = 3
    timestep = 1e-2  # unitless
    steps = 100
    delta_r = 0.001

    unitless_pressure = np.zeros(N)
    correlation_function_data_list = []

    for i in range(N):
        sim = Simulation(
            unit_cells_along_axis=unit_cells,
            unitless_density=unitless_density,
            unitless_temperature=unitless_temperature,
            steps=steps,
            time_step=timestep
        )

        sim.run_sim()

        fpath = sim.fpath

        properties = load_json(fpath)
        positions = load_and_concat(fpath, "positions")

        correlation_function_data, distance = correlation_function(positions, box_length=properties["box_size"], delta_r=delta_r)
        correlation_function_data_list.append(correlation_function_data)

        unitless_pressure[i] = pressure_over_rho(positions) * unitless_density

    # Make an array from the list
    correlation_function_data_array = np.array(correlation_function_data_list)

    # Yeet the temporary variables
    del correlation_function_data_list, correlation_function_data

    # Plot and print results
    print(f"Unitless pressure: {np.mean(unitless_pressure)} +/- {np.std(unitless_pressure)}")
    plot_correlation_function(np.mean(correlation_function_data_array, axis=0), distance, properties)


if __name__ == "__main__":
    main()
    #run_statistics(1, 1, 2)
    #from_existing_data()
