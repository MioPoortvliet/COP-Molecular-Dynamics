from src.utils import N_runs
from src.utils import pressure_fpaths, correlation_function_fpaths
from src.plotting_utilities import plot_energies
from src.molecular_dynamics import Simulation
from src.IO_utils import load_and_concat
import numpy as np


def to_pressure(unitless_pressure):
	sim = Simulation(verbosity=0)
	sigma = sim.sigma
	mass = sim.particle_mass
	epsilon = sim.epsilon_over_kb * sim.kb
	return epsilon / sigma ** 3 * unitless_pressure


def mole_per_liter_to_kg_per_m3(mole_per_liter):
	avogadro = 6.0221409e+23
	sim = Simulation(verbosity=0)
	mass = sim.particle_mass
	return mole_per_liter * 1e3 * mass*avogadro


def check_pressure(known_density, known_temperature, known_pressure):

	paths = N_runs(N=2, density=known_density, temperature=known_temperature, steps=500, treshold=0.01, verbosity=1)
	plot_energies(np.sum(load_and_concat(paths[0], 'velocities')**2, axis=-1) / 2, load_and_concat(paths[0], 'potential_energy'))

	pressure, pressure_error = pressure_fpaths(paths)
	print(known_density, known_temperature, known_pressure)
	print(pressure, "+/-", pressure_error, known_pressure)
	print(f"That is a relative error of {1-pressure/known_pressure}, compare this to the predicted relative error {pressure_error/pressure}")

	#assert np.abs(to_pressure(pressure) - known_pressure) <= to_pressure(pressure_error)



if __name__ == "__main__":
	known_densities = [35.15, 49.26, 34.57, 1.603]#mole_per_liter_to_kg_per_m3(23.10) #kg/m3
	known_temperatures = [148, 100, 150, 300] #K
	known_pressures = [1e6, 10e5, 10e5, 1e5] #* 101325 #Pa

	for known_density, known_temperature, known_pressure in zip(known_densities, known_temperatures, known_pressures):
		check_pressure(known_density, known_temperature, known_pressure)

