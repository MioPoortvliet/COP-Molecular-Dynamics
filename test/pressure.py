from src.utils import N_runs
from src.utils import pressure_fpaths, correlation_function_fpaths
from src.plotting_utilities import plot_energies
from src.molecular_dynamics import Simulation
from src.IO_utils import load_and_concat
import numpy as np


def to_pressure(unitless_pressure):
	sim = Simulation(verbosity=3)
	sigma = sim.sigma
	mass = sim.particle_mass
	epsilon = sim.epsilon_over_kb * sim.kb
	return epsilon / sigma ** 3 * unitless_pressure


def mole_per_liter_to_kg_per_m3(mole_per_liter):
	avogadro = 6.0221409e+23
	sim = Simulation(verbosity=3)
	mass = sim.particle_mass
	print(mass * avogadro)
	return mole_per_liter * 1e3 * mass*avogadro


def check_pressure(known_density, known_temperature, known_pressure):

	paths = N_runs(N=1, density=known_density, temperature=known_temperature, steps=5000, treshold=0.01, verbosity=2)
	plot_energies(np.sum(load_and_concat(paths[0], 'velocities')**2, axis=-1) / 2, load_and_concat(paths[0], 'potential_energy'))

	pressure, pressure_error = pressure_fpaths(paths)
	print(to_pressure(pressure), "+/-", to_pressure(pressure_error), known_pressure)
	print(f"That is a relative error of {1-to_pressure(pressure)/known_pressure}")

	#assert (to_pressure(pressure[0] - pressure[1]) <= known_pressure) and (to_pressure(pressure[0] + pressure[1]) >= known_pressure)



if __name__ == "__main__":
	known_densities = [49.26, 34.57, 1.603]#mole_per_liter_to_kg_per_m3(23.10) #kg/m3
	known_temperatures = [100, 150, 300] #K
	known_pressures = [10e5, 10e5, 1e5] #* 101325 #Pa

	for known_density, known_temperature, known_pressure in zip(known_densities, known_temperatures, known_pressures):
		check_pressure(known_density, known_temperature, known_pressure)

