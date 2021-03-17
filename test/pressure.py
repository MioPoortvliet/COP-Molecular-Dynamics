from src.utils import N_runs
from src.utils import pressure_and_correlation_function
from src.plotting_utilities import plot_energies
from src.molecular_dynamics import Simulation
from src.IO_utils import load_and_concat
import numpy as np


def to_pressure(unitless_pressure):
	sim = Simulation()
	sigma = sim.sigma
	mass = sim.particle_mass
	epsilon = sim.epsilon_over_kb * sim.kb
	return epsilon / sigma ** 3 * unitless_pressure


def mole_per_liter_to_kg_per_m3(mole_per_liter):
	avogadro = 6.0221409e+23
	sim = Simulation()
	mass = sim.particle_mass
	print(mass * avogadro)
	return mole_per_liter * 1e3 * mass*avogadro

if __name__ == "__main__":
	known_density = 34.57#mole_per_liter_to_kg_per_m3(23.10) #kg/m3
	known_temperature = 150 #K
	known_pressure = 10e5 #* 101325 #Pa

	paths = N_runs(N=1, density=known_density, temperature=known_temperature, steps=5000, treshold=0.01)
	plot_energies(np.sum(load_and_concat(paths[0], 'velocities')**2, axis=-1) / 2, load_and_concat(paths[0], 'potential_energy'))

	pressure, _ = pressure_and_correlation_function(paths, cleanup=True)
	print(to_pressure(np.array(pressure)), known_pressure)

	#assert (to_pressure(pressure[0] - pressure[1]) <= known_pressure) and (to_pressure(pressure[0] + pressure[1]) >= known_pressure)

