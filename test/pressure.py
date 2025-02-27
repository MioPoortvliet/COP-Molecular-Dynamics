"""
Test if the pressure is close (enough) to the literature value.
https://www.engineeringtoolbox.com/argon-density-specific-weight-temperature-pressure-d_2089.html
Authors: Mio Poortvliet
"""
from src.utils import N_runs
from src.utils import pressure_fpaths
from src.plotting_utilities import plot_energies
from src.molecular_dynamics import Simulation


def mole_per_liter_to_kg_per_m3(mole_per_liter):
	avogadro = 6.0221409e+23
	sim = Simulation(verbosity=0)
	mass = sim.particle_mass
	return mole_per_liter * 1e3 * mass*avogadro


def check_pressure(known_density, known_temperature, known_pressure, N=20):

	print(known_density, known_temperature, known_pressure)
	paths = N_runs(N=N, density=known_density, temperature=known_temperature, steps=20000, treshold=0.05, verbosity=0)
	plot_energies(paths[0])

	pressure, pressure_error = pressure_fpaths(paths)
	print(pressure, "+/-", pressure_error, known_pressure)
	print(f"That is a relative error of {1-pressure/known_pressure}, compare this to the predicted relative error {pressure_error/pressure}")

	#assert np.abs(to_pressure(pressure) - known_pressure) <= to_pressure(pressure_error)



if __name__ == "__main__":
	known_densities = [35.15, 49.26, 34.57, 1.603]#mole_per_liter_to_kg_per_m3(23.10) #kg/m3
	known_temperatures = [148, 100, 150, 300] #K
	known_pressures = [1e6, 10e5, 10e5, 1e5] #* 101325 #Pa

	for known_density, known_temperature, known_pressure in zip(known_densities, known_temperatures, known_pressures):
		check_pressure(known_density, known_temperature, known_pressure)

