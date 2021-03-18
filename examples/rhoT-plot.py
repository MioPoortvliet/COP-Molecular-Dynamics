"""
Plot the pressure in a grid of density-temperature points
Authors: Mio Poortvliet
"""
import numpy as np

from src.plotting_utilities import rhoT_plot
from src.IO_utils import load_and_concat, load_json
from src.process_results import find_pressure
from src.utils import N_runs


def calc_pressure_for_rhoT(temprange, densityrange, N=1) -> np.ndarray:
	"""
	Calculates the data to make a density-temperature plot with pressure on the z-axis.

	:type temprange: np.ndarray
	:type densityrange:  np.ndarray
	:param N: number of simulations to run for each datapoint
	:type N: int
	:return: pressure array
	:rtype: np.ndarray
	"""
	allpaths = []
	pressure = np.zeros(shape=(temprange.size, densityrange.size, N))
	for i, T in enumerate(temprange):
		for j, rho in enumerate(densityrange):
			print(T, rho)
			paths = N_runs(unitless_density=rho, unitless_temperature=T, N=N, verbosity=2)
			allpaths.extend(paths)# Calculate the unitless pressure
			for k, path in enumerate(paths):
				positions = load_and_concat(path, "positions")
				pressure[i, j, k] = find_pressure(positions, properties=load_json(path))[0]

	# Alternatively save the data to file and read it out later using plot_rhoT_from_file
	np.save("data/pressure/pressure1.npy", pressure)
	np.save("data/pressure/temprange1.npy", temprange)
	np.save("data/pressure/rhorange1.npy", densityrange)

	return pressure


def plot_rhoT_from_file(path="data/pressure/"):
	"""
	Open files in path and make density-temperature plot.
	"""
	pressure = np.load(path+"pressure1.npy")
	temprange = np.load(path+"temprange1.npy")
	densityrange = np.load(path+"rhorange1.npy")

	rhoT_plot(pressure.mean(axis=-1), temprange, densityrange, "from_file")


if __name__ == "__main__":
	resolution = 1 # total resolution is this number squared!

	temprange=np.linspace(2.5, 3.5, resolution)
	densityrange=np.linspace(0.5, 1.4, resolution)[::-1]

	pressures = calc_pressure_for_rhoT(temprange, densityrange, N=5)
	rhoT_plot(pressures.mean(axis=-1), temprange, densityrange, "live_calculation")

	#plot_rhoT_from_file()
