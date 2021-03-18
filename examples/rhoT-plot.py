import numpy as np
from typing import Tuple

from src.plotting_utilities import rhoT_plot
from src.IO_utils import del_dir, load_and_concat, load_json
from src.process_results import find_pressure
from src.utils import N_runs


def calc_pressure_for_rhoT(temprange, densityrange, N=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

	[del_dir(path) for path in allpaths]

	# Alternatively save the data to file and read it out later using plot_rhoT_from_file
	np.save("data/pressure/pressure1.npy", pressure)
	np.save("data/pressure/temprange1.npy", temprange)
	np.save("data/pressure/rhorange1.npy", densityrange)

	return pressure, temprange, densityrange


def plot_rhoT_from_file():
	pressure = np.load("data/pressure/pressure1.npy")
	temprange = np.load("data/pressure/temprange1.npy")
	densityrange = np.load("data/pressure/rhorange1.npy")


	rhoT_plot(pressure.mean(axis=-1), temprange, densityrange)


if __name__ == "__main__":
	resolution = 1 # total resolution is this number squared!
	pressures, temprange, densityrange = calc_pressure_for_rhoT(
		temprange=np.linspace(2.5, 3.5, resolution),
		densityrange=np.linspace(0.5, 1.4, resolution)[::-1],
		N=5)

	rhoT_plot(pressures.mean(axis=-1), temprange, densityrange)

	#plot_rhoT_from_file()
