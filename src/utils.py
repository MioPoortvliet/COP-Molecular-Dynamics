"""
Helper functions to run the simulation and calculate results in batches.
Authors: Mio Poortvliet, Jonah Post
"""

import numpy as np
import matplotlib as plt
from src.molecular_dynamics import Simulation
from src.IO_utils import load_json, load_and_concat, del_dir
from src.process_results import correlation_function, find_pressure
from src.plotting_utilities import plot_correlation_function
from typing import Tuple
from src.math_utils import sum_squared

def N_runs(
		N,
		**kwargs
	) -> str:
	fpaths = []
	for i in range(N):
		# Run simulation and store filepath of output data
		sim = Simulation(
			**kwargs
		)
		sim.run_sim()

		fpaths.append(sim.fpath)

	return fpaths


def pressure_fpaths(paths):
	pressure = np.zeros((len(paths), 2))

	if type(paths) == str:
		paths = [paths]

	for i, path in enumerate(paths):
		# Load properties of last run
		properties = load_json(path)
		# Load positions of last run
		positions = load_and_concat(path, "positions")

		# Calculate the unitless pressure
		pressure[i,::] = find_pressure(positions, properties)

	print(f"Pressure: {np.mean(pressure[::,0])} +/- {np.std(pressure[::,0], ddof=1)}")

	return np.mean(pressure[::,0]), np.std(pressure[::,0], ddof=1)


def correlation_function_fpaths(paths) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	correlation_function_data_list = []

	if type(paths) == str:
		paths = [paths]

	for i, path in enumerate(paths):
		# Load properties of last run
		properties = load_json(path)
		# Load positions of last run
		positions = load_and_concat(path, "positions")

		# Use these to calculate the correlation function
		correlation_function_data, distance = correlation_function(positions, max_length=properties["box_size"])
		correlation_function_data_list.append(correlation_function_data)

	# Make an array from the list
	correlation_function_data_array = np.array(correlation_function_data_list)

	# Yeet the temporary variables
	del correlation_function_data_list, correlation_function_data

	# Plot and print results
	plot_correlation_function(correlation_function_data_array.mean(axis=0), distance, properties)

	# dead degrees of freedom needs to be 1; 1 data point has std of infty!
	# Return (pressure, pressure error) and (correlation function bins, correlation function, correlation function error)
	return distance, correlation_function_data_array.mean(axis=0), correlation_function_data_array.std(ddof=1)