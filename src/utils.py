import numpy as np
import matplotlib as plt
from src.molecular_dynamics import Simulation
from src.IO_utils import load_json, load_and_concat, del_dir
from src.process_results import correlation_function, pressure_over_rho
from src.plotting_utilities import plot_correlation_function
from typing import Tuple

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


def pressure_and_correlation_function(paths, cleanup=False) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
	unitless_pressure = np.zeros(len(paths))
	correlation_function_data_list = []

	for i, path in enumerate(paths):
		# Load properties of last run
		properties = load_json(path)
		# Load positions of last run
		positions = load_and_concat(path, "positions")

		# Use these to calculate the correlation function
		correlation_function_data, distance = correlation_function(positions, max_length=properties["box_size"])
		correlation_function_data_list.append(correlation_function_data)

		# Calculate the unitless pressure
		unitless_pressure[i] = pressure_over_rho(positions) * properties["unitless_density"]

	# Make an array from the list
	correlation_function_data_array = np.array(correlation_function_data_list)

	# Yeet the temporary variables
	del correlation_function_data_list, correlation_function_data

	# Plot and print results
	print(f"Unitless pressure: {np.mean(unitless_pressure)} +/- {np.std(unitless_pressure, ddof=1)}")
	plot_correlation_function(correlation_function_data_array.mean(axis=0), distance, properties)

	# Throw away the simulation data
	if cleanup:
		[del_dir(path) for path in paths]

	# dead degrees of freedom needs to be 1; 1 data point has std of infty!
	# Return (pressure, pressure error) and (correlation function bins, correlation function, correlation function error)
	return (
		(
			np.mean(unitless_pressure),
			np.std(unitless_pressure, ddof=1)
		),
		(
			distance,
			correlation_function_data_array.mean(axis=0),
			correlation_function_data_array.std(ddof=1)
		 )
	)