"""
Process raw data from the simulation to useful results.
Authors: Mio Poortvliet, Jonah Post
"""

import numpy as np
from src.physics import *
from typing import Tuple
from scipy.spatial import distance_matrix


def correlation_function(array: np.ndarray, max_length: float) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Returns the correlation function of the array, up to max_length

	:param array: Data you want the correlation function over
	:type array: np.ndarray
	:param max_length: largest length of the system
	:type max_length: foat
	:return: correlation between each element in array
	:rtype: np.ndarray
	"""
	time_steps, particles, dimension = array.shape

	# Set up bins
	distance_bins = np.linspace(0.00001*max_length, max_length/2 * np.sqrt(dimension), 300)
	delta_r = distance_bins[1]-distance_bins[0]

	# Calculate distance between pairs histogram, averaged over time
	average_distance_frequency = np.zeros(distance_bins.size-1)
	for row in array:
		average_distance_frequency += distance_hist(row, distance_bins, max_length)

	# Divide by samples to get average
	average_distance_frequency /= time_steps

	# Then the correlation function
	pair_correlation_function = 2 * max_length**dimension / (particles * (particles-1)) * average_distance_frequency / (4 * np.pi * distance_bins[1:]**2 * delta_r)

	return pair_correlation_function, distance_bins[:-1]


def pressure_sum(positions: np.ndarray) -> np.ndarray:
	"""
	Partial sum needed to calculate the pressure. Takes positions.
	:param positions: array of positions to use when calculating pressure
	:type positions: np.ndarray
	:return: sum over the derivative of the potential, used to calculate the pressure.
	:rtype:
	"""
	# This does not take minimum image convention into account! You did this before passing positions to this sum.
	distances = distance_matrix(positions, positions)

	# Use triu(arr, 1) to only select the upper triangle indices where i>j.
	dau_U = deriv_of_U_wrt_r(distances[np.triu_indices_from(distances, 1)])

	return np.sum(distances[np.triu_indices_from(distances, 1)] * dau_U)


def find_pressure(array: np.ndarray, properties) -> Tuple[float, float]:
	"""
	Calculates the pressure divided by rho
	:param array: positions
	:type array: np.ndarray
	:return: pressure over rho
	:rtype: float
	"""
	# Create arrays
	particles = array.shape[1]
	time_steps = array.shape[0]

	# Calculate average of nasty sum term
	sum_term = np.zeros(time_steps)
	for tstep, row in enumerate(array):
		# Minimal image convention
		row = (row + properties["box_size"] / 2) % properties["box_size"] - properties["box_size"] / 2
		sum_term[tstep] = pressure_sum(row/properties["sigma"])

	# Pressure is calculated
	pressure_array = properties["unitless_temperature"] - 1/(3*particles)*sum_term / 2
	pressure_array *= properties["unitless_density"]

	# Pressure is converted to units of Pa
	pressure_array = to_pressure(pressure_array, sigma=properties["sigma"], epsilon=properties["epsilon"], mass=properties["particle_mass"])

	return pressure_array.mean(), pressure_array.std(ddof=1)


def to_pressure(unitless_pressure, sigma, epsilon, mass) -> np.ndarray:
	"""
	Gives units to pressure
	:type unitless_pressure:
	:type sigma: float
	:type epsilon: float
	:rtype: np.ndarray
	"""
	return epsilon / sigma ** 3 * unitless_pressure