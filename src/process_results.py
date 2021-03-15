import numpy as np
from src.utils import *
from src.physics import *


def correlation_function(array: np.ndarray, max_length: float) -> (np.ndarray, np.ndarray):
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

	distance = np.linspace(0.00001*max_length, max_length/2 * np.sqrt(dimension), 300)
	delta_r = distance[1]-distance[0]

	average_distance_frequency = np.zeros(distance.size-1)
	for row in array:
		average_distance_frequency += distance_hist(row, distance)

	# Divide by samples to get average
	average_distance_frequency /= time_steps

	pair_correlation_function = 2 * max_length**dimension / (particles * (particles-1)) * average_distance_frequency / (4 * np.pi * distance[1:]**2 * delta_r)

	return pair_correlation_function, distance[:-1]


def pressure_sum(positions: np.ndarray) -> np.ndarray:
	"""
	Partial sum needed to calculate the pressure. Takes positions.
	:param positions: array of positions to use when calculating pressure
	:type positions: np.ndarray
	:return: sum over the derivative of the potential, used to calculate the pressure.
	:rtype:
	"""
	distances = distance_matrix(positions, positions)

	# Use triu(arr, 1) to only select the upper triangle indices where i>j.
	dau_U = deriv_of_U_wrt_r(distances[np.triu_indices_from(distances, 1)])

	return np.sum(distances[np.triu_indices_from(distances, 1)] * dau_U)


def pressure_over_rho(array: np.ndarray) -> float:
	"""
	Calculates the pressure divided by rho
	:param array: positions
	:type array: np.ndarray
	:return: pressure over rho
	:rtype: float
	"""
	particles = array.shape[1]
	time_steps = array.shape[0]

	mean_sum_term = 0
	for row in array:
		mean_sum_term += pressure_sum(row)

	# Divide by samples to get average
	mean_sum_term /= time_steps

	return 1 - 1/(3*particles)*mean_sum_term / 2