import numpy as np
from src.utils import *
from src.physics import *


def correlation_function(array, box_length, delta_r = 0.001) -> np.array:
	time_steps, particles, dimension = array.shape

	distance = np.arange(delta_r, box_length/2 * np.sqrt(dimension), delta_r)

	average_distance_frequency = np.zeros(distance.size-1)
	for row in array:
		average_distance_frequency += distance_hist(row, distance)

	# Divide by samples to get average
	average_distance_frequency /= time_steps

	pair_correlation_function = 2 * box_length**dimension / (particles * (particles-1)) * average_distance_frequency / (4 * np.pi * distance[:-1]**2 * delta_r)

	return pair_correlation_function, distance[:-1]


def pressure_sum(positions):
	distances = distance_matrix(positions, positions)

	# Use triu(arr, 1) to only select the upper triangle indices where i>j.
	dau_U = deriv_of_U_wrt_r(distances[np.triu_indices_from(distances, 1)])

	return np.sum(distances[np.triu_indices_from(distances, 1)] * dau_U)


def pressure_over_rho(array):
	particles = array.shape[1]
	time_steps = array.shape[0]

	mean_sum_term = 0
	for row in array:
		mean_sum_term += pressure_sum(row)

	# Divide by samples to get average
	mean_sum_term /= time_steps

	return 1 - 1/(3*particles)*mean_sum_term / 2