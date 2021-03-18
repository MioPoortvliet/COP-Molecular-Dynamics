import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from numba import njit


def sum_squared(arr) -> np.ndarray:
	"""
	Returns the square root of the sum of the elements squared along the last axis.
	:param arr: array where the sum should be computed over
	:type arr: np.ndarray
	:return: square root of sum over squared elements along last axis
	:rtype: np.ndarray
	"""
	return np.sqrt(np.sum(arr ** 2, axis=-1))



def get_distance_vectors(positions_at_time, box_size, dimension) -> np.ndarray:
	"""Produces distances r_{ij} without r_{ii}, using minimal image convention. Takes direction into account!"""
	distance_vectors = np.zeros(shape=(positions_at_time.shape[0] - 1, positions_at_time.shape[0], dimension))
	# print(positions_at_time)
	for i, position in enumerate(positions_at_time):
		# Old solution:
		#distance_vectors[::, i, ::] = position - np.delete(positions_at_time, i, axis=0)

		# This i sactually faster! Unfortunately no boolean masking using numba
		mask = np.ones(shape=positions_at_time.shape, dtype=np.bool_)
		mask[i, ::] = False
		distance_vectors[::, i, ::] = positions_at_time[i, ::] - positions_at_time[mask].reshape((distance_vectors.shape[0], distance_vectors.shape[-1]))

	# Apply periodic boundary conditions
	distance_vectors = (distance_vectors + box_size / 2) % box_size - box_size / 2

	# When in doubt, check this!
	# Manually checked using debugger and print()
	# assert np.all(np.abs(distance_vectors) <= self.box_size/2)

	return distance_vectors

# Slower than using numpy
@njit()
def get_distance_vectors_jitted(positions_at_time, box_size, dimension) -> np.ndarray:
	"""Produces distances r_{ij} without r_{ii}, using minimal image convention. Takes direction into account!"""
	distance_vectors = np.zeros(shape=(positions_at_time.shape[0]-1, positions_at_time.shape[0], dimension))
	# print(positions_at_time)
	for i in range(positions_at_time.shape[0]):
		addition = False
		for j in range(positions_at_time.shape[0]):
			if i != j:
				distance_vectors[j-addition, i, ::] = positions_at_time[i,::] - positions_at_time[j,::]
			else:
				addition = True

	distance_vectors = (distance_vectors + box_size / 2) % box_size - box_size / 2

	# assert np.all(np.abs(distance_vectors) <= self.box_size/2)
	#assert np.all(get_distance_vectors_old(positions_at_time, box_size, dimension) == distance_vectors)
	return distance_vectors


def apply_periodic_boundaries(positions, period) -> np.ndarray:
	"""Simply apply modulus. Is it faster to check first? Probably not."""
	return np.mod(positions, period)


def distance_hist(array, distance_bins, max_length) -> np.array:
	"""

	:param array: points wherebetween distance should be computed
	:type array: np.ndarray
	:param bins: bins to make hist
	:type bins: np.array
	:return: array of histogram data
	:rtype: np.array
	"""
	distances = sum_squared(get_distance_vectors(array, max_length, 3))
	#print(distances, sum_squared(get_distance_vectors(array, distance_bins[-1]/np.sqrt(3), 3)) )
	hist, _ = np.histogram(distances.flatten(), bins=distance_bins)
	hist = hist / 2 # because we count every pair double!
	return hist



