import numpy as np


def sum_squared(arr) -> np.ndarray:
	return np.sqrt(np.sum(arr ** 2, axis=-1))


def get_distance_vectors(positions_at_time, box_size, dimension) -> np.ndarray:
	"""Produces distances r_{ij} without r_{ii}, using minimal image convention"""
	distance_vectors = np.zeros(shape=(positions_at_time.shape[0] - 1, positions_at_time.shape[0], dimension))
	# print(positions_at_time)
	for i, position in enumerate(positions_at_time):
		distance_vectors[::, i, ::] = position - np.delete(positions_at_time, i, axis=0)

	distance_vectors = (distance_vectors + box_size / 2) % box_size - box_size / 2

	# assert np.all(np.abs(distance_vectors) <= self.box_size/2)

	return distance_vectors


def apply_periodic_boundaries(positions, box_size) -> np.ndarray:
	"""Simply apply modulus. Is it faster to check first? Probably not."""
	return np.mod(positions, box_size)