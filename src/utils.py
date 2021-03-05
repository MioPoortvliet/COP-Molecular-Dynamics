import numpy as np
import scipy.stats

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


def apply_periodic_boundaries(positions, period) -> np.ndarray:
	"""Simply apply modulus. Is it faster to check first? Probably not."""
	return np.mod(positions, period)

def initialize_maxwellian_velocities(temperature, particles, dimension):
    return np.reshape(np.array(scipy.stats.norm.rvs(scale=np.sqrt(temperature), size=particles * dimension)),
						  (particles, dimension))


def fcc_lattice(unit_cells, atom_spacing):
	"""Produces a fcc lattice of unit_cells x unit_cells x unit_cells"""
	unit_cell = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]) * atom_spacing
	lattice = np.zeros(shape=(4 * unit_cells ** 3, 3))
	for x in range(unit_cells):
		for y in range(unit_cells):
			for z in range(unit_cells):
				for i, cell in enumerate(unit_cell):
					lattice[x * 4 * unit_cells ** 2 + y * 4 * unit_cells + z * 4 + i,
					::] = cell + atom_spacing * 2 * (np.array([x, y, z]))
	# print(lattice)

	return lattice
