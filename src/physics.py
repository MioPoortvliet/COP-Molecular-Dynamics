import numpy as np
from src.utils import *


def force(distance_vectors) -> np.ndarray:
	"""put in distances for every particle, four nearest neighbours with x, y, z components. shape = (particles-1, particles, dimensions)"""
	distances = sum_squared(distance_vectors)

	force = np.zeros(shape=distance_vectors.shape[1::])
	# print(distance_vectors[0,0,::]/distances[0,0])
	for dimension in range(distance_vectors.shape[-1]):
		force[::, dimension] = np.sum(
			4 * (12 / distances ** 13 - 6 / distances ** 7) * distance_vectors[::, ::, dimension] / distances,
			axis=0)

	return force


def deriv_of_U_wrt_r(distances) -> np.ndarray:
	return - 4 * (12 / distances ** 13 - 6 / distances ** 7)


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