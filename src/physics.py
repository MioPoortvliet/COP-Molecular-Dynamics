import numpy as np
from src.math_utils import *

@njit()
def force(distance_vectors:np.ndarray, distances:np.ndarray) -> np.ndarray:
	"""
	put in distances for every particle, four nearest neighbours with x, y, z components. shape = (particles-1, particles, dimensions)

	:param distance_vectors: distance from one particle to another, with coordinates and direction
	:type distance_vectors: np.ndarray
	:param distances: length of distance from one particle to another
	:type distances: np.ndarray
	:return: Array of forces (including direction)
	:rtype: np.ndarray
	"""
	# 50% speed increase by broadcasting properly!

	# Though it would make sense to compute distances in this function,
	# we avoid computing it twice this way (we use it in potential energy too).
	distances=distances.reshape((*distance_vectors.shape[:-1], 1))

	force = np.zeros(shape=distance_vectors.shape[1::])
	force[::, ::] = np.sum(
		4 * (12 / distances ** 13 - 6 / distances ** 7) * distance_vectors[::, ::, ::] / distances,
		axis=0)

	return force


def deriv_of_U_wrt_r(distances:np.ndarray, sigma, epsilon) -> np.ndarray:
	"""
	Derivative of U with respect to r

	:param distances: length of distance from one particle to another
	:type distances: np.ndarray
	:return: Deriv of U for each given distance
	:rtype: np.ndarray
	"""
	return - 4 * epsilon * (12 * sigma**12 / distances ** 13 - 6 * sigma**6 / distances ** 7)


def initialize_maxwellian_velocities(temperature: float, particles: int, dimension: float) -> np.ndarray:
	"""
	Given a temperature, initializes an array of shape (particles, dimension)
	in a maxwellian velocity distribution to match the temperature

	:param temperature: Temperature to initialize the velocities at
	:type temperature: float
	:param particles: number of particles
	:type particles: int
	:param dimension: dimension of space
	:type dimension: int
	:return: array of velocities
	:rtype: np.ndarray
	"""
	return np.reshape(np.array(scipy.stats.norm.rvs(scale=np.sqrt(temperature), size=particles * dimension)),
						  (particles, dimension))


def fcc_lattice(unit_cells:int, atom_spacing:float) -> np.ndarray:
	"""
	Produces a fcc lattice of unit_cells x unit_cells x unit_cells

	:param unit_cells: number of unit cells in lattice
	:type unit_cells: int
	:param atom_spacing: distance between half a unit cell
	:type atom_spacing: float
	:return: positions of atoms in lattice
	:rtype: np.ndarray
	"""
	unit_cell = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]) * atom_spacing
	lattice = np.zeros(shape=(4 * unit_cells ** 3, 3))

	for x in range(unit_cells):
		for y in range(unit_cells):
			for z in range(unit_cells):
				for i, cell in enumerate(unit_cell):
					# write to the lattice the unit cells uniquely offset by x, y, z
					lattice[x * 4 * unit_cells ** 2 + y * 4 * unit_cells + z * 4 + i,
					::] = cell + atom_spacing * 2 * (np.array([x, y, z]))

	return lattice
