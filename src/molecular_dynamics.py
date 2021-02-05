import numpy as np


class Simulation():

	def __init__(self, particles, dimension, box_size, time_step, end_time, vel_max=1, particle_mass = 6.6335e-26, epsilon_over_kb=119.8, sigma=3.405) -> None:
		"""todo"""

		# Store constants
		if type(box_size) is int:
			self.box_size = np.repeat(box_size, dimension)
		else:
			assert len(box_size) == dimension
			self.box_size = box_size

		self.particles = particles
		self.dimension = dimension
		self.time_step = time_step
		self.end_time = end_time
		self.vel_max = vel_max
		self.particle_mass = particle_mass
		self.epsilon_over_kb = epsilon_over_kb
		self.sigma = sigma

		# Initialize arrays
		self.positions = np.random.rand(np.ceil(self.end_time/self.time_step).astype(int), self.particles, self.dimension) * self.box_size
		self.velocities = np.random.rand(np.ceil(self.end_time/self.time_step).astype(int), self.particles, self.dimension) * self.vel_max



	def run_sim(self) -> None:
		""""""
		# We don't want to calculate the last time index plus one! So end it one early.
		for time_index in np.arange(np.ceil(self.end_time/self.time_step-1).astype(int), dtype=int):
			self.update(time_index)
			self.positions[time_index+1] = self.apply_periodic_boundaries(self.positions[time_index+1])


	def apply_periodic_boundaries(self, positions) -> np.ndarray:
		"""Simply apply modulus. Is it faster to check first? Probably not."""

		return np.mod(positions, self.box_size)


	def update(self, time_index) -> None:
		distance_vectors = self.distance_vectors(self.positions[time_index])

		self.positions[time_index+1] = self.positions[time_index] + self.velocities[time_index] * self.time_step
		self.velocities[time_index+1] = self.velocities[time_index] + 1/self.particle_mass * self.force(distance_vectors) * self.time_step


	def force(self, distance_vectors) -> np.ndarray:
		"""put in distances for every particle, four nearest neighbours with x, y, z components. shape = (particles-1, particles, dimensions)"""
		distances = self.sum_squared(distance_vectors)

		return np.sum(4 * self.epsilon_over_kb * ( 12 * self.sigma ** 12 / distances ** 13 -  6 * self.sigma ** 6 / distances ** 7) * distance_vectors/distances, axis=0)


	def distance_vectors(self, positions_at_time) -> np.ndarray:
		distance_vectors = np.zeros(shape=(self.particles-1, self.particles, self.dimension))

		for i, position in enumerate(positions_at_time):
			distance_vectors[::,i,::] = np.delete(positions_at_time, i, axis=0) - position

		return distance_vectors


	def sum_squared(self, arr) -> np.ndarray:
		return np.sum(arr**2)