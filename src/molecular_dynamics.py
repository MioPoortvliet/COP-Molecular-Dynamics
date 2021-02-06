import numpy as np


class Simulation():

	def __init__(self, particles, dimension, box_size, time_step, end_time, vel_max=None, particle_mass=1, epsilon_over_kb=1, sigma=1) -> None:
		"""todo particle mass: 6.6335e-26 kg epsilon_over_kb=119.8, sigma=3.405e-10"""

		# Store constants
		if type(box_size) in (int, float):
			self.box_size = np.repeat(box_size, dimension)
		else:
			assert len(box_size) == dimension
			self.box_size = box_size

		self.particles = int(particles)
		self.dimension = int(dimension)
		self.time_step = time_step
		self.end_time = end_time
		if vel_max is not None:
			self.vel_max = vel_max
		else:
			self.vel_max = np.min(self.box_size)/self.time_step/0.001
		self.particle_mass = particle_mass
		self.epsilon_over_kb = epsilon_over_kb
		self.sigma = sigma

		self.force_treshold = self.particle_mass * np.mean(self.box_size) / self.time_step

		# Initialize arrays
		self.positions = np.random.rand(np.ceil(self.end_time/self.time_step).astype(int), self.particles, self.dimension) * self.box_size
		self.velocities = (np.random.rand(np.ceil(self.end_time/self.time_step).astype(int), self.particles, self.dimension) - 0.5) * 2 * self.vel_max

		self.kb = 1#.38e-23


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

		#print(self.velocities[np.abs(self.velocities) > 1e8])

	def force(self, distance_vectors) -> np.ndarray:
		"""put in distances for every particle, four nearest neighbours with x, y, z components. shape = (particles-1, particles, dimensions)"""
		distances = self.sum_squared(distance_vectors)

		force = np.zeros(shape=distance_vectors.shape[1::])
		for dimension in range(self.dimension):
			force[::,dimension] = np.sum(-4 * self.epsilon_over_kb * self.kb * ( 12 * self.sigma ** 12 / distances ** 13 -  6 * self.sigma ** 6 / distances ** 7) * distance_vectors[::,::,dimension]/distances, axis=0)

		#print(np.max(force, axis=0)/self.particle_mass)
		#force[np.abs(force) > self.force_treshold] = self.force_treshold * np.sign(force[np.abs(force) > self.force_treshold])

		#print(np.mean(force, axis=0)/self.particle_mass, np.std(force, axis=0)/self.particle_mass)

		return force


	def distance_vectors(self, positions_at_time) -> np.ndarray:
		distance_vectors = np.zeros(shape=(positions_at_time.shape[0]-1, positions_at_time.shape[0], self.dimension))
		#print(positions_at_time)
		for i, position in enumerate(positions_at_time):
			distance_vectors[::,i,::] = np.delete(positions_at_time, i, axis=0) - position
			# need to take tiling into account
			for dim in range(self.dimension):
				distance_vectors[::,i,dim][distance_vectors[::,i,dim] > self.box_size[dim]/2] -= self.box_size[dim]
				distance_vectors[::,i,dim][distance_vectors[::,i,dim] < -self.box_size[dim]/2] += self.box_size[dim]

		return distance_vectors


	def sum_squared(self, arr) -> np.ndarray:
		return np.sum(arr**2, axis=-1)



if __name__ == "__main__":
	import matplotlib.pyplot as plt

	sim = Simulation(1e2, 1, box_size=1e-6, time_step=1e-6, end_time=5e-3)
	points=int(1e5)
	x = np.linspace(-1e1, 1e1, points).reshape((1, points, 1))
	print(sim.box_size/sim.time_step)
	plt.plot(x.flatten(), sim.force(x)/sim.particle_mass * sim.time_step)
	plt.ylim(-0.000003, 0.000003)
	plt.show()