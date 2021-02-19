import numpy as np
from src.IO_utils import *
from datetime import datetime

class Simulation():

	def __init__(self, particles, dimension, box_size, end_time, time_step=1e-3, vel_max=None, particle_mass=6.6335e-26, epsilon_over_kb=119.8, sigma=3.405e-10, steps_between_writing=1000, fpath="data/") -> None:
		"""todo particle mass: 6.6335e-26 kg epsilon_over_kb=119.8 K, sigma=3.405e-10 m"""

		self.kb = 1.38e-23
		# Store constants
		if type(box_size) in (int, float):
			self.box_size = np.repeat(box_size/sigma, dimension)
		else:
			assert len(box_size) == dimension
			self.box_size = box_size / sigma

		self.particles = int(particles)
		self.dimension = int(dimension)
		self.end_time = end_time / np.sqrt(particle_mass * sigma**2 / (epsilon_over_kb * self.kb))
		self.time_step = time_step # is already dimensioinless, it is the h
		self.steps_between_writing = steps_between_writing

		self.max_timesteps = np.ceil(self.end_time/self.time_step-1).astype(int)

		# make a comment
		if vel_max is not None:
			self.vel_max = vel_max / np.sqrt(epsilon_over_kb * self.kb / particle_mass)
		else:
			self.vel_max = np.min(self.box_size)/self.time_step/0.001

		self.particle_mass = particle_mass
		self.epsilon_over_kb = epsilon_over_kb
		self.sigma = sigma

		#self.force_treshold = self.particle_mass * np.mean(self.box_size) / self.time_step

		# Initialize arrays
		self.positions = np.zeros(shape=(self.steps_between_writing, self.particles, self.dimension))
		self.positions[0,::,::] = np.array([[1.5, 1.5], [3, 3], [4.3, 3], [5.6, 3], [1, 3]])[:self.particles,::]
		#self.positions[0,::,::] = np.random.rand(self.particles, self.dimension) * self.box_size
		self.velocities = np.zeros(shape=(self.steps_between_writing, self.particles, self.dimension))
		self.potential_energy = np.zeros(shape=(self.steps_between_writing, self.particles))

		self.make_file_structure(fpath)

	def make_file_structure(self, fpath):
		self.fpath = fpath + datetime.today().replace(microsecond=0).isoformat().replace(":", "-") + "/"

		ensure_dir(self.fpath)

		self.fpath_positions = self.fpath+"/positions-"
		self.fpath_velocities = self.fpath+"/velocities-"
		self.fpath_potential_energy = self.fpath+"/potential_energy-"


	def run_sim(self) -> None:
		""""""
		# We don't want to calculate the last time index plus one! So end it one early.

		for cycle in np.arange(np.ceil(self.max_timesteps / self.steps_between_writing), dtype=np.int):
			for time_index in np.arange(min(self.max_timesteps-cycle*self.steps_between_writing, self.steps_between_writing-1), dtype=int):
				self.update(time_index)
				self.positions[time_index+1] = self.apply_periodic_boundaries(self.positions[time_index+1])

			# Append data to file
			self.to_file(self.fpath_positions+str(cycle), self.positions[:time_index])
			self.to_file(self.fpath_velocities+str(cycle), self.velocities[:time_index])
			self.to_file(self.fpath_potential_energy+str(cycle), self.potential_energy[:time_index])

			# Reset arrays
			self.positions[0,::,::] = self.positions[time_index,::,::]
			self.velocities[0,::,::] = self.velocities[time_index,::,::]
			self.potential_energy[0,::] = self.potential_energy[time_index,::]


	def apply_periodic_boundaries(self, positions) -> np.ndarray:
		"""Simply apply modulus. Is it faster to check first? Probably not."""

		return np.mod(positions, self.box_size)


	def update(self, time_index) -> None:
		# Calc new velocities
		distance_vectors = self.distance_vectors(self.positions[time_index])
		self.velocities[time_index+1] = self.velocities[time_index] + self.force(distance_vectors) * self.time_step

		# Calc new positions
		self.positions[time_index+1] = self.positions[time_index] + self.velocities[time_index+1] * self.time_step

		# Calculate potential energy
		distances = self.sum_squared(self.distance_vectors(self.positions[time_index]))
		self.potential_energy[time_index] = np.sum( 4 * ( ( distances**(-12) ) - (distances**(-6)) ), axis=0) / 2



		#print(self.velocities[np.abs(self.velocities) > 1e8])

	def force(self, distance_vectors) -> np.ndarray:
		"""put in distances for every particle, four nearest neighbours with x, y, z components. shape = (particles-1, particles, dimensions)"""
		distances = self.sum_squared(distance_vectors)

		force = np.zeros(shape=distance_vectors.shape[1::])
		#print(distance_vectors[0,0,::]/distances[0,0])
		for dimension in range(self.dimension):
			force[::,dimension] = np.sum(4 * ( 12 / distances ** 13 - 6 / distances ** 7) * distance_vectors[::,::,dimension]/distances, axis=0)

		return force


	def distance_vectors(self, positions_at_time) -> np.ndarray:
		distance_vectors = np.zeros(shape=(positions_at_time.shape[0]-1, positions_at_time.shape[0], self.dimension))
		#print(positions_at_time)
		for i, position in enumerate(positions_at_time):
			distance_vectors[::,i,::] = position - np.delete(positions_at_time, i, axis=0)
			# need to take tiling into account
			for dim in range(self.dimension):
				distance_vectors[::,i,dim][distance_vectors[::,i,dim] > self.box_size[dim]/2] -= self.box_size[dim]
				distance_vectors[::,i,dim][distance_vectors[::,i,dim] < -self.box_size[dim]/2] += self.box_size[dim]

		return distance_vectors


	def sum_squared(self, arr) -> np.ndarray:
		return np.sqrt(np.sum(arr**2, axis=-1))


	def to_file(self, fpath, data):
		print("Writing to "+fpath)
		with open(fpath+".npy", 'wb') as file:
			np.save(file, data)




if __name__ == "__main__":
	import matplotlib.pyplot as plt

	sim = Simulation(2, 2, box_size=1e-9, time_step=1e-3, end_time=1e-11, vel_max=1)
	points=int(1e5)
	x = np.linspace(-1e1, 1e1, points).reshape((1, points, 1))
	print(sim.box_size/sim.time_step)
	plt.plot(x.flatten(), sim.force(x)/sim.particle_mass * sim.time_step)
	plt.ylim(-0.000003, 0.000003)
	plt.show()