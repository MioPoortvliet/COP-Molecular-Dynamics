import numpy as np
from src.IO_utils import *
from datetime import datetime
import json
from scipy.stats import chi


class Simulation():

	def __init__(self, particles, dimension, box_size, end_time, time_step=1e-3, particle_mass=6.6335e-26,
				 epsilon_over_kb=119.8, sigma=3.405e-10, steps_between_writing=1000, fpath="data/") -> None:
		"""todo particle mass: 6.6335e-26 kg epsilon_over_kb=119.8 K, sigma=3.405e-10 m"""

		self.kb = 1.38e-23
		# Store constants
		if type(box_size) in (int, float):
			self.box_size = np.repeat(box_size / sigma, dimension)
		else:
			assert len(box_size) == dimension
			self.box_size = box_size / sigma

		self.particles = 4 * 3 ** 3  # int(particles)
		self.dimension = int(dimension)
		self.end_time = end_time / np.sqrt(particle_mass * sigma ** 2 / (epsilon_over_kb * self.kb))
		self.time_step = time_step  # is already dimensioinless, it is the h
		self.steps_between_writing = steps_between_writing

		self.max_timesteps = np.ceil(self.end_time / self.time_step - 1).astype(int)

		self.particle_mass = particle_mass
		self.epsilon_over_kb = epsilon_over_kb
		self.sigma = sigma

		# self.force_treshold = self.particle_mass * np.mean(self.box_size) / self.time_step

		# Initialize arrays
		self.positions = np.zeros(shape=(self.steps_between_writing, self.particles, self.dimension))
		# self.positions[0,::,::] = np.array([[1.5, 1.5], [3, 3], [4.3, 3], [5.6, 3], [1, 3]])[:self.particles,::]
		self.positions[0, ::, ::] = self.fcc_lattice(unit_cells=3, atom_spacing=self.box_size.min() / (2 * 3))
		# x = np.linspace(0, self.box_size[0], int(self.particles**(1/self.dimension)))
		# for i, y in enumerate(x):
		#	for j, z in enumerate(x):
		#		self.positions[0, i*x.size+j, ::] = (y,z)

		self.velocities = np.zeros(shape=(self.steps_between_writing, self.particles, self.dimension))
		self.velocities[0, :, :] = self.initialize_velocities()
		self.potential_energy = np.zeros(shape=(self.steps_between_writing, self.particles))

		self.make_file_structure(fpath)
		self.write_header_file()

	def write_header_file(self):
		header = {}
		header["particles"] = int(self.particles)
		header["dimension"] = int(self.dimension)
		header["end_time_diensionless"] = float(self.end_time)
		header["time_step"] = float(self.time_step)
		header["steps_between_writing"] = int(self.steps_between_writing)
		header["total_steps"] = int(self.max_timesteps)
		header["particle_mass"] = float(self.particle_mass)
		header["epsilon_over_kb"] = float(self.epsilon_over_kb)
		header["sigma"] = float(self.sigma)
		header["kb"] = float(self.kb)

		print(header)

		with open(self.fpath + "00-header.json", "w") as file:
			json.dump(header, file)

	def make_file_structure(self, fpath):
		self.fpath = fpath + datetime.today().replace(microsecond=0).isoformat().replace(":", "-") + "/"

		ensure_dir(self.fpath)

		self.fpath_positions = self.fpath + "positions-"
		self.fpath_velocities = self.fpath + "velocities-"
		self.fpath_potential_energy = self.fpath + "potential_energy-"

	def run_sim(self) -> None:
		""""""
		# We don't want to calculate the last time index plus one! So end it one early.


		self.forces = np.zeros(shape=(2, self.particles, self.dimension))

		self.thermalize()

		self.update_euler(0)
		for cycle in np.arange(np.ceil(self.max_timesteps / self.steps_between_writing), dtype=np.int):
			maxtime = min(self.max_timesteps - cycle * self.steps_between_writing, self.steps_between_writing - 1)
			self.run_for_steps(maxtime)

			# Append data to file
			self.to_file(self.fpath_positions + str(cycle), self.positions[:maxtime])
			self.to_file(self.fpath_velocities + str(cycle), self.velocities[:maxtime])
			self.to_file(self.fpath_potential_energy + str(cycle), self.potential_energy[:maxtime])

			# Reset arrays
			self.positions[0:2, ::, ::] = self.positions[maxtime:maxtime + 2, ::, ::]
			self.velocities[0:2, ::, ::] = self.velocities[maxtime:maxtime + 2, ::, ::]
			self.potential_energy[0:2, ::] = self.potential_energy[maxtime:maxtime + 2, ::]

	def thermalize(self, steps=500, treshold_percentage = 0.1):
		# It should not be necessary to thermalize longer than this as steps_between_writing can be quite large
		assert steps <= self.steps_between_writing
		velocity_rescaler = 0

		while np.abs(velocity_rescaler - 1) > treshold_percentage:
			self.update_euler(0)
			self.run_for_steps(steps)

			# Reset array and keep last values
			self.positions[0, ::, ::] = self.positions[steps,::, ::]
			velocity_rescaler = np.sqrt((self.particles-1)*self.dimension/np.sum(self.velocities**2))
			print(velocity_rescaler)
			self.velocities[0, ::, ::] = velocity_rescaler*self.velocities[steps,::, ::]
			self.potential_energy[0, ::] = self.potential_energy[steps, ::]



	def run_for_steps(self, steps):
		for time_index in np.arange(1, steps, dtype=int):
			self.update_verlet(time_index)
			self.positions[time_index + 1] = self.apply_periodic_boundaries(self.positions[time_index + 1])


	def apply_periodic_boundaries(self, positions) -> np.ndarray:
		"""Simply apply modulus. Is it faster to check first? Probably not."""

		return np.mod(positions, self.box_size)

	def update_euler(self, time_index) -> None:
		# Calc new positions
		self.positions[time_index + 1] = self.positions[time_index] + self.velocities[time_index] * self.time_step

		# Calc new velocities
		distance_vectors = self.distance_vectors(self.positions[time_index])
		self.forces[1] = self.force(distance_vectors)
		self.velocities[time_index + 1] = self.velocities[time_index] + self.forces[1] * self.time_step

		# Calculate potential energy
		distances = self.sum_squared(self.distance_vectors(self.positions[time_index]))
		self.potential_energy[time_index] = np.sum(4 * ((distances ** (-12)) - (distances ** (-6))), axis=0) / 2

	def update_verlet(self, time_index) -> None:
		# Calc new positions
		self.positions[time_index + 1] = self.positions[time_index] + self.velocities[
			time_index] * self.time_step + self.time_step ** 2 / 2 * self.forces[1]

		# Calc new force
		distance_vectors = self.distance_vectors(self.positions[time_index + 1])
		self.forces[0] = self.forces[1]
		self.forces[1] = self.force(distance_vectors)

		# Calc new velocities
		self.velocities[time_index + 1] = self.velocities[time_index] + self.time_step / 2 * np.sum(self.forces, axis=0)

		# Calculate potential energy
		distances = self.sum_squared(self.distance_vectors(self.positions[time_index]))
		self.potential_energy[time_index] = np.sum(4 * ((distances ** (-12)) - (distances ** (-6))), axis=0) / 2

	def force(self, distance_vectors) -> np.ndarray:
		"""put in distances for every particle, four nearest neighbours with x, y, z components. shape = (particles-1, particles, dimensions)"""
		distances = self.sum_squared(distance_vectors)

		force = np.zeros(shape=distance_vectors.shape[1::])
		# print(distance_vectors[0,0,::]/distances[0,0])
		for dimension in range(self.dimension):
			force[::, dimension] = np.sum(
				4 * (12 / distances ** 13 - 6 / distances ** 7) * distance_vectors[::, ::, dimension] / distances,
				axis=0)

		return force

	def distance_vectors(self, positions_at_time) -> np.ndarray:
		distance_vectors = np.zeros(shape=(positions_at_time.shape[0] - 1, positions_at_time.shape[0], self.dimension))
		# print(positions_at_time)
		for i, position in enumerate(positions_at_time):
			distance_vectors[::, i, ::] = position - np.delete(positions_at_time, i, axis=0)

		distance_vectors = (distance_vectors + self.box_size / 2) % self.box_size - self.box_size / 2

		# assert np.all(np.abs(distance_vectors) <= self.box_size/2)

		return distance_vectors

	def sum_squared(self, arr) -> np.ndarray:
		return np.sqrt(np.sum(arr ** 2, axis=-1))

	def initialize_velocities(self):
		return np.reshape(self.maxwellian_distribution_1D(self.particles * self.dimension),
						  (self.particles, self.dimension))

	def maxwellian_distribution_1D(self, n):
		return np.array(chi.rvs(1, size=n)) * (np.random.choice(a=[False, True], size=n) * 2 - 1)

	def to_file(self, fpath, data):
		print("Writing to " + fpath)
		with open(fpath + ".npy", 'wb') as file:
			np.save(file, data)

	def fcc_lattice(self, unit_cells, atom_spacing):
		"Produces a fcc lattice of unit_cells x unit_cells x unit_cells"
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


if __name__ == "__main__":
	import matplotlib.pyplot as plt

	sim = Simulation(2, 2, box_size=1e-9, time_step=1e-3, end_time=1e-11)
	points = int(1e5)
	x = np.linspace(-1e1, 1e1, points).reshape((1, points, 1))
	print(sim.box_size / sim.time_step)
	plt.plot(x.flatten(), sim.force(x) / sim.particle_mass * sim.time_step)
	plt.ylim(-0.000003, 0.000003)
	plt.show()
