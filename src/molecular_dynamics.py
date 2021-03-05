import numpy as np
from src.IO_utils import *
from src.utils import *
from datetime import datetime
import json
import scipy.stats
# from scipy.stats import chi
# from scipy.stats import rv_continuous

class Simulation():

	def __init__(self, end_time=1e-11, density=1e5, unitless_density=None, temperature=293.15, unitless_temperature=None, time_step=1e-3, unit_cells_along_axis=3, particle_mass=6.6335e-26,
				 epsilon_over_kb=119.8, sigma=3.405e-10, steps_between_writing=1000, fpath="data/") -> None:
		"""todo particle mass: 6.6335e-26 kg epsilon_over_kb=119.8 K, sigma=3.405e-10 m"""

		self.kb = 1.38e-23
		# Store constants

		self.unit_cells_along_axis = unit_cells_along_axis

		self.dimension = 3
		self.particles = 4 * unit_cells_along_axis ** self.dimension  # int(particles)
		if unitless_density is None:
			self.unitless_density = density * sigma**self.dimension
		else:
			self.unitless_density = unitless_density
		self.box_size = (self.particles / self.unitless_density) ** (1/self.dimension)

		self.end_time = end_time / np.sqrt(particle_mass * sigma ** 2 / (epsilon_over_kb * self.kb))
		self.time_step = time_step  # is already dimensioinless, it is the h
		self.steps_between_writing = steps_between_writing

		self.max_timesteps = np.ceil(self.end_time / self.time_step - 1).astype(int)

		self.particle_mass = particle_mass
		self.epsilon_over_kb = epsilon_over_kb
		self.sigma = sigma
		if unitless_temperature is None:
			self.temperature = temperature / epsilon_over_kb
		else:
			self.temperature = unitless_temperature

		# self.force_treshold = self.particle_mass * np.mean(self.box_size) / self.time_step

		# Initialize arrays
		self.positions = np.zeros(shape=(self.steps_between_writing, self.particles, self.dimension))
		# self.positions[0,::,::] = np.array([[1.5, 1.5], [3, 3], [4.3, 3], [5.6, 3], [1, 3]])[:self.particles,::]
		self.positions[0, ::, ::] = fcc_lattice(unit_cells=self.unit_cells_along_axis, atom_spacing=self.box_size / (2 * self.unit_cells_along_axis))
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
			to_file(self.fpath_positions + str(cycle), self.positions[:maxtime])
			to_file(self.fpath_velocities + str(cycle), self.velocities[:maxtime])
			to_file(self.fpath_potential_energy + str(cycle), self.potential_energy[:maxtime])

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
			velocity_rescaler = np.sqrt((self.particles-1)*self.dimension*self.temperature/np.sum(self.velocities[steps,::,::]**2))
			print(velocity_rescaler)
			self.velocities[0, ::, ::] = velocity_rescaler*self.velocities[steps,::, ::]
			self.potential_energy[0, ::] = self.potential_energy[steps, ::]



	def run_for_steps(self, steps):
		for time_index in np.arange(1, steps, dtype=int):
			self.update_verlet(time_index)
			self.positions[time_index + 1] = apply_periodic_boundaries(self.positions[time_index + 1], self.box_size)



	def update_euler(self, time_index) -> None:
		# Calc new positions
		self.positions[time_index + 1] = self.positions[time_index] + self.velocities[time_index] * self.time_step

		# Calc new velocities
		distance_vectors = get_distance_vectors(self.positions[time_index], self.box_size, self.dimension)
		self.forces[1] = self.force(distance_vectors)
		self.velocities[time_index + 1] = self.velocities[time_index] + self.forces[1] * self.time_step

		# Calculate potential energy
		distances = sum_squared(get_distance_vectors(self.positions[time_index], self.box_size, self.dimension))
		self.potential_energy[time_index] = np.sum(4 * ((distances ** (-12)) - (distances ** (-6))), axis=0) / 2

	def update_verlet(self, time_index) -> None:
		# Calc new positions
		self.positions[time_index + 1] = self.positions[time_index] + self.velocities[
			time_index] * self.time_step + self.time_step ** 2 / 2 * self.forces[1]

		# Calc new force
		distance_vectors = get_distance_vectors(self.positions[time_index + 1], self.box_size, self.dimension)
		self.forces[0] = self.forces[1]
		self.forces[1] = self.force(distance_vectors)

		# Calc new velocities
		self.velocities[time_index + 1] = self.velocities[time_index] + self.time_step / 2 * np.sum(self.forces, axis=0)

		# Calculate potential energy
		distances = sum_squared(get_distance_vectors(self.positions[time_index], self.box_size, self.dimension))
		self.potential_energy[time_index] = np.sum(4 * ((distances ** (-12)) - (distances ** (-6))), axis=0) / 2

	def force(self, distance_vectors) -> np.ndarray:
		"""put in distances for every particle, four nearest neighbours with x, y, z components. shape = (particles-1, particles, dimensions)"""
		distances = sum_squared(distance_vectors)

		force = np.zeros(shape=distance_vectors.shape[1::])
		# print(distance_vectors[0,0,::]/distances[0,0])
		for dimension in range(self.dimension):
			force[::, dimension] = np.sum(
				4 * (12 / distances ** 13 - 6 / distances ** 7) * distance_vectors[::, ::, dimension] / distances,
				axis=0)

		return force


	def initialize_velocities(self):
		return np.reshape(self.maxwellian_distribution_1D(self.particles * self.dimension),
						  (self.particles, self.dimension))

	def maxwellian_distribution_1D(self, n):
		return np.array(scipy.stats.norm.rvs(scale=np.sqrt(self.temperature), size=n))



if __name__ == "__main__":
	import matplotlib.pyplot as plt

	sim = Simulation(2, 2, box_size=1e-9, time_step=1e-3, end_time=1e-11)
	points = int(1e5)
	x = np.linspace(-1e1, 1e1, points).reshape((1, points, 1))
	print(sim.box_size / sim.time_step)
	plt.plot(x.flatten(), sim.force(x) / sim.particle_mass * sim.time_step)
	plt.ylim(-0.000003, 0.000003)
	plt.show()
