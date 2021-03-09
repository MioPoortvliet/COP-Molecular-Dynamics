import numpy as np
from src.IO_utils import *
from src.utils import *
from src.physics import *
from datetime import datetime
import json
import warnings


class Simulation():

	def __init__(
			self,
			end_time=None,
			steps = 1000,
			density=1e5,
			unitless_density=None,
			temperature=293.15,
			unitless_temperature=None,
			time_step=1e-3,
			unit_cells_along_axis=3,
			particle_mass=6.6335e-26,
			epsilon_over_kb=119.8,
			sigma=3.405e-10,
			steps_between_writing=1000,
			fpath="data/",
			verbosity=1
	) -> None:
		"""todo particle mass: 6.6335e-26 kg epsilon_over_kb=119.8 K, sigma=3.405e-10 m"""
		self.verbosity = verbosity
		self.print_(1, "Initializing...")
		# Store constants
		self.kb = 1.38e-23

		self.unit_cells_along_axis = unit_cells_along_axis

		self.dimension = 3
		self.particles = 4 * unit_cells_along_axis ** self.dimension  # int(particles)
		if unitless_density is None:
			self.density = density * sigma**self.dimension
		else:
			self.density = unitless_density

		if self.density > 1.5:
			warnings.warn("Density is so high that the simulation might fail to thermalize.")

		self.box_size = (self.particles / self.density) ** (1/self.dimension)

		self.time_step = time_step  # is already dimensioinless, it is the h
		self.steps_between_writing = steps_between_writing

		if end_time is None:
			self.max_timesteps = steps
			self.end_time = self.max_timesteps * self.time_step
		else:
			self.end_time = end_time / np.sqrt(particle_mass * sigma ** 2 / (epsilon_over_kb * self.kb))
			self.max_timesteps = np.ceil(self.end_time / self.time_step - 1).astype(int)

		self.particle_mass = particle_mass
		self.epsilon_over_kb = epsilon_over_kb
		self.sigma = sigma
		if unitless_temperature is None:
			self.temperature = temperature / epsilon_over_kb
		else:
			self.temperature = unitless_temperature

		# Initialize arrays
		self.positions = np.zeros(shape=(self.steps_between_writing, self.particles, self.dimension))
		# self.positions[0,::,::] = np.array([[1.5, 1.5], [3, 3], [4.3, 3], [5.6, 3], [1, 3]])[:self.particles,::]
		self.positions[0, ::, ::] = fcc_lattice(unit_cells=self.unit_cells_along_axis, atom_spacing=self.box_size / (2 * self.unit_cells_along_axis))

		self.velocities = np.zeros(shape=(self.steps_between_writing, self.particles, self.dimension))
		self.velocities[0, :, :] = initialize_maxwellian_velocities(self.temperature, self.particles, self.dimension)
		self.potential_energy = np.zeros(shape=(self.steps_between_writing, self.particles))

		self.make_file_structure(fpath)
		self.write_header_file()
		self.print_(2, f"Files will be output to {self.fpath}.\n")


	def print_(self, level, *args, **kwargs):
		if self.verbosity >= level:
			print(*args, **kwargs)


	def write_header_file(self):
		"""This function writes all used parameters to a 'header file' in the output dir."""
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
		header["unitless_density"] = float(self.density)
		header["unitless_temperature"] = float(self.temperature)
		header["box_size"] = float(self.box_size)

		with open(self.fpath + "00-header.json", "w") as file:
			json.dump(header, file)


	def make_file_structure(self, fpath):
		self.fpath = fpath + datetime.today().isoformat().replace(":", "-") + "/"

		ensure_dir(self.fpath)

		self.fpath_positions = self.fpath + "positions-"
		self.fpath_velocities = self.fpath + "velocities-"
		self.fpath_potential_energy = self.fpath + "potential_energy-"


	def run_sim(self) -> None:
		"""Function to be called after setting up the simulation, this starts the calculation."""
		self.forces = np.zeros(shape=(2, self.particles, self.dimension))

		# If the temperature is zero there is nothing to thermalize, it is all not moving.
		if self.temperature != 0:
			self.thermalize()

		# We need to calculate one step to be able to use verlet
		self.update_euler(0)
		for cycle in np.arange(np.ceil(self.max_timesteps / self.steps_between_writing), dtype=np.int):
			maxtime = min(self.max_timesteps - cycle * self.steps_between_writing, self.steps_between_writing - 1)
			self.print_(2, f"Simulating {maxtime} steps...")
			self.run_for_steps(maxtime)

			# Append data to file
			self.print_(1, f"Writing {maxtime+1}/{self.max_timesteps} steps to {self.fpath}.")
			to_file(self.fpath_positions + str(cycle), self.positions[:maxtime])
			to_file(self.fpath_velocities + str(cycle), self.velocities[:maxtime])
			to_file(self.fpath_potential_energy + str(cycle), self.potential_energy[:maxtime])

			# Reset arrays
			self.positions[0:2, ::, ::] = self.positions[maxtime:maxtime + 2, ::, ::]
			self.velocities[0:2, ::, ::] = self.velocities[maxtime:maxtime + 2, ::, ::]
			self.potential_energy[0:2, ::] = self.potential_energy[maxtime:maxtime + 2, ::]

		self.print_(1, f"\nDone! Output to {self.fpath}. \n")


	def thermalize(self, steps=100, treshold_percentage=0.15) -> None:
		# It should not be necessary to thermalize longer than this as steps_between_writing can be quite large
		assert steps <= self.steps_between_writing
		self.print_(2, "Starting thermalization.")

		velocity_rescaler = 0
		while np.abs(velocity_rescaler - 1) > treshold_percentage:
			self.update_euler(0)
			self.run_for_steps(steps)

			# Reset array and keep last values
			self.positions[0, ::, ::] = self.positions[steps,::, ::]
			velocity_rescaler = np.sqrt((self.particles-1)*self.dimension*self.temperature/np.sum(self.velocities[steps,::,::]**2))
			self.velocities[0, ::, ::] = velocity_rescaler*self.velocities[steps,::, ::]
			self.potential_energy[0, ::] = self.potential_energy[steps, ::]

			self.print_(1, f"Thermalization stops if {abs(velocity_rescaler - 1):1.3f} <= {treshold_percentage}.")
			assert velocity_rescaler > 1e-10



	def run_for_steps(self, steps) -> None:
		for time_index in np.arange(1, steps, dtype=int):
			self.update_verlet(time_index)


	def update_euler(self, time_index) -> None:
		# Calc new positions
		self.positions[time_index + 1] = apply_periodic_boundaries(
			self.positions[time_index]
			+ self.velocities[time_index] * self.time_step
			, self.box_size
		)

		# Calc new velocities
		distance_vectors = get_distance_vectors(self.positions[time_index], self.box_size, self.dimension)
		self.forces[1] = force(distance_vectors)
		self.velocities[time_index + 1] = self.velocities[time_index] + self.forces[1] * self.time_step

		# Calculate potential energy
		distances = sum_squared(get_distance_vectors(self.positions[time_index], self.box_size, self.dimension))
		self.potential_energy[time_index] = np.sum(4 * ((distances ** (-12)) - (distances ** (-6))), axis=0) / 2


	def update_verlet(self, time_index) -> None:
		# Calc new positions
		self.positions[time_index + 1] = apply_periodic_boundaries(
			self.positions[time_index]
			+ self.velocities[time_index] * self.time_step
			+ self.time_step ** 2 / 2 * self.forces[1]
			, self.box_size
		)

		# Calc new force
		distance_vectors = get_distance_vectors(self.positions[time_index + 1], self.box_size, self.dimension)
		self.forces[0] = self.forces[1]
		self.forces[1] = force(distance_vectors)

		# Calc new velocities
		self.velocities[time_index + 1] = self.velocities[time_index] + self.time_step / 2 * np.sum(self.forces, axis=0)

		# Calculate potential energy
		distances = sum_squared(get_distance_vectors(self.positions[time_index], self.box_size, self.dimension))
		self.potential_energy[time_index] = np.sum(4 * ((distances ** (-12)) - (distances ** (-6))), axis=0) / 2



if __name__ == "__main__":
	pass
