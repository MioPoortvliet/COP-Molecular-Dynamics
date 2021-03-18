"""
Contains the main simulation class.
Authors: Mio Poortvliet, Jonah Post
"""

import numpy as np
from src.IO_utils import *
from src.physics import *
from datetime import datetime
import json
import warnings


class Simulation():

	def __init__(
			self,
			end_time=None,
			steps = 1000,
			density=1.602,
			unitless_density=None,
			temperature=293.15,
			unitless_temperature=None,
			time_step=1e-2,
			unit_cells_along_axis=3,
			particle_mass=6.6335e-26,
			epsilon_over_kb=119.8,
			sigma=3.405e-10,
			steps_between_writing=1000,
			fpath="data/",
			verbosity=1,
			treshold=0.1,
			steps_for_thermalizing=999
	) -> None:
		"""
		Initialize the simulation class. Stores all variables and sets the simulation up to be able to run.

		:param end_time: Time in seconds to end the simulation at.
		:type end_time: float
		:param steps: Maximum steps to calulate, use instead of end_time.
		:type steps: int
		:param density: Density in kg/m^3.
		:type density: float
		:param unitless_density: Unitless density, use instead of density.
		:type unitless_density: float
		:param temperature: Temperature in Kelvin.
		:type temperature: float
		:param unitless_temperature: Unitless temperature, use instead of temperature.
		:type unitless_temperature: float
		:param time_step: unitless timestep size, should be around 1e-2 or smaller.
		:type time_step: float
		:param unit_cells_along_axis: number of unit cells to create along an axis.
		:type unit_cells_along_axis: int
		:param particle_mass: mass of the particles.
		:type particle_mass: float
		:param epsilon_over_kb: interaction strength over Boltzman's constant.
		:type epsilon_over_kb: float
		:param sigma: characteristic length.
		:type sigma: float
		:param steps_between_writing: after this many steps the simulation writes the data to file.
		:type steps_between_writing: int
		:param fpath: root directory of where to save the simulation results.
		:type fpath: str
		:param verbosity: a number from 0 to 3 controlling how explicit the program prints what it's doing.
		:type verbosity: int
		:param treshold: When to stop thermalization as a relative energy difference.
		:type treshold: float
		:param steps_for_thermalizing: Number of steps to thermalize for, should be less than steps_between_writing.
		:type steps_for_thermalizing: int
		"""
		self.verbosity = verbosity
		self.print_(1, "Initializing...")

		# Store constants
		self.kb = 1.38e-23
		self.particle_mass = particle_mass
		self.epsilon_over_kb = epsilon_over_kb
		self.sigma = sigma

		# Particle properties
		self.dimension = 3
		self.unit_cells_along_axis = unit_cells_along_axis
		self.particles = 4 * unit_cells_along_axis ** self.dimension  # int(particles)

		# Set density
		if unitless_density is None:
			self.unitless_density = density * sigma ** self.dimension / particle_mass
			self.density = density
		else:
			self.unitless_density = unitless_density
			self.density = unitless_density * particle_mass / sigma ** self.dimension
		if self.unitless_density > 1.5:
			warnings.warn("Density is so high that the simulation might fail to thermalize.")

		# Set temperature
		if unitless_temperature is None:
			self.unitless_temperature = temperature / epsilon_over_kb
			self.temperature = temperature
		else:
			self.unitless_temperature = unitless_temperature
			self.temperature = unitless_temperature * epsilon_over_kb

		# This allows us to calculate the box size
		self.box_size = (self.particles / self.unitless_density) ** (1/self.dimension)

		# Variables to tweak the simulation
		self.time_step = time_step  # is already dimensioinless, it is the h
		self.steps_between_writing = steps_between_writing
		self.treshold = treshold
		self.steps_for_thermalizing = steps_for_thermalizing

		# Set end time
		if end_time is None:
			self.max_timesteps = steps
			self.end_time = self.max_timesteps * self.time_step
		else:
			self.end_time = end_time / np.sqrt(particle_mass * sigma ** 2 / (epsilon_over_kb * self.kb))
			self.max_timesteps = np.ceil(self.end_time / self.time_step).astype(int)

		# Initialize arrays
		self.positions = np.zeros(shape=(self.steps_between_writing, self.particles, self.dimension))
		self.positions[0, ::, ::] = fcc_lattice(unit_cells=self.unit_cells_along_axis, atom_spacing=self.box_size / (2 * self.unit_cells_along_axis))

		self.velocities = np.zeros(shape=(self.steps_between_writing, self.particles, self.dimension))
		self.velocities[0, :, :] = initialize_maxwellian_velocities(self.unitless_temperature, self.particles, self.dimension)

		self.potential_energy_array = np.zeros(shape=(self.steps_between_writing, self.particles))

		# Set up IO
		self.make_file_structure(fpath)
		self.write_header_file()

		self.print_(2, f"Files will be output to {self.fpath}.\n")


	def print_(self, level:int, *args, **kwargs):
		"""
		Just a function to control the verbosity of print statements. Sometimes you want to print less, sometimes more.
		:param level: Verbosity level, compared to what is set in self.verbosity
		:type level: int
		:param args: goes to print()
		:param kwargs: goes to print()
		:return: None
		:rtype: None
		"""
		if self.verbosity >= level:
			print(*args, **kwargs)


	def write_header_file(self) -> None:
		"""
		This function writes all used parameters to a header file '00-header.json' in the output dir.
		:return: None
		:rtype: None
		"""
		header = {}
		header["particles"] = int(self.particles)
		header["dimension"] = int(self.dimension)
		header["end_time_diensionless"] = float(self.end_time)
		header["end_time"] = self.end_time * np.sqrt(self.particle_mass * self.sigma ** 2 / (self.epsilon_over_kb * self.kb))
		header["time_step"] = float(self.time_step)
		header["steps_between_writing"] = int(self.steps_between_writing)
		header["total_steps"] = int(self.max_timesteps)
		header["particle_mass"] = float(self.particle_mass)
		header["epsilon_over_kb"] = float(self.epsilon_over_kb)
		header["epsilon"] = float(self.epsilon_over_kb * self.kb)
		header["sigma"] = float(self.sigma)
		header["kb"] = float(self.kb)
		header["unitless_density"] = float(self.unitless_density)
		header["density"] = float(self.density)
		header["unitless_temperature"] = float(self.unitless_temperature)
		header["temperature"] = float(self.temperature)
		header["box_size"] = float(self.box_size*self.sigma)
		header["path"] = str(self.fpath)

		with open(self.fpath + "00-header.json", "w") as file:
			json.dump(header, file)


	def make_file_structure(self, fpath:str) -> None:
		"""
		Creates directories and sets up filenames to be used when writing data later.

		:param fpath: filepath root where the structure should be created
		:type fpath: str
		:return: None
		:rtype: None
		"""
		# Path is determined by the current datetime, removing illegal characters
		self.fpath = fpath + datetime.today().isoformat().replace(":", "-") + "/"

		ensure_dir(self.fpath)

		self.fpath_positions = self.fpath + "positions-"
		self.fpath_velocities = self.fpath + "velocities-"
		self.fpath_potential_energy = self.fpath + "potential_energy-"


	def run_sim(self) -> None:
		"""
		Function to be called after setting up the simulation, this starts the calculation.
		:return: None
		:rtype: None
		"""

		self.forces = np.zeros(shape=(2, self.particles, self.dimension))

		# If the temperature is zero there is nothing to thermalize, it is all not moving.
		if self.unitless_temperature != 0:
			self.thermalize()
		else:
			self.update_euler(0)
			distance_vectors = get_distance_vectors(self.positions[0], self.box_size, self.dimension)
			distances = sum_squared(distance_vectors)
			self.potential_energy_array[0,::] = self.potential_energy(distances)
			del distances, distance_vectors

		# We need to calculate one step to be able to use verlet
		# Because we thermalized, we already calculated everything for the first two steps
		for cycle in np.arange(np.ceil(self.max_timesteps / self.steps_between_writing), dtype=np.int):
			maxtime = min(self.max_timesteps - cycle * self.steps_between_writing, self.steps_between_writing) - 1
			self.print_(2, f"Simulating {maxtime+1} steps...")
			self.run_for_steps(maxtime)

			# Append data to file
			self.print_(1, f"Writing {(maxtime+1)*(cycle+1)}/{self.max_timesteps} steps to {self.fpath}")
			# Write to file in SI units
			to_file(self.fpath_positions + str(cycle), self.to_units_position(self.positions[:maxtime]))
			to_file(self.fpath_velocities + str(cycle), self.to_units_velocity(self.velocities[:maxtime]) )
			to_file(self.fpath_potential_energy + str(cycle), self.potential_energy_array[:maxtime])

			# Reset arrays
			self.positions[0:2, ::, ::] = self.positions[maxtime:maxtime + 2, ::, ::]
			self.velocities[0:2, ::, ::] = self.velocities[maxtime:maxtime + 2, ::, ::]
			self.potential_energy_array[0:2, ::] = self.potential_energy_array[maxtime:maxtime+2, ::]

		self.print_(1, f"\nDone! Output to {self.fpath}. \n")


	def potential_energy(self, distances) -> np.ndarray:
		"""
		Returns potential energy given distances between particles

		:param distances: unitless distance array
		:type distances: np.ndarray
		:return: Potential energy
		:rtype: np.array
		"""
		distances = self.to_units_position(distances)

		#Divide by 2 because we are double counting pairs
		return np.sum(4 * (self.sigma**12 * (distances ** (-12)) - (self.sigma**6 * distances ** (-6))), axis=0) * self.epsilon_over_kb * self.kb / 2


	def to_units_position(self, unitless) -> np.ndarray:
		"""
		Gives m units to array

		:param unitless:  unitless distance array
		:type unitless:  np.ndarray
		:return: distance array
		:rtype: np.array
		"""
		return unitless * self.sigma


	def to_units_velocity(self, unitless) -> np.ndarray:
		"""
		Gives m/s units to array

		:param unitless: unitless velocity data
		:type unitless:
		:return: velocity data
		:rtype: np.array
		"""
		return unitless * np.sqrt(self.epsilon_over_kb * self.kb / self.particle_mass)


	def thermalize(self) -> None:
		"""
		Makes sure the kinetic energy of the system compared to the expected thermal energy is within the set treshold.

		:param steps: Number of steps to calculate between re-normalizing the velocities
		:type steps: int
		:param treshold: The maximum allowable relative error between the kinetic energy and expected thermal energy
		:type treshold: float
		:return: None
		:rtype: None
		"""
		# It should not be necessary to thermalize longer than this as steps_between_writing can be quite large
		# If the array is not large enough to contain these steps throw an error!
		if self.steps_for_thermalizing >= self.steps_between_writing:
			raise ValueError("Can not store the steps needed to thermalize, either increase the steps between writing or decrease the steps in the thermalization.")

		self.print_(2, "Starting thermalization.")

		velocity_rescaler = 0
		while np.abs(velocity_rescaler - 1) > self.treshold:
			# Need some way to propegate with one timestep of initial conditions
			self.update_euler(0)

			self.run_for_steps(self.steps_for_thermalizing)

			velocity_rescaler = np.sqrt((self.particles-1)*self.dimension*self.unitless_temperature/np.sum(self.velocities[self.steps_for_thermalizing,::,::]**2))

			# Reset array and keep last values
			self.positions[0:2, ::, ::] = self.positions[self.steps_for_thermalizing-1:self.steps_for_thermalizing+1,::, ::]
			self.velocities[0:2, ::, ::] = velocity_rescaler*self.velocities[self.steps_for_thermalizing-1:self.steps_for_thermalizing+1,::, ::]
			self.potential_energy_array[0:2, ::] = self.potential_energy_array[self.steps_for_thermalizing-1:self.steps_for_thermalizing+1, ::]

			self.print_(1, f"Thermalization stops if {abs(velocity_rescaler - 1):1.3f} <= {self.treshold}.")

			# If the density is too high, numerical errors do not allow this process to function.
			assert velocity_rescaler > 1e-10


	def run_for_steps(self, steps: int) -> None:
		"""
		Run simulation for steps timesteps.

		:param steps: Number of steps to run
		:type steps: int
		:return: None
		:rtype: None
		"""
		for time_index in np.arange(1, steps, dtype=int):
			self.update_verlet(time_index)


	def update_euler(self, time_index: int) -> None:
		"""
		Calculate forwards in time using the Euler alogrithm

		:param time_index: timestep to calculate from
		:type time_index: int
		:return: None
		:rtype: None
		"""
		# Calc new positions
		self.positions[time_index + 1] = apply_periodic_boundaries(
			self.positions[time_index]
			+ self.velocities[time_index] * self.time_step
			, self.box_size
		)

		# Calc new velocities
		distance_vectors = get_distance_vectors(self.positions[time_index+1], self.box_size, self.dimension)
		distances = sum_squared(distance_vectors)

		self.forces[1] = force(distance_vectors, distances)
		self.velocities[time_index + 1] = self.velocities[time_index] + self.forces[1] * self.time_step

		# Calculate potential energy
		self.potential_energy_array[time_index+1] = self.potential_energy(distances)


	def update_verlet(self, time_index: int) -> None:
		"""
		Calculate forwards in time using the Verlet velocity algorithm
		:param time_index: timestep to calculate from
		:type time_index: int
		:return: None
		:rtype: None
		"""
		# Calc new positions
		self.positions[time_index + 1] = apply_periodic_boundaries(
			self.positions[time_index]
			+ self.velocities[time_index] * self.time_step
			+ self.time_step ** 2 / 2 * self.forces[1]
			, self.box_size
		)

		# Calc new force
		distance_vectors = get_distance_vectors(self.positions[time_index + 1], self.box_size, self.dimension)
		distances = sum_squared(distance_vectors)
		self.forces[0] = self.forces[1]
		self.forces[1] = force(distance_vectors, distances)

		# Calc new velocities
		self.velocities[time_index + 1] = self.velocities[time_index] + self.time_step / 2 * np.sum(self.forces, axis=0)

		# Calculate potential energy
		self.potential_energy_array[time_index+1] = self.potential_energy(distances)



if __name__ == "__main__":
	pass
