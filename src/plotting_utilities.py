import matplotlib.pyplot as plt
import numpy as np

def plot_positions(pos, vel) -> None:
	plt.plot(pos[::,::,0], '.')
	plt.xlabel("Time index")
	plt.ylabel("Position")
	plt.show()

	plt.hist(vel[-1,::,0], bins=100)
	plt.xlabel("Velocity")
	plt.ylabel("Counts")
	plt.show()

	"""
	plt.scatter(arr[-1,::,0],arr[-1,::,1])#, '.', ms=0.1)
	plt.xlabel("Time index")
	plt.ylabel("Position")
	plt.show()
	"""


def plot_energies(kinetic_energy, potential_energy):
	plt.plot(np.sum(kinetic_energy, axis=-1), '.', label="ke")
	plt.plot(np.sum(potential_energy, axis=-1), '.', label="pe")
	plt.plot(np.sum(potential_energy, axis=-1)+np.sum(kinetic_energy, axis=-1), '.', label="total")
	plt.legend()
	plt.show()


def plot_correlation_function(correlation_function, distance, properties):
	plt.bar(distance, correlation_function)
	plt.xlabel("Unitless distance from molecule $r$ [-]")
	plt.ylabel("Correlation function $g(r)$ [-]")
	plt.title(f'{properties["particles"]} particles, {properties["total_steps"]} steps, $\\rho={properties["unitless_density"]}$, $T={properties["unitless_temperature"]}$')
	plt.show()