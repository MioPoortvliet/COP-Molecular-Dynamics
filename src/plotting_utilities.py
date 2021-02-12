import matplotlib.pyplot as plt
import numpy as np

def plot_positions(pos, vel, end_time) -> None:
	plt.plot(pos[::,::,0], '.')
	plt.xlabel("Time index")
	plt.ylabel("Position")
	plt.show()

	plt.hist(vel[-1,::,0])
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
