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


def rhoT_plot(pressure_array, temprange, densityrange):
	cb = plt.imshow(np.rot90(pressure_array[::-1,::]))
	plt.colorbar(cb)
	if temprange.size > 7:
		plt.xticks(ticks=np.arange(0, temprange.size, int(temprange.size/5)), labels=np.round(temprange, 2)[::int(temprange.size/5)])
	else:
		plt.xticks(ticks=np.arange(temprange.size), labels=np.round(temprange, 2))

	plt.xlabel("Temperature")
	if densityrange.size > 7:
		plt.yticks(ticks=np.arange(0, densityrange.size, int(densityrange.size/5))[::-1], labels=np.round(densityrange, 2)[::int(densityrange.size/5)])
	else:
		plt.yticks(ticks=np.arange(densityrange.size)[::-1], labels=np.round(densityrange, 2))
	plt.ylabel("Density")
	plt.title("Pressure")
	plt.show()
