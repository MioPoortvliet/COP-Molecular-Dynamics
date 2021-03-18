import matplotlib.pyplot as plt
from src.IO_utils import load_and_concat, load_json
import numpy as np

def plot_positions(fpath, ticks=5) -> None:
	positions = load_and_concat(fpath, "positions")
	velocities = load_and_concat(fpath, "velocities")
	properties = load_json(fpath)
	plt.plot(positions[::,::,0], '.')
	plt.xlabel("Time $t$")
	plt.xticks(np.linspace(0, velocities.shape[0], ticks), [f"{x:,.2e}" for x in np.linspace(0, properties["end_time"], ticks)])
	plt.ylabel("Position")
	plt.savefig(f'img/{fpath.replace("/","")}-positions.pdf')
	plt.show()

	plt.hist(velocities[-1,::,0], bins=100)
	plt.xlabel("Velocity")
	plt.ylabel("Counts")
	plt.savefig(f'img/{fpath.replace("/","")}-velocities.pdf')
	plt.show()

	"""
	plt.scatter(arr[-1,::,0],arr[-1,::,1])#, '.', ms=0.1)
	plt.xlabel("Time index")
	plt.ylabel("Position")
	plt.show()
	"""


def plot_energies(path, ticks=5):
	properties = load_json(path)

	velocities = load_and_concat(path, "velocities")
	potential_energy = load_and_concat(path, "potential_energy")

	kinetic_energy = .5 * np.sum(velocities ** 2, axis=-1) * properties["particle_mass"]

	plt.plot(np.sum(kinetic_energy, axis=-1), '.', label="ke")
	plt.plot(np.sum(potential_energy, axis=-1), '.', label="pe")
	plt.plot(np.sum(potential_energy, axis=-1)+np.sum(kinetic_energy, axis=-1), '.', label="total")

	plt.xlabel("Time $t$")
	plt.xticks(np.linspace(0, velocities.shape[0], ticks), [f"{x:,.2e}" for x in np.linspace(0, properties["end_time"], ticks)])
	plt.ylabel("Energy [J]")
	plt.legend()
	plt.savefig(f'img/{path.replace("/","")}-energies.pdf')
	plt.show()


def plot_correlation_function(correlation_function, distance, properties):
	plt.bar(distance, correlation_function, width=distance[1]-distance[0])
	plt.xlabel("Distance from molecule $r$ [-]")
	plt.ylabel("Correlation function $g(r)$ [-]")
	plt.title(f'{properties["particles"]} particles, {properties["total_steps"]} steps, $\\rho={properties["density"]}$, $T={properties["temperature"]}$')
	plt.savefig(f'img/{properties["path"].replace("/","")}-correlation_function.pdf')
	plt.show()


def rhoT_plot(pressure_array, temprange, densityrange, fpath):
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
	plt.savefig(f'img/{fpath.replace("/","")}-pressure.pdf')
	plt.show()
