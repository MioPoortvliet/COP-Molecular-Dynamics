from src.molecular_dynamics import Simulation
from src.plotting_utilities import *


def main():
	sim = Simulation(particles=3, dimension=2, box_size=(10, 100), time_step=0.1, end_time=10)
	sim.run_sim()



if __name__ == "__main__":
	main()