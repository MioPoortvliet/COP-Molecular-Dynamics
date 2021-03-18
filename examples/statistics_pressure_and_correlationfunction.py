"""
Run statistics for temperatures and densities.
Authors: Mio Poortvliet
"""
from src.utils import N_runs, pressure_fpaths, correlation_function_fpaths


if __name__ == "__main__":
    temps = [60, 120, 360]
    densities = [1680*1.2, 1680*0.8, 1680*0.3]
    for temp, density in zip(temps, densities):
        paths = N_runs(density=density, temperature=temp, N=10, steps=100, treshold=0.05, verbosity=0)
        print(temp, density)
        pressure_fpaths(paths)
        #correlation_function_fpaths(paths)