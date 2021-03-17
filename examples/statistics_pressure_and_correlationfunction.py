import numpy as np
from typing import Tuple

from src.utils import N_runs, pressure_and_correlation_function


def run_statistics(
		unitless_density,
		unitless_temperature,
		N,
		steps_per_run=100,
		cleanup=True
					) -> Tuple[
									Tuple[float, float],
									Tuple[np.ndarray, np.ndarray, np.ndarray]
								]:


	paths = N_runs(
		unitless_density=unitless_density,
		unitless_temperature=unitless_temperature,
		N=N,
		steps_per_run=steps_per_run
	)
	return paths


if __name__ == "__main__":
	paths = run_statistics(unitless_density=1, unitless_temperature=1, N=2, steps_per_run=2000)
	pressure_and_correlation_function(paths, cleanup=True)