"""
Make an animation of the simulated particle positions.
Authors: Jonah Post
"""
from src.animation import Animation
from src.utils import N_runs


if __name__ == "__main__":
	# Get the first path in the list of length one
	path = N_runs(
		N=1,
		temperature=300,
		density=1.602,
		steps=5000,
		time_step=1e-2
	)[0]

	# Initialize animation class
	ani = Animation(path, frameskip=10)
	# Show the animation
	ani.run()