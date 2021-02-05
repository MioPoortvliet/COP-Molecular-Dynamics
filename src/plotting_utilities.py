import matplotlib.pyplot as plt

def plot_positions(arr, end_time):
	plt.plot(arr[::,::,0], '.', ms=0.1)
	plt.xlabel("Time index")
	plt.ylabel("Position")
	plt.show()