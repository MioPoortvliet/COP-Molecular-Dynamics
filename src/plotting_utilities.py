import matplotlib.pyplot as plt

def plot_positions(arr, end_time):
	plt.plot(arr[::,::,0], '.', ms=1)
	plt.xlabel("Time index")
	plt.ylabel("Position")
	plt.show()


	plt.scatter(arr[-1,::,0],arr[-1,::,1])#, '.', ms=0.1)
	plt.xlabel("Time index")
	plt.ylabel("Position")
	plt.show()