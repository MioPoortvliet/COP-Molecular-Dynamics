import matplotlib.pyplot as plt

def plot_positions(arr, end_time):
	plt.scatter(arr[0,::,0],arr[0,::,1])#, '.', ms=0.1)
	plt.xlabel("Time index")
	plt.ylabel("Position")
	plt.show()