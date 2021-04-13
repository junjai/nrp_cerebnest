import numpy as np
import matplotlib.pyplot as plt
import sys

print(sys.argv)
file_name = sys.argv[1]

spikes = np.loadtxt(file_name, delimiter=",", skiprows=1)
plt.figure
[n_spikes,cells]=np.histogram(spikes[:, 0],np.arange(np.min(spikes[:, 0]), np.max(spikes[:, 0])))
plt.bar(cells[:-1], n_spikes / (np.max(spikes[:, 1] * 0.001)))
plt.title("Firing rate per neuron", fontsize=30)
plt.xlabel("Neuron ID", fontsize=20)
plt.ylabel("Firing rate [Hz]", fontsize=20)
plt.tick_params(labelsize=20)
plt.show()
