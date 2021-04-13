import numpy as np
import matplotlib.pyplot as plt
import sys
import os

iFolder = 0
_files = os.listdir(os.getcwd())
folders = []
for f in _files:
    if 'csv_records_' in f:
        folders.append(f)
        print(f)
folder_name = folders[iFolder]
file_name = 'all_spikes.csv'
path_name = os.getcwd() + '/' + folder_name + '/' + file_name
print(path_name)
spikes = np.loadtxt(path_name, delimiter=",", skiprows=1)
plt.figure
[n_spikes,cells]=np.histogram(spikes[:, 0],np.arange(np.min(spikes[:, 0]), np.max(spikes[:, 0])))
plt.bar(cells[:-1], n_spikes / (np.max(spikes[:, 1] * 0.001)))
plt.title("Firing rate per neuron", fontsize=30)
plt.xlabel("Neuron ID", fontsize=20)
plt.ylabel("Firing rate [Hz]", fontsize=20)
plt.tick_params(labelsize=20)
plt.show()
