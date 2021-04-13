import numpy as np
import matplotlib.pyplot as plt
import sys

folder_name = sys.argv[1]

neck = np.loadtxt(folder_name + "neck_angle.csv", delimiter=",", skiprows=1)
eye = np.loadtxt(folder_name + "eye_angle.csv", delimiter=",", skiprows=1)

plt.figure
plt.plot(neck[:, 0], (180.0 / np.pi) * neck[:, 1], label="Neck angle")
plt.plot(eye[:, 0],  (180.0 / np.pi) * eye[:, 1], label="Eye angle")
plt.legend(fontsize=20)
plt.title("Neck angle and gaze angle", fontsize=30)
plt.ylabel("Angles [deg]", fontsize=20)
plt.tick_params(labelsize=20)
plt.show()
