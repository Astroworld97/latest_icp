import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from numpy.linalg import eig
import random
import colorsys

x = 1
y = 10
z = 6

# Define the HSV color values
#red
# hue = 0   # Hue value in the range 0 to 1
# saturation = 1  # Saturation value in the range 0 to 1
# value = 1  # Value (brightness) value in the range 0 to 1

#wood
# hue = 17/360 # Hue value in the range 0 to 1
# saturation = 125/255 # Saturation value in the range 0 to 1
# value = 210/255 # Value (brightness) value in the range 0 to 1

# Convert the HSV color to an RGB color
r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
color = np.array([r,g,b])
color=color.reshape(1,-1)

# Pass the RGB color to the c parameter in the scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=color, marker='o')

plt.show()