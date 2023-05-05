import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import colorsys

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the cylinder parameters
r = 1.0
h = 12.0
num_points = 1200

# Define the colors for each half of the cylinder
color1 = (17/360, 125/255, 210/255) # HSV values
color2 = (0.0, 1.0, 1.0) # HSV values

# Convert HSV colors to RGB colors
color1_rgb = colorsys.hsv_to_rgb(color1[0], color1[1], color1[2])
color2_rgb = colorsys.hsv_to_rgb(color2[0], color2[1], color2[2])

# Create the cylinder surface
theta = np.linspace(0, 2*np.pi, num_points)
z = np.linspace(0, h, num_points)
r, theta = np.meshgrid(r, theta)
x = r*np.cos(theta)
y = r*np.sin(theta)
z, _ = np.meshgrid(z, theta)

# Define the color values for each section of the cylinder surface
facecolors = np.zeros((num_points, num_points, 4))
for i in range(num_points):
    if i < num_points - 200:
        facecolors[:,i,:] = (*color1_rgb, 1)
    else:
        facecolors[:,i,:] = (*color2_rgb, 1)

# Plot the cylinder surface
ax.plot_surface(x, y, z, facecolors=facecolors, alpha=0.3)

# Show the plot
plt.show()
