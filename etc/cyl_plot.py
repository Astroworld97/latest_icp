import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the cylinder height and radius
h = 2
r = 1

# Define the number of points to use for the cylinder surface
num_points = 50

# Generate the x and y coordinates of the cylinder surface
theta = np.linspace(0, 2*np.pi, num_points)
x = r * np.cos(theta)
y = r * np.sin(theta)

# Generate the z coordinates of the cylinder surface
z = np.linspace(0, h, num_points)
z = np.tile(z, (num_points, 1)).T

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the cylinder surface
ax.plot_surface(x, y, z)

# Set the axis limits and labels
ax.set_xlim(-r, r)
ax.set_ylim(-r, r)
ax.set_zlim(0, h)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
