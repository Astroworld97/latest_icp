import numpy as np

# Define cylinder parameters
height = 1.0
radius = 0.5
num_segments = 32
num_height_segments = 10

# Generate point data for cylinder
theta = np.linspace(0, 2*np.pi, num_segments)
z = np.linspace(0, height, num_height_segments)
z, theta = np.meshgrid(z, theta)
x = radius * np.cos(theta)
y = radius * np.sin(theta)

# Reshape arrays for plotting
x = x.ravel()
y = y.ravel()
z = z.ravel()

# Combine arrays into (x, y, z) coordinates
cylinder_points = np.column_stack((x, y, z))