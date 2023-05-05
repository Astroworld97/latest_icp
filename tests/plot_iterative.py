import matplotlib.pyplot as plt
import numpy as np
import colorsys

# Generate some sample data
x = np.random.rand(10)
y = np.random.rand(10)

# Create a figure and axis object
fig, ax = plt.subplots()

# Iterate over the data and plot each point
for i in range(len(x)):
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    color = np.array([r,g,b])
    color=color.reshape(1,-1)
    ax.scatter(x[i], y[i], c='r', marker='o')

# Set the axis limits and labels
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Scatter Plot')

plt.show()
