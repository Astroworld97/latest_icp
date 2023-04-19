#quaternion implementation from 577 class notes
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from numpy.linalg import eig
from helpers import *
import random
import sys

# Define the cylinder height and radius
height = 12
r = 0.435 #0.87/2

# generate 100 points on the cylinder
points = []
num_points = 100

def is_negative():
    rand_int = random.randint(0, 1)
    rand_sign = -1 if rand_int == 0 else 1
    return rand_sign


for i in range(num_points):
    point = []
    x = random.uniform(-0.435, 0.435)
    sign_y = is_negative()
    y = (math.sqrt(r**2-x**2)) * sign_y
    z = random.uniform(-height/2, height/2)
    point = [x,y,z]
    points.append(point)

r2 = r**2

for point in points:
    x2 = point[0]**2
    y2 = point[1]**2
    suma = x2 + y2
    if np.isclose(suma, r**2, rtol=1e-3, atol=1e-3):
        print("a and b are equivalent")
    else:
        print("a and b are not equivalent")
        print("error!!!")
        sys.exit(1)

# Convert the points to a numpy array
points = np.array(points)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points as a scatter plot
ax.scatter(points[:, 0], points[:, 1], points[:, 2])

# Add labels and set the axis limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-r, r)
ax.set_ylim(-r, r)
ax.set_zlim(-height/2, height/2)

# Show the plot
plt.show()
