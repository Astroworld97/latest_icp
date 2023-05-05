#quaternion implementation from 577 class notes. All units in inches.
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from numpy.linalg import eig
from helpers_english_units import *
import colorsys
import hashlib
import copy

#main
#section 1: define constants and data structures
maxIterations = 10000
tolerance = 0.1
matchDict = {}
colorDictP = {}
# Define the cylinder height and radius
h = 12
r = .435 #.87/2
modelBlueRange = [h - 2, h] #section of analytical model with color tape
modelRedRange = [0.00, 2.00] #section of analytical model that is wooden

#section 2: define arrays (aka point clouds) and initialize dictionaries
point_cloud_p, colorDictP = np.array(generate_point_cloud_p(r, h, colorDictP))
plot(point_cloud_p, colorDictP)
point_cloud_q, colorDictP = flip_180(point_cloud_p, colorDictP)
plot(point_cloud_p, colorDictP)