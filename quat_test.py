import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from helpers import *

#main

h = 12
r = .435 #.87/2
point_cloud_p = generate_point_cloud_p(r, h)
point_cloud_p_orig = point_cloud_p.copy()
# # plot_single_point_cloud(point_cloud_p_orig)
point_cloud_p_moved = apply_initial_translation_and_rotation(point_cloud_p)
plot_single_point_cloud(point_cloud_p_orig)
plot_single_point_cloud(point_cloud_p_moved)
plot_two_point_clouds(point_cloud_p_orig, point_cloud_p_moved)
