#quaternion implementation from 577 class notes
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from numpy.linalg import eig
from helpers import *

#main
#section 1: define constants and data structures
maxIterations = 1000
tolerance = 2
matchDict = {}
# Define the cylinder height and radius
h = 12
r = .435 #.87/2

#section 2: define arrays (aka point clouds) and initialize dictionaries
point_cloud_q = np.array(generate_point_cloud_p(r, h))
copy = point_cloud_q.copy()
point_cloud_p = apply_initial_translation_and_rotation(copy)
# point_cloud_p = add_noise(point_cloud_p)
# plot_two_point_clouds(point_cloud_p, point_cloud_q)
plot(point_cloud_p)
M = np.zeros((4, 4)) 
b = 0
q = [0,0,0,0] #aka quat
q_centroid = [0,0,6]
# createMatchDictionary(point_cloud_p, matchDict)

# #section 3: iterate
for i in range(maxIterations):

    match(point_cloud_p, matchDict, q, i, q_centroid) #fill the matchDict and the distDict with the current matches

    if(i>0): #only check for error after the 0th loop
        err = error(point_cloud_p, point_cloud_q, b, q, matchDict)
        print(err)
        if err<tolerance:
            break

    p_centroid = point_cloud_centroid(point_cloud_p)

    for point_p in point_cloud_p:
        point_p = tuple(point_p)
        point_q = matchDict[point_p]
        p_prime = calc_single_prime(point_p, p_centroid)
        q_prime = calc_single_prime(point_q, q_centroid)
        P_i = create_prime_matrix_p(p_prime)
        Q_i = create_prime_matrix_q(q_prime)
        M_i = calc_single_M(P_i, Q_i)
        M+=M_i

    q = calc_quat(M) #aka quat
    norm = quat_norm(q)
    q = [q[0]/norm, q[1]/norm, q[2]/norm, q[3]/norm]
    q_star = quat_conjugate(q) #conjugate of q, aka quat
    b = calc_b(q_centroid, p_centroid, q, q_star)
    for i, point_p in enumerate(point_cloud_p):
        left_curr = quat_mult(q, point_p)
        right_curr = quat_mult(left_curr, q_star)
        Rp = [right_curr[1],right_curr[2], right_curr[3]]
        point_p = [Rp[0] + b[0], Rp[1] + b[1], Rp[2] + b[2]]
        point_cloud_p[i] = point_p
    plot(point_cloud_p)
    
# for i, point_p in enumerate(point_cloud_p):
#     point_p = tuple(point_p)
#     point_q = closest_point_on_cylinder(point_p, 12, .435, [0, 0, 0])
#     point_p = point_q
#     point_cloud_p[i] = point_p

err = error(point_cloud_p, point_cloud_q, b, q, matchDict)
print(err)
print(p_centroid)
print(q_centroid)
plot(point_cloud_p)