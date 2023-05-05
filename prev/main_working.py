#quaternion implementation from 577 class notes
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from numpy.linalg import eig
from helpers_working import *

#main
#section 1: define constants and data structures
maxIterations = 1000
tolerance = 0.1
matchDict = {}
# Define the cylinder height and radius
h = 12
r = .435 #.87/2

#section 2: define arrays (aka point clouds) and initialize dictionaries
point_cloud_q = np.array(generate_point_cloud_p(r, h))
copy = point_cloud_q.copy()
point_cloud_p = apply_initial_translation_and_rotation(copy)
# point_cloud_p = add_noise(point_cloud_p)
plot(point_cloud_p)
M = np.zeros((4, 4)) 
b = 0
quat = [0,0,0,0] #aka quat
p_centroid = point_cloud_centroid(point_cloud_p)
q_centroid = [0,0,6]

#section 3a: single iteration (translation)
centroid_diff_x = q_centroid[0] - p_centroid[0]
centroid_diff_y = q_centroid[1] - p_centroid[1]
centroid_diff_z = q_centroid[2] - p_centroid[2]
for i, point_p in enumerate(point_cloud_p):
    point_p = [point_p[0] + centroid_diff_x, point_p[1] + centroid_diff_y, point_p[2] + centroid_diff_z]
    point_cloud_p[i] = point_p
plot(point_cloud_p)

# #section 3b: iterate (rotation)
point_cloud_p_best = point_cloud_p
best_err = 10000000
for i in range(maxIterations):

    match(point_cloud_p, matchDict, quat, i, q_centroid) #fill the matchDict with the current matches

    if(i>0): #only check for error after the 0th loop
        err = error(point_cloud_p, point_cloud_q, b, quat, matchDict)
        if err<best_err:
            best_err = err
            print("update")
            point_cloud_p_best = point_cloud_p
        print(err)
        if err<tolerance:
            break

    p_centroid = point_cloud_centroid(point_cloud_p)

    for point_p in point_cloud_p:
        point_p = tuple(point_p)
        point_q = matchDict[point_p]
        p_prime = calc_single_prime(point_p, p_centroid)
        q_prime = calc_single_prime(matchDict[point_p], q_centroid)
        P_i = create_prime_matrix_p(p_prime)
        Q_i = create_prime_matrix_q(q_prime)
        M_i = calc_single_M(P_i, Q_i)
        M+=M_i
    # plot(point_cloud_p)

    quat = calc_quat(M) #aka quat
    norm = quat_norm(quat)
    quat = [quat[0]/norm, quat[1]/norm, quat[2]/norm, quat[3]/norm]
    quat_star = quat_conjugate(quat) #conjugate of q, aka quat
    b = calc_b(q_centroid, p_centroid, quat, quat_star)
    for i, point_p in enumerate(point_cloud_p):
        left_curr = quat_mult(quat, point_p)
        right_curr = quat_mult(left_curr, quat_star)
        Rp = [right_curr[1],right_curr[2], right_curr[3]]
        point_p = [Rp[0] + b[0], Rp[1] + b[1], Rp[2] + b[2]]
        point_cloud_p[i] = point_p
print(best_err)
plot(point_cloud_p_best)

err = error(point_cloud_p, point_cloud_q, b, quat, matchDict)
print(err)
print(p_centroid)
print(q_centroid)
# plot(point_cloud_p)