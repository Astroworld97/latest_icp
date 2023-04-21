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
tolerance = 0.001
matchDict = {}
# Define the cylinder height and radius
h = 12
r = .435 #.87/2

#section 2: define arrays (aka point clouds) and initialize dictionaries
point_cloud_q = np.array(generate_point_cloud_p(r, h))
point_cloud_p = point_cloud_q.copy()
point_cloud_q = assign_color_orange(point_cloud_q)
point_cloud_p = assign_color_orange(point_cloud_p)
point_cloud_p = apply_initial_translation_and_rotation(point_cloud_p)
# plot(point_cloud_p)
# point_cloud_p = add_noise(point_cloud_p)
# plot_single_point_cloud(point_cloud_p)
# plot_single_point_cloud(point_cloud_q)
# plot_two_point_clouds(point_cloud_p, point_cloud_q)
M = np.zeros((4, 4)) 
b = 0
q = [0,0,0,0] #aka quat
q_centroid = [0,0,6]
createMatchDictionary(point_cloud_p, matchDict)

# #section 3: iterate
for i in range(maxIterations):

    match(point_cloud_p, matchDict, q, i, q_centroid, point_cloud_q) #fill the matchDict and the distDict with the current matches

    if(i>0): #only check for error after the 0th loop
        err = error(point_cloud_p, point_cloud_q, b, q, matchDict)
        print(err)
        if err<tolerance:
            break

    p_centroid = point_cloud_centroid_p(point_cloud_p)

    for point_p in point_cloud_p:
        point_p = tuple(point_p)
        point_q = matchDict[point_p]
        p_prime = calc_single_prime(point_p, p_centroid)
        q_prime = calc_single_prime(point_q, q_centroid)
        P_i = create_prime_matrix_p(p_prime)
        Q_i = create_prime_matrix_q(q_prime)
        M_i = calc_single_M(P_i, Q_i)
        M+=M_i

    q = calc_q(M) #aka quat
    q_star = quat_conjugate(q) #conjugate of q, aka quat
    b = calc_b(q_centroid, p_centroid, q, q_star)
    if abs(p_centroid[0] - q_centroid[0]) < 0.001 and abs(p_centroid[1] - q_centroid[1]) < 0.001 and abs(p_centroid[2] - q_centroid[2]) < 0.001:
        for i, point_p in enumerate(point_cloud_p):
            is_orange = point_p[3]
            point_p = extract_vect_from_quat(quat)
            left_curr = quat_mult(q, point_p)
            right_curr = quat_mult(left_curr, q_star)
            Rp = [right_curr[1],right_curr[2], right_curr[3]]
            point_p = [Rp[0], Rp[1], Rp[2], is_orange]
            point_cloud_p[i] = point_p
    else:
        for i, point_p in enumerate(point_cloud_p):
            is_orange = point_p[3]
            point_p = [point_p[0]+b[0], point_p[1]+b[1], point_p[2]+b[2], is_orange]
            point_cloud_p[i] = point_p
        
    print("")
    plot(point_cloud_p)
    print(p_centroid)
    print(q_centroid)
    