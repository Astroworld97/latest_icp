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
r = .87/2

#section 2: define arrays (aka point clouds) and initialize dictionaries
# point_cloud_p_1 = np.array([[[ 10,   5,  20],
#         [ 15,  20,  10],
#         [  5,  10,  15],
#         [ 20,  15,   5]],

#        [[ 20,  10,  15],
#         [  5,  15,  20],
#         [ 15,   5,  10],
#         [ 10,  20,   5]],

#        [[ 10,  15,   5],
#         [ 20,   5,  10],
#         [  5,  20,  15],
#         [ 15,  10,  20]],

#        [[ 15,  10,  30],
#         [ 10,  25,   5],
#         [ 20,   5,  15],
#         [  5,  20,  10]]])

point_cloud_p = np.array(generate_point_cloud_p(r, h))
print(point_cloud_p.shape)

point_cloud_p = apply_initial_translation_and_rotation(point_cloud_p)
# point_cloud_p = add_noise(point_cloud_p)
match(point_cloud_p, matchDict)
point_cloud_q = np.array(list((matchDict.values())))
point_cloud_q = point_cloud_q.reshape(point_cloud_p.shape)

M = np.zeros((4, 4)) 
b = 0
q = [0,0,0,0] #aka quat

# #section 3: iterate
for i in range(maxIterations):

    match(point_cloud_p, matchDict) #fill the matchDict and the distDict with the current matches

    if(i>0): #only check for error after the 0th loop
        err = error(point_cloud_p, point_cloud_q, b, q, matchDict)
        print(err)
        if err<tolerance:
            break
        # print(point_cloud_p)

    p_centroid = point_cloud_centroid_p(point_cloud_p)
    q_centroid = point_cloud_centroid_q(point_cloud_q)

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
    for i, point_p in enumerate(point_cloud_p):
        # if is_point_on_cyl(point_p, r, h):
        #     continue
        # else:
        point_p = point_p.astype(float)
        point_p += b
        point_cloud_p[i] = point_p
    
    plot(point_cloud_p)