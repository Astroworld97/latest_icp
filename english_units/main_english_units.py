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
point_cloud_q, colorDictP = np.array(generate_point_cloud_p(r, h, colorDictP))
point_cloud_p = point_cloud_q.copy()
# colorDictP = txt_to_dict('exportDict.txt')
point_cloud_p = list(colorDictP.keys())
plot(point_cloud_p, colorDictP)
point_cloud_p, colorDictP = apply_initial_translation_and_rotation(point_cloud_p, colorDictP)
# point_cloud_p, colorDictP = flip_180(point_cloud_p, colorDictP, h)
# point_cloud_p = add_noise(point_cloud_p, colorDictP)
plot(point_cloud_p, colorDictP)
M = np.zeros((4, 4)) 
b = 0
quat = [0,0,0,0] #aka quat
p_centroid = point_cloud_centroid(point_cloud_p)
q_centroid = [0,0,6]

# #section 3: iterate (rotation)
point_cloud_p_best = point_cloud_p
colorDictP_best = colorDictP
best_err = 10000000
for i in range(maxIterations):

    match(point_cloud_p, matchDict, quat, i, q_centroid, colorDictP, modelBlueRange, modelRedRange) #fill the matchDict with the current matches

    if(i>0): #only check for error after the 0th loop
        err = error(point_cloud_p, point_cloud_q, b, quat, matchDict, colorDictP, modelBlueRange, modelRedRange)
        if err<best_err:
            best_err = err
            print("update")
            point_cloud_p_best = point_cloud_p
            colorDictP_best = colorDictP
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

    quat = calc_quat(M) #aka quat
    norm = quat_norm(quat)
    quat = [quat[0]/norm, quat[1]/norm, quat[2]/norm, quat[3]/norm]
    quat_star = quat_conjugate(quat) #conjugate of q, aka quat
    b = calc_b(q_centroid, p_centroid, quat, quat_star)
    for i, point_p in enumerate(point_cloud_p):
        left_curr = quat_mult(quat, point_p)
        right_curr = quat_mult(left_curr, quat_star)
        Rp = [right_curr[1], right_curr[2], right_curr[3]]
        point_color = colorDictP[tuple(point_p)]
        colorDictP[tuple(point_p)] = ()
        point_p = [Rp[0] + b[0], Rp[1] + b[1], Rp[2] + b[2]]
        colorDictP[tuple(point_p)] = point_color
        point_cloud_p[i] = point_p
print(best_err)
plot(point_cloud_p_best, colorDictP_best)
err = error(point_cloud_p, point_cloud_q, b, quat, matchDict, colorDictP_best, modelBlueRange, modelRedRange)
print(p_centroid)
print(q_centroid)

# #check if needs to flip
flipped = False
for i, point_p in enumerate(point_cloud_p_best):
    to_check = tuple(point_cloud_p_best[i])
    to_check_color =  colorDictP_best[to_check]
    if ((modelBlueRange[0] <= to_check[2] <= modelBlueRange[1]) and (color_match(to_check_color, (0.0, 1.0, 1.0)))) or ((modelRedRange[0] <= to_check[2] <= modelRedRange[1]) and (color_match(to_check_color, (240/360, 1.0, 1.0)))):
        flip_180(point_cloud_p_best, colorDictP_best, h)
        flipped = True
        break
if(flipped):
    plot(point_cloud_p_best, colorDictP_best)
    err = error(point_cloud_p_best, point_cloud_q, b, quat, matchDict, colorDictP_best, modelBlueRange, modelRedRange)
    print(err)
    print(p_centroid)
    print(q_centroid)
    
#section 4: check and iterate again (rotation)
# point_cloud_p_best = point_cloud_p
if(flipped):
    point_cloud_p = point_cloud_p_best
    best_err = 10000000
    for i in range(maxIterations):

        match(point_cloud_p, matchDict, quat, i, q_centroid, colorDictP, modelBlueRange, modelRedRange) #fill the matchDict with the current matches

        if(i>0): #only check for error after the 0th loop
            err = error(point_cloud_p, point_cloud_q, b, quat, matchDict, colorDictP, modelBlueRange, modelRedRange)
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

        quat = calc_quat(M) #aka quat
        norm = quat_norm(quat)
        quat = [quat[0]/norm, quat[1]/norm, quat[2]/norm, quat[3]/norm]
        quat_star = quat_conjugate(quat) #conjugate of q, aka quat
        b = calc_b(q_centroid, p_centroid, quat, quat_star)
        for i, point_p in enumerate(point_cloud_p):
            left_curr = quat_mult(quat, point_p)
            right_curr = quat_mult(left_curr, quat_star)
            Rp = [right_curr[1], right_curr[2], right_curr[3]]
            point_color = colorDictP[tuple(point_p)]
            colorDictP[tuple(point_p)] = ()
            point_p = [Rp[0] + b[0], Rp[1] + b[1], Rp[2] + b[2]]
            colorDictP[tuple(point_p)] = point_color
            point_cloud_p[i] = point_p
    print(best_err)
    plot(point_cloud_p_best, colorDictP)
    err = error(point_cloud_p, point_cloud_q, b, quat, matchDict, colorDictP, modelBlueRange, modelRedRange)
    print(p_centroid)
    print(q_centroid)