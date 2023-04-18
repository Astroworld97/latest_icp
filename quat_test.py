import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from helpers import *

def dot_product(p, q): #inputs are quaternions. Remember the 0th element is irrelevant here.
    return p[1]*q[1] + p[2]*q[2] + p[3]*q[3] #returns a scalar

def cross_product(p, q): #inputs are quaternions. Remember the 0th element is irrelevant here.
    i_hat = p[2]*q[3] - p[3]*q[2]
    j_hat = p[3]*q[1] - p[1]*q[3]
    k_hat = p[1]*q[2] - p[2]*q[1]
    return [i_hat, j_hat, k_hat] #returns a vector

def quat_mult(p,q): #quaternion multiplication. Inputs p and q are quaternions.
    first_term = p[0]*q[0]
    dot_prod = dot_product(p, q)
    first_term = first_term - dot_prod
    cross_prod = cross_product(p, q)
    i_hat = p[0]*q[1] + q[0]*p[1] + cross_prod[0] 
    j_hat = p[0]*q[2] + q[0]*p[2] + cross_prod[1]
    k_hat = p[0]*q[3] + q[0]*p[3] + cross_prod[2]
    return [first_term, i_hat, j_hat, k_hat]

def conjugate(quat): #inputs are quaternions. Calculates the conjugate of a quaternion.
    first_term = quat[0]
    i_hat = -quat[1]
    j_hat = -quat[2]
    k_hat = -quat[3]
    return [first_term, i_hat, j_hat, k_hat]

def point_cloud_centroid(point_cloud): #computes centroid/mean/center of mass of point cloud
    arr = point_cloud
    sum_x = 0
    sum_y = 0
    sum_z = 0

    if len(point_cloud)==0:
        return [0,0,0]

    for point in arr:
        sum_x += point[0]
        sum_y += point[1]
        sum_z += point[2]
    
    avg_x = round(sum_x/len(arr), 4)
    avg_y = round(sum_y/len(arr), 4)
    avg_z = round(sum_z/len(arr), 4)

    return [avg_x, avg_y, avg_z] 

def calc_single_prime(point, centroid):
    prime_x = point[0] - centroid[0]
    prime_y = point[1] - centroid[1] 
    prime_z = point[2] - centroid[2]
    return [prime_x, prime_y, prime_z]

def create_prime_matrix_p(prime_vector):
    values = [0, -1*prime_vector[1], -1*prime_vector[2], -1*prime_vector[3], -1*prime_vector[1], 0, -1*prime_vector[3], -1*prime_vector[2], -1*prime_vector[2], -1*prime_vector[3], 0, -1*prime_vector[1], -1*prime_vector[3], -1*prime_vector[2], -1*prime_vector[1], 0]
    return np.array(values).reshape((4, 4))

def create_prime_matrix_q(prime_vector):
    values = [0, -1*prime_vector[1], -1*prime_vector[2], -1*prime_vector[3], prime_vector[1], 0, -1*prime_vector[3], prime_vector[2], prime_vector[2], prime_vector[3], 0, -1*prime_vector[1], prime_vector[3], -1*prime_vector[2], prime_vector[1], 0]
    return np.array(values).reshape((4, 4))

#functions
def apply_initial_translation_and_rotation(inputArr):
    # Define rotation matrix for 5 degree rotation around y-axis
    inputArr = np.array(inputArr)
    r = Rotation.from_euler('y', 20, degrees=True)
    rot_matrix = r.as_matrix()

    # Define translation vector
    translation = np.array([1, 1, 1])

    # Apply rotation and translation
    outputArr = inputArr.dot(rot_matrix.T) + translation

    # Round the values in the output matrix to integers
    outputArr = np.round(outputArr).astype(int)

    return outputArr.tolist()

#main

p = [3, 1, -2, 1]
q = [2, -1, 2, 3]

# point_cloud_q = [
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ]

# point_cloud_q = [
#     [2, 3, 4],
#     [5, 6, 7],
#     [8, 9, 10]
# ]

# point_cloud_p = point_cloud_q.copy()
# point_cloud_q = apply_initial_translation_and_rotation(point_cloud_q)

# print(point_cloud_p)
# print(point_cloud_q)

# p_centroid = point_cloud_centroid(point_cloud_p)
# q_centroid = point_cloud_centroid(point_cloud_q)

# print(p_centroid)
# print(q_centroid)

# point_p_0 = point_cloud_p[0]
# point_q_0 = point_cloud_q[0]

# print(point_p_0)
# print(point_q_0)

# prime_p = calc_single_prime(point_p_0, p_centroid)
# prime_q = calc_single_prime(point_q_0, q_centroid)

# print(prime_p)
# print(prime_q)

# print(dot_product(p, q))
# print(cross_product(p, q))
print(quat_mult(p,q))
print(conjugate(p))
