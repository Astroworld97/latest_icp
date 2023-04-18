#quaternion implementation from 577 class notes
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from numpy.linalg import eig
from helpers import *
import copy

q = [3,1,-2,1]
v = [-1,2,3]
q_star = quat_conjugate(q)

#method 1: left & right quat_mult
left = quat_mult(q, v)
right = quat_mult(left, q_star)
right = extract_vect_from_quat(right)
print(right)

#method 2: equivalent aka eqn 7
norm_squared = (math.sqrt(q[1]**2 + q[2]**2 + q[3]**2))**2
first_part_scalar = (q[0]**2 - norm_squared)
first_part = copy.deepcopy(v)
for i in range(len(first_part)):
    first_part[i]*=first_part_scalar

second_part_scalar = 2*quat_dot_product(q, v)
second_part = extract_vect_from_quat(q)
for i in range(len(second_part)):
    second_part[i]*=second_part_scalar

third_part_scalar = 2*q[0]
third_part = quat_cross_product(q, v)
for i in range(len(third_part)):
    third_part[i]*=third_part_scalar

eqn_7 = np.array(first_part) + np.array(second_part) + np.array(third_part)
eqn_7 = eqn_7.tolist()
for i in range(len(eqn_7)):
    eqn_7[i] = math.floor(eqn_7[i])
print(eqn_7)

#Both methods should print results equivalent to each other