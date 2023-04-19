#helper functions
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from numpy.linalg import eig
import random

def apply_initial_translation_and_rotation(inputArr):
    # Define rotation matrix for 5 degree rotation around y-axis
    #Test 1:
    r = Rotation.from_euler('y', 50, degrees=True)
    #Test 2:
    # r = Rotation.from_euler('x', 45, degrees=True)
    # #Test 3:
    # r = Rotation.from_euler('z', 90, degrees=True)
    # #Test 4:
    # r = Rotation.from_euler('x', -180, degrees=True)
    # #Test 5:
    # r = Rotation.from_euler('y', -60, degrees=True)
    # #Test 6:
    # r = Rotation.from_euler('z', 30, degrees=True)
    # #Test 7:
    # r = Rotation.from_euler('zy', [45, 30], degrees=True)
    # #Test 8:
    # r = Rotation.from_euler('xz', [90, 45], degrees=True)
    # #Test 9:
    # r = Rotation.from_euler('yx', [60, 90], degrees=True)
    rot_matrix = r.as_matrix()

    # Define translation vector
    #test 1:
    translation = np.array([0, 0, 0])
    #test 2:
    # translation = np.array([1.2, -0.5, 2.3])
    # #test 3:
    # translation = np.array([-3.4, 2.1, -1.9])
    # #test 4:
    # translation = np.array([0.8, 1.1, 2.9])
    # #test 5:
    # translation = np.array([2.3, -1.6, -0.9])
    # #test 6:
    # translation = np.array([-1.1, -0.9, 3.2])
    # Apply rotation and translation
    outputArr = inputArr.dot(rot_matrix.T) + translation

    # Round the values in the output matrix to integers
    outputArr = np.round(outputArr).astype(int)

    return outputArr

def closest_point_on_cylinder(point, height, rad, origin):
    #point is at x==0 and y==0, arbitrary height
    # if point[0] == 0 and point[1] == 0:
    #     if point[2]>=(height/2):
    #         z = height/2
    #     # elif point[2]<(-height/2):
    #     #     z = -height/2
    #     else:
    #         z = -height/2
    #     return [point[0],point[1],z]
    # if point[0] == 0 and point[1] == 0 and point[2] == 0:
    #     return [point[0],point[1], point[2]]

    #point is on the circle's circumference, arbitrary height
    # if (point[0]**2 + point[1]**2 == rad**2):
    #     if point[2]>=(height/2):
    #         z = height/2
    #     elif point[2]<=(-height/2):
    #         z = -height/2
    #     else:
    #         z = point[2]
    #     return [point[0], point[1], z]
    
    #point is at arbitrary x, y, and z
    if point[2]>=(height/2):
        z = height/2
    elif point[2]<=(-height/2):
        z = -height/2
    else:
        z = point[2]

    x = point[0]/math.sqrt(point[0]**2 + point[1]**2)
    y = point[1]/math.sqrt(point[0]**2 + point[1]**2)
    x = x * rad
    y = y * rad
    left = round((x**2 + y**2), 4)
    right = round((rad**2), 4)
    assert left == right, print([x,y,z])
    return [x,y,z]

def quat_dot_product(p, q): #inputs are quaternions. Remember the 0th element is irrelevant here.
    if len(p)==3:
        p = vect_to_quat(p)
    if len(q)==3:
        q = vect_to_quat(q)
    return p[1]*q[1] + p[2]*q[2] + p[3]*q[3] #returns a scalar

def quat_cross_product(p, q): #inputs are quaternions. Remember the 0th element is irrelevant here.
    if len(p)==3:
        p = vect_to_quat(p)
    if len(q)==3:
        q = vect_to_quat(q)
    i_hat = p[2]*q[3] - p[3]*q[2]
    j_hat = p[3]*q[1] - p[1]*q[3]
    k_hat = p[1]*q[2] - p[2]*q[1]
    return [i_hat, j_hat, k_hat] #returns a vector

def quat_conjugate(quat): #inputs are quaternions. Calculates the conjugate of a quaternion.
    first_term = quat[0]
    i_hat = -quat[1]
    j_hat = -quat[2]
    k_hat = -quat[3]
    return [first_term, i_hat, j_hat, k_hat] 

def vect_to_quat(vect): #adds a zero for the real part to turn a vector into a quaternion
    return [0, vect[0], vect[1], vect[2]]

def extract_vect_from_quat(quat): #extracts the vector part from a quaternion.
    return [quat[1], quat[2], quat[3]]

def quat_mult(p,q): #quaternion multiplication. Inputs p and q are quaternions.
    if len(p)==3:
        p = vect_to_quat(p)
    if len(q)==3:
        q = vect_to_quat(q)
    first_term = p[0]*q[0]
    dot_prod = quat_dot_product(p, q)
    first_term = first_term - dot_prod
    cross_prod = quat_cross_product(p, q)
    i_hat = p[0]*q[1] + q[0]*p[1] + cross_prod[0] 
    j_hat = p[0]*q[2] + q[0]*p[2] + cross_prod[1]
    k_hat = p[0]*q[3] + q[0]*p[3] + cross_prod[2]
    return [first_term, i_hat, j_hat, k_hat]

def create_prime_matrix_p(prime_vector):
    prime_vector = [0, prime_vector[0], prime_vector[1], prime_vector[2]]
    values = [0, -1*prime_vector[1], -1*prime_vector[2], -1*prime_vector[3], prime_vector[1], 0, prime_vector[3], -1*prime_vector[2], prime_vector[2], -1*prime_vector[3], 0, prime_vector[1], prime_vector[3], prime_vector[2], -1*prime_vector[1], 0]
    return np.array(values).reshape((4, 4))

def create_prime_matrix_q(prime_vector):
    prime_vector = [0, prime_vector[0], prime_vector[1], prime_vector[2]]
    values = [0, -1*prime_vector[1], -1*prime_vector[2], -1*prime_vector[3], prime_vector[1], 0, -1*prime_vector[3], prime_vector[2], prime_vector[2], prime_vector[3], 0, -1*prime_vector[1], prime_vector[3], -1*prime_vector[2], prime_vector[1], 0]
    return np.array(values).reshape((4, 4))

def point_cloud_centroid_p(point_cloud): #computes centroid/mean/center of mass of point cloud
    arr = point_cloud
    sum_x = 0
    sum_y = 0
    sum_z = 0

    if len(point_cloud)==0:
        return [0,0,0]

    num_points = 0
    for point in arr:
        sum_x += point[0]
        sum_y += point[1]
        sum_z += point[2]
        num_points+=1
    
    avg_x = sum_x/num_points
    avg_y = sum_y/num_points
    avg_z = sum_z/num_points

    return [avg_x, avg_y, avg_z]

def point_cloud_centroid_q(point_cloud): #computes centroid/mean/center of mass of point cloud
    arr = point_cloud
    sum_x = 0
    sum_y = 0
    sum_z = 0

    if len(point_cloud)==0:
        return [0,0,0]

    num_points = 0
    for point in arr:
        sum_x += point[0]
        sum_y += point[1]
        sum_z += point[2]
        num_points+=1
    
    avg_x = sum_x/num_points
    avg_y = sum_y/num_points
    avg_z = sum_z/num_points

    return [avg_x, avg_y, avg_z]

def calc_single_prime(point, centroid):
    prime_x = point[0] - centroid[0]
    prime_y = point[1] - centroid[1] 
    prime_z = point[2] - centroid[2]
    return [prime_x, prime_y, prime_z]

def calc_single_M(P_i, Q_i):
    return np.matmul(P_i.T,Q_i)

def calc_q(M):
    eigenvalues,eigenvectors=eig(M)
    max_eigval_idx = np.argmax(eigenvalues)
    max_eigvec = eigenvectors[:, max_eigval_idx]
    return max_eigvec

def calc_b(q_centroid, p_centroid, q, q_star):
    first_mult = quat_mult(q, p_centroid)
    second_mult = quat_mult(first_mult, q_star)
    vect = [second_mult[1], second_mult[2], second_mult[3]]
    retval = np.array(q_centroid) - np.array(vect)
    return retval

def createMatchDictionary(point_cloud_p, matchDict): #creates the match dictionary, which associates each point in arr1 to the point closest to in arr2
    # for i, row in enumerate(point_cloud_p):
    #     for j, point_p in enumerate(row):
    #         matchDict[tuple(point_p)] = None
    for point in point_cloud_p:
        matchDict[tuple(point)] = None

def match(point_cloud_p, matchDict): #matches each of the points in one array to its closest corresponding point in the other array
    createMatchDictionary(point_cloud_p, matchDict)
    for point_p in point_cloud_p:
        point_q = tuple(closest_point_on_cylinder(point_p, 12, 0.87/2, [0, 0, 12/2]))
        point_p = tuple(point_p)
        matchDict[point_p] = point_q

def error(point_cloud_p, point_cloud_q, b, q, matchDict): 
    tot = 0
    q_star = quat_conjugate(q)
    for point_p in point_cloud_p:
        point_p = tuple(point_p)
        point_q = matchDict[point_p]
        Rp_i_left = quat_mult(q, point_p)
        Rp_i_right = quat_mult(Rp_i_left, q_star)
        Rp_i = Rp_i_right
        Rp_i = extract_vect_from_quat(Rp_i)
        curr = Rp_i + b - point_q
        norm_squared = (math.sqrt(curr[0]**2 + curr[1]**2 + curr[2]**2))**2
        tot+=norm_squared
    return tot

def is_point_on_cyl(point, rad, height):
    if (point[0]**2 + point[1]**2 == rad**2) and point[2] > (-height/2) and point[2] < (height/2):
        return True
    else:
        return False

def add_noise(point_cloud_p):
    for i, row in enumerate(point_cloud_p):
        for j, point_p in enumerate(row):
                noise_x = np.random.normal(0,1,1)
                noise_y = np.random.normal(0,1,1)
                noise_z = np.random.normal(0,1,1)
                point_p_x = point_p[0] + noise_x
                point_p_y = point_p[1] + noise_y
                point_p_z = point_p[2] + noise_z
                # point_p = [point_p_x, point_p_y, point_p_z]
                point_cloud_p[i][j][0] = point_p_x
                point_cloud_p[i][j][1] = point_p_y
                point_cloud_p[i][j][2] = point_p_z
    return point_cloud_p

def is_negative():
    rand_int = random.randint(0, 1)
    rand_sign = -1 if rand_int == 0 else 1
    return rand_sign

def generate_point_cloud_p(r, height): #cylinder point cloud
    # generate 100 points on the cylinder
    points = []
    num_points = 100

    for i in range(num_points):
        point = []
        x = random.uniform(-0.435, 0.435)
        sign_y = is_negative()
        y = (math.sqrt(r**2-x**2)) * sign_y
        z = random.uniform(-height/2, height/2)
        point = [x,y,z]
        points.append(point)
    return points

def plot(point_cloud_p):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # extract x, y, z coordinates from arr1 and arr2
    # arr1_x = [point[0] for point in arr1]
    # arr1_y = [point[1] for point in arr1]
    # arr1_z = [point[2] for point in arr1]
    point_cloud_p_x = []
    point_cloud_p_y = []
    point_cloud_p_z = []
    for point_p in point_cloud_p:
        point_cloud_p_x.append(point_p[0])
        point_cloud_p_y.append(point_p[1])
        point_cloud_p_z.append(point_p[2])

    # Define the cylinder height and radius
    h = 12
    r = .87/2

    # Define the number of points to use for the cylinder surface
    num_points = 50

    # Generate the x and y coordinates of the cylinder surface
    theta = np.linspace(0, 2*np.pi, num_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Generate the z coordinates of the cylinder surface
    z = np.linspace(0, h, num_points)
    z = np.tile(z, (num_points, 1)).T

    # plot the points
    ax.scatter(point_cloud_p_x, point_cloud_p_y, point_cloud_p_z, c='r', marker='o')

    # Plot the cylinder surface
    ax.plot_surface(x, y, z, alpha=0.5)

    # Set the axis limits and labels
    # ax.set_xlim(-r, r)
    # ax.set_ylim(-r, r)
    # ax.set_zlim(0, h)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()