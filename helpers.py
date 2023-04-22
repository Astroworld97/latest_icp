#helper functions
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from numpy.linalg import eig
import random

def create_random_quat():
    # random.seed(22)
    real_component = random.randint(1, 10)
    i_hat = random.randint(1, 10)
    j_hat = random.randint(1, 10)
    k_hat = random.randint(1, 10)
    return [real_component, i_hat, j_hat, k_hat]
    # return [-0.7240972, 0, 0, -0.6896979] #234 deg rotation about z 
    # return [0.707, 0, 0.707, 0] #90 deg rotation about y 
    # return [0.707, 0, 0.707, 0] #90 deg rotation about y 
    # return [-0.8733046, -0.4871745, 0, 0] #45 deg rotation about x

def create_random_translation_vector():
    # random.seed(14)
    i_hat = random.randint(1, 5)
    j_hat = random.randint(1, 5)
    k_hat = random.randint(1, 5)
    return [i_hat, j_hat, k_hat]

def create_random_centroid_vector():
    random.seed(19)
    i_hat = random.randint(1, 10)
    j_hat = random.randint(1, 10)
    k_hat = random.randint(1, 10)
    return [i_hat, j_hat, k_hat]

def quat_norm(quat): #returns the norm of the quaternion
    norm = np.sqrt(quat[0]**2 + quat[1]**2 + quat[2]**2 + quat[3]**2)
    return norm


def apply_initial_translation_and_rotation(point_cloud): #buggy: only applies translation in its current state
    quat = create_random_quat()
    norm = quat_norm(quat)
    quat = [quat[0]/norm, quat[1]/norm, quat[2]/norm, quat[3]/norm]
    quat_star = quat_conjugate(quat)
    p_centroid = point_cloud_centroid_p(point_cloud)
    translation_vector = create_random_translation_vector()
    dummy_q_centroid = create_random_centroid_vector()

    # left = quat_mult(quat, p_centroid)
    # right = quat_mult(left, quat_star)
    # Rp = right
    # Rp = [Rp[1], Rp[2], Rp[3]]
    # b = [dummy_q_centroid[0]-Rp[0], dummy_q_centroid[1]-Rp[1], dummy_q_centroid[2]-Rp[2]]
    for i in range(len(point_cloud)):
        point = point_cloud[i]
        left = quat_mult(quat, point) #
        right = quat_mult(left, quat_star) #
        rotation = right #
        moved_point = [rotation[1] + translation_vector[0], rotation[2] + translation_vector[1], rotation[3] + translation_vector[2]]#
        # moved_point = [point[0] + translation_vector[0], point[1] + translation_vector[1], point[2] + translation_vector[2]]#
        point_cloud[i] = moved_point
    return point_cloud
    

def closest_point_on_cylinder(point, height, rad, origin):
    #point is at arbitrary x, y, and z
    if point[2]>=(height):
        z = height
    elif point[2]<=0:
        z = 0
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
    retval = [q_centroid[0]-vect[0], q_centroid[1]-vect[1], q_centroid[2]-vect[2]]
    return retval

def createMatchDictionary(point_cloud_p, matchDict): #creates the match dictionary, which associates each point in arr1 to the point closest to in arr2
    matchDict.clear()
    for point in point_cloud_p:
        matchDict[tuple(point)] = None

def match(point_cloud_p, matchDict, q, it, q_centroid): #matches each of the points in one array to its closest corresponding point in the other array
    if it>0:
            matchDict.clear()
    for i, point_p in enumerate(point_cloud_p):
    #     if it>0:
    #         p_centroid = point_cloud_centroid_p(point_cloud_p)
    #         p_prime = calc_single_prime(point_p, p_centroid)
    #         q_star = quat_conjugate(q)
    #         left = quat_mult(q, p_prime)
    #         right = quat_mult(left, q_star)
    #         Rp = [right[1], right[2], right[3]]
    #         point_p = [Rp[0]+q_centroid[0], Rp[1]+q_centroid[1], Rp[2]+q_centroid[2]]
    #         point_cloud_p[i] = point_p
        point_q = tuple(closest_point_on_cylinder(point_p, 12, 0.87/2, [0, 0, 12/2]))
        point_p = tuple(point_p)
        matchDict[point_p] = point_q

def error(point_cloud_p, point_cloud_q, b, q, matchDict): 
    tot = 0
    q_star = quat_conjugate(q)
    for point_p in point_cloud_p:
        point_p = tuple(point_p)
        # point_q = matchDict[point_p]
        point_q = closest_point_on_cylinder(point_p, 12, .435, [0, 0, 12/2])
        Rp_i_left = quat_mult(q, point_p)
        Rp_i_right = quat_mult(Rp_i_left, q_star)
        Rp_i = Rp_i_right
        Rp_i = extract_vect_from_quat(Rp_i)
        # curr = Rp_i + b - point_q
        curr = [Rp_i[0] + b[0] - point_q[0], Rp_i[1] + b[1] - point_q[1], Rp_i[2] + b[2] - point_q[2]]
        norm_squared = (math.sqrt(curr[0]**2 + curr[1]**2 + curr[2]**2))**2
        tot+=norm_squared
    return tot

def is_point_on_cyl(point, rad, height):
    if (point[0]**2 + point[1]**2 == rad**2) and point[2] > (-height/2) and point[2] < (height/2):
        return True
    else:
        return False

def add_noise(point_cloud_p):
    for i, point_p in enumerate(point_cloud_p):
        #The first argument to numpy.random.normal is the mean of the distribution (in this case, 0), the second argument is the standard deviation (in this case, 1), and the third argument is the size of the output array (in this case, 1).
        upper = 0.3
        noise_x = np.random.normal(0,upper,1)
        noise_y = np.random.normal(0,upper,1)
        noise_z = np.random.normal(0,upper,1)
        point_p_x = point_p[0] + noise_x
        point_p_y = point_p[1] + noise_y
        point_p_z = point_p[2] + noise_z
        point_cloud_p[i][0] = point_p_x
        point_cloud_p[i][1] = point_p_y
        point_cloud_p[i][2] = point_p_z
    return point_cloud_p

def is_negative():
    rand_int = random.randint(0, 1)
    rand_sign = -1 if rand_int == 0 else 1
    return rand_sign

# def shift(point_cloud):
#     for i in range(len(point_cloud)):


def generate_point_cloud_p(r, h): #cylinder point cloud: r = radius; h = height
    # generate 100 points on the cylinder
    points = []
    num_points = 100

    for i in range(num_points):
        point = []
        x = random.uniform(-0.435, 0.435)
        sign_y = is_negative()
        y = (math.sqrt(r**2-x**2)) * sign_y
        z = random.uniform(-h/2, h/2)
        point = [x,y,z]
        points.append(point)
    return points

def plot_single_point_cloud(point_cloud_p):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    point_cloud_p_x = []
    point_cloud_p_y = []
    point_cloud_p_z = []
    for point_p in point_cloud_p:
        point_cloud_p_x.append(point_p[0])
        point_cloud_p_y.append(point_p[1])
        point_cloud_p_z.append(point_p[2])

    # plot the points
    ax.scatter(point_cloud_p_x, point_cloud_p_y, point_cloud_p_z, c='r', marker='o')

    # Set the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Set the x, y, and z limits
    # ax.set_xlim([-0.5, 0.5])
    # ax.set_ylim([-0.5, 0.5])
    # ax.set_zlim(-10, 10)

    # Show the plot
    plt.show()

def plot_two_point_clouds(point_cloud_p, point_cloud_p_moved):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    point_cloud_p_x = []
    point_cloud_p_y = []
    point_cloud_p_z = []
    for point_p in point_cloud_p:
        point_cloud_p_x.append(point_p[0])
        point_cloud_p_y.append(point_p[1])
        point_cloud_p_z.append(point_p[2])

    point_cloud_p_moved_x = []
    point_cloud_p_moved_y = []
    point_cloud_p_moved_z = []
    for point_p in point_cloud_p_moved:
        point_cloud_p_moved_x.append(point_p[0])
        point_cloud_p_moved_y.append(point_p[1])
        point_cloud_p_moved_z.append(point_p[2])

    # plot the points
    ax.scatter(point_cloud_p_x, point_cloud_p_y, point_cloud_p_z, c='r', marker='o')
    ax.scatter(point_cloud_p_moved_x, point_cloud_p_moved_y, point_cloud_p_moved_z, c='b', marker='*')

    # Set the axis limits and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()

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
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()