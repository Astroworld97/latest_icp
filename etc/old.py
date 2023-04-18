import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

#Note: XYZ values are indices 0,1,2; RGB values are 3,4,5 

def add_RGB_values(arrA):
    np.random.seed(123)
    arrB = np.random.randint(low=0, high=256, size=(4,4,3)) #sets colors in between 0 and 255
    arrC = np.concatenate((arrA, arrB), axis=2)
    return arrC.astype(np.uint8)

def apply_initial_translation_and_rotation(inputArr):
    # Define rotation matrix for 5 degree rotation around y-axis
    r = Rotation.from_euler('y', 20, degrees=True)
    rot_matrix = r.as_matrix()

    # Define translation vector
    translation = np.array([1, 1, 1])

    # Apply rotation and translation
    outputArr = inputArr.dot(rot_matrix.T) + translation

    # Round the values in the output matrix to integers
    outputArr = np.round(outputArr).astype(int)

    return outputArr

def createDistDictionary(arr2): #creates the distance dictionary, where the distance to every point is initially set to infinity
    for i in range(arr2.shape[0]):
        for j in range(arr2.shape[1]):
            distDict[tuple(arr2[i][j])] = math.inf

def createMatchDictionary(arr2): #creates the match dictionary, which associates each point in arr1 to the point closest to in arr2
    for i in range(arr2.shape[0]):
        for j in range(arr2.shape[1]):
            matchDict[tuple(arr2[i][j])] = None

def within_color_range(pt1,pt2):
    if abs(pt2[3]-pt1[3])<=10 and abs(pt2[4]-pt1[4])<=10 and abs(pt2[5]-pt1[5])<=10:
        return True
    else:
        return False

def distance(pt1, pt2): #calculates the distance between point 1 (pt1) and point 2 (pt2)
    x = (pt2[0]-pt1[0])**2
    y = (pt2[1]-pt1[1])**2
    z = (pt2[2]-pt1[2])**2
    retval = math.sqrt(x+y+z)
    return retval

def match(arr1, arr2): #matches each of the points in one array to its closest corresponding point in the other array
    list1 = np.array(arr1)
    list2 = np.array(arr2)
    for i in range(list2.shape[0]):
        currPoint = tuple(arr2[i])
        for j in range(list1.shape[0]):
            # if(within_color_range(arr1[i], arr2[j])):
            currDist = distance(arr2[i], arr1[j])
            if currDist<distDict[currPoint]:
                distDict[currPoint] = currDist
                matchDict[currPoint] = arr1[j]

def error(arr1, arr2): #error metric (MSE)
    sum=0
    for i in range(len(arr1)):
        x_sum = (arr1[i][0]-arr2[i][0])**2
        y_sum = (arr1[i][1]-arr2[i][1])**2
        z_sum = (arr1[i][2]-arr2[i][2])**2
        sum+=x_sum+y_sum+z_sum
    sum/=len(arr1)
    return sum

def point_cloud_mean(point_cloud): #computes centroid/mean/center of mass of point cloud
    arr = point_cloud
    sum_x = 0
    sum_y = 0
    sum_z = 0

    for point in arr:
        sum_x += point[0]
        sum_y += point[1]
        sum_z += point[2]
    
    avg_x = sum_x/len(arr)
    avg_y = sum_y/len(arr)
    avg_z = sum_z/len(arr)

    return (avg_x, avg_y, avg_z)

def point_cloud_mean_array(point_cloud):
    retval = []
    toAdd = point_cloud_mean(point_cloud)
    for i in range(len(point_cloud)):
        retval.append(toAdd)
    return retval

def prime_point_cloud(point_cloud, point_cloud_mean): #calculates difference between a point and the point cloud mean
    # retval = []
    # point_cloud_mean = np.array(point_cloud_mean)
    # for p in point_cloud:
    #     p = np.array(p)
    #     toAdd = p-point_cloud_mean
    #     retval.append(list(toAdd))
    # return retval
    retval = []
    for p in point_cloud:
        p = np.asarray(p)
        point_cloud_mean = np.asarray(point_cloud_mean)
        toAdd = p-point_cloud_mean
        toAdd = tuple(toAdd)
        retval.append(toAdd)     
    return retval

def point_cloud_to_tuple_list(point_cloud):
    return [tuple(point) for row in point_cloud for point in row]

def tuple_list_to_list_of_lists(l):
    retval = []
    for i in l:
        toAdd = list(i)
        retval.append(toAdd)
    return retval

def create_prime_vectors(arr1, arr2):
    p_mean = point_cloud_mean(arr1)
    q_mean = point_cloud_mean(arr2)
    p_mean = np.array(p_mean)
    q_mean = np.array(q_mean)
    p_prime = prime_point_cloud(arr1, p_mean)
    q_prime = prime_point_cloud(arr2, q_mean)
    return p_prime, q_prime

def move_centroid_and_point_cloud(arr1, arr2): #moves centroid of the p point cloud to the same centroid of the q point cloud
    p_cloud = arr1
    q_cloud = arr2
    p_centroid = point_cloud_mean(p_cloud)
    q_centroid = point_cloud_mean(q_cloud)
    translation = q_centroid - p_centroid
    new_p_centroid = p_centroid + translation
    new_p_cloud = p_cloud + translation
    return new_p_centroid, new_p_cloud

def dot_product(p, q): #inputs are quaternions. Remember the 0th element is irrelevant here.
    return p[1]*q[1] + p[2]*q[2] + p[3]*q[3] #returns a scalar

def cross_product(p, q): #inputs are quaternions. Remember the 0th element is irrelevant here.
    i_hat = p[2]*q[3] - p[3]*q[2]
    j_hat = p[3]*q[1] - p[1]*q[3]
    k_hat = p[1]*q[2] - p[2]*q[1]
    return [i_hat, j_hat, k_hat] #returns a vector

def conjugate(quat): #inputs are quaternions. Calculates the conjugate of a quaternion.
    first_term = quat[0]
    i_hat = -quat[1]
    j_hat = -quat[2]
    k_hat = -quat[3]
    return [first_term, i_hat, j_hat, k_hat] 

def quat_mult(p,q): #quaternion multiplication. Inputs p and q are quaternions.
    first_term = p[0]*q[0]
    dot_prod = dot_product(p, q)
    first_term = first_term - dot_prod
    cross_prod = cross_product(p, q)
    i_hat = p[0]*q[1] + q[0]*p[1] + cross_prod[0] 
    j_hat = p[0]*q[2] + q[0]*p[2] + cross_prod[1]
    k_hat = p[0]*q[3] + q[0]*p[3] + cross_prod[2]
    return [first_term, i_hat, j_hat, k_hat]

def b(q_mean, R, p_mean):
    return q_mean - R*p_mean

def plot(arr1,arr2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # extract x, y, z coordinates from arr1 and arr2
    arr1_x = [point[0] for point in arr1]
    arr1_y = [point[1] for point in arr1]
    arr1_z = [point[2] for point in arr1]

    arr2_x = [point[0] for point in arr2]
    arr2_y = [point[1] for point in arr2]
    arr2_z = [point[2] for point in arr2]

    # plot the points
    ax.scatter(arr1_x, arr1_y, arr1_z, c='r', marker='o')
    ax.scatter(arr2_x, arr2_y, arr2_z, c='b', marker='^')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


#MAIN

#section 1: define constants and data structures
maxIterations = 10000
tolerance = 0.001
distDict = {}
matchDict = {}
# q0 = np.array([1,0,0,0,0,0,0]).T
# R = build_R(q0)

#section 2: define arrays (aka point clouds) and initialize dictionaries
arr1 = np.array([
    [[100, 50, 200], [150, 200, 100], [50, 100, 150], [200, 150, 50]],
    [[200, 100, 150], [50, 150, 200], [150, 50, 100], [100, 200, 50]],
    [[100, 150, 50], [200, 50, 100], [50, 200, 150], [150, 100, 200]],
    [[150, 100, 200], [100, 150, 50], [200, 50, 150], [50, 200, 100]]
])

arr2 = arr1.copy()
# arr2 = add_noise(arr1)
arr2 = apply_initial_translation_and_rotation(arr2)
# arr1 = add_RGB_values(arr1)
# arr2 = add_RGB_values(arr2)
createDistDictionary(arr2)
createMatchDictionary(arr2)

arr1 = point_cloud_to_tuple_list(arr1)
arr2 = point_cloud_to_tuple_list(arr2)


###start test area###
p = arr1
q = arr2
p_mean = point_cloud_mean(p)
q_mean = point_cloud_mean(q)
test = prime_point_cloud(p, p_mean)
print(test)
print(prime_point_cloud(q, q_mean))
###end test area###

# # #section 3: iterate
# for i in range(maxIterations):
#     plot(arr1, arr2)
#     match(arr1, arr2) #fill the matchDict and the distDict with the current matches
#     err = error(arr1, arr2)
#     if err<tolerance:
#         break

#     sigma = create_cross_cov_matrix(arr1, arr2) #cross-covariance matrix

    
    
#     p_mean = np.array(point_cloud_mean(arr1))
#     p_mean = p_mean.reshape((1,3))
#     x_mean = np.array(point_cloud_mean(arr2))
#     x_mean = x_mean.reshape((1,3))
#     # x_mean_transpose = x_mean.T
#     qt = build_qt(x_mean, R, p_mean)
#     arr1 = arr1+qt


#     # H = np.array(H)
#     # U, Sigma, Vt = np.linalg.svd(H)
#     # R = np.dot(U, Vt)
#     # # R1 = U @ Vt #note: R==R1
#     # arr1_mean = np.array(point_cloud_mean_array(arr1))
#     # arr2_mean = np.array(point_cloud_mean_array(arr2))
#     # t = arr1_mean - np.dot(R, arr2_mean)
#     # p_prime = prime_point_cloud(arr1, arr1_mean)
#     # arr1 = np.dot(R, p_prime) + arr2_mean
#     err = error(arr1, arr2)
#     print(err)

# plot(arr1, arr2)