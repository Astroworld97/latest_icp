#for storing surplus code

#orig error fxn using eqns 14 and 15
def error(point_cloud_p, point_cloud_q, b, q, matchDict): 
    tot = 0
    q_star = quat_conjugate(q)
    for row in point_cloud_p:
        for point_p in row:
            point_p = tuple(point_p)
            point_q = matchDict[point_p]
            p_centroid = point_cloud_centroid_p(point_cloud_p)
            q_centroid = point_cloud_centroid_q(point_cloud_q)
            p_prime = calc_single_prime(point_p, p_centroid)
            q_prime = calc_single_prime(point_q, q_centroid)
            first_mult = quat_mult(q, p_prime)
            second_mult = quat_mult(first_mult, q_star)
            Rp_prime = second_mult
            Rp_prime = np.array(Rp_prime)
            q_prime = np.array(q_prime)
            vect = [Rp_prime[1], Rp_prime[2], Rp_prime[3]]
            norm_squared = (np.linalg.norm(vect - q_prime))**2
            tot+=norm_squared
    return tot

    def calc_eqn_20(quat, p_prime, q_prime):
        left = quat_mult(quat, p_prime)
        right = quat_mult(q_prime, quat)
        dot = quat_dot_product(left, right)
        return dot

def createDistDictionary(arr2): #creates the distance dictionary, where the distance to every point is initially set to infinity
    for i in range(arr1.shape[0]):
        for j in range(arr2.shape[1]):
            distDict[tuple(arr1[i][j])] = math.inf

    # for row in point_cloud_p:
    #     for point_p in row:
    #         point_p = tuple(point_p)
    #         point_q = matchDict[point_p]
    #         p_prime = calc_single_prime(point_p, p_centroid)
    #         q_prime = calc_single_prime(point_q, q_centroid)
    #         eqn_20_curr = calc_eqn_20(q, p_prime, q_prime)
    #         eqn_20_total+=eqn_20_curr


    # eqn_20_total = 0 #aka R * p_prime

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
    # outputArr = np.round(outputArr).astype(int)

    return outputArr

    # def apply_initial_translation_and_rotation(inputArr):
    # # Define rotation matrix for 5 degree rotation around y-axis
    # inputArr = np.array(inputArr)
    # r = Rotation.from_euler('y', 20, degrees=True)
    # rot_matrix = r.as_matrix()

    # # Define translation vector
    # translation = np.array([1, 1, 1])

    # # Apply rotation and translation
    # outputArr = inputArr.dot(rot_matrix.T) + translation

    # # Round the values in the output matrix to integers
    # outputArr = np.round(outputArr).astype(int)

    # return outputArr.tolist()

##     functions
# def dot_product(p, q): #inputs are quaternions. Remember the 0th element is irrelevant here.
#     return p[1]*q[1] + p[2]*q[2] + p[3]*q[3] #returns a scalar

# def cross_product(p, q): #inputs are quaternions. Remember the 0th element is irrelevant here.
#     i_hat = p[2]*q[3] - p[3]*q[2]
#     j_hat = p[3]*q[1] - p[1]*q[3]
#     k_hat = p[1]*q[2] - p[2]*q[1]
#     return [i_hat, j_hat, k_hat] #returns a vector

# def quat_mult(p,q): #quaternion multiplication. Inputs p and q are quaternions.
#     first_term = p[0]*q[0]
#     dot_prod = dot_product(p, q)
#     first_term = first_term - dot_prod
#     cross_prod = cross_product(p, q)
#     i_hat = p[0]*q[1] + q[0]*p[1] + cross_prod[0] 
#     j_hat = p[0]*q[2] + q[0]*p[2] + cross_prod[1]
#     k_hat = p[0]*q[3] + q[0]*p[3] + cross_prod[2]
#     return [first_term, i_hat, j_hat, k_hat]

# def conjugate(quat): #inputs are quaternions. Calculates the conjugate of a quaternion.
#     first_term = quat[0]
#     i_hat = -quat[1]
#     j_hat = -quat[2]
#     k_hat = -quat[3]
#     return [first_term, i_hat, j_hat, k_hat]

# def point_cloud_centroid(point_cloud): #computes centroid/mean/center of mass of point cloud
#     arr = point_cloud
#     sum_x = 0
#     sum_y = 0
#     sum_z = 0

#     if len(point_cloud)==0:
#         return [0,0,0]

#     for point in arr:
#         sum_x += point[0]
#         sum_y += point[1]
#         sum_z += point[2]
    
#     avg_x = round(sum_x/len(arr), 4)
#     avg_y = round(sum_y/len(arr), 4)
#     avg_z = round(sum_z/len(arr), 4)

#     return [avg_x, avg_y, avg_z] 

# def calc_single_prime(point, centroid):
#     prime_x = point[0] - centroid[0]
#     prime_y = point[1] - centroid[1] 
#     prime_z = point[2] - centroid[2]
#     return [prime_x, prime_y, prime_z]

# def create_prime_matrix_p(prime_vector):
#     values = [0, -1*prime_vector[1], -1*prime_vector[2], -1*prime_vector[3], -1*prime_vector[1], 0, -1*prime_vector[3], -1*prime_vector[2], -1*prime_vector[2], -1*prime_vector[3], 0, -1*prime_vector[1], -1*prime_vector[3], -1*prime_vector[2], -1*prime_vector[1], 0]
#     return np.array(values).reshape((4, 4))

# def create_prime_matrix_q(prime_vector):
#     values = [0, -1*prime_vector[1], -1*prime_vector[2], -1*prime_vector[3], prime_vector[1], 0, -1*prime_vector[3], prime_vector[2], prime_vector[2], prime_vector[3], 0, -1*prime_vector[1], prime_vector[3], -1*prime_vector[2], prime_vector[1], 0]
#     return np.array(values).reshape((4, 4))

# def apply_initial_translation_and_rotation(inputArr):
    
#     return 

# def create_random_quat():
#     random.seed(22)
#     real_component = random.randint(1, 10)
#     i_hat = random.randint(1, 10)
#     j_hat = random.randint(1, 10)
#     k_hat = random.randint(1, 10)
#     return [real_component, i_hat, j_hat, k_hat]

# def create_random_translation_vector():
#     random.seed(14)
#     i_hat = random.randint(1, 10)
#     j_hat = random.randint(1, 10)
#     k_hat = random.randint(1, 10)
#     return [i_hat, j_hat, k_hat]

# def apply_initial_translation_and_rotation(point_cloud_p):
#     quat = create_random_quat()
#     quat_star = quat_conjugate(quat)
#     p_centroid = point_cloud_centroid_p(point_cloud_p)
#     translation_vector = create_random_translation_vector()
#     dummy_q_centroid = create_random_centroid_vector()
#     left = quat_mult(quat, p_centroid)
#     right = quat_mult(left, quat_star)
#     Rp = right
#     b = dummy_q_centroid - Rp
#     for i in len(point_cloud_p):
#         point = point_cloud_p[i]
#         moved_point = point + b
#         point_cloud_p[i] = moved_point
#     return point_cloud_p

# def plot_single_point_cloud(point_cloud_p):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     point_cloud_p_x = []
#     point_cloud_p_y = []
#     point_cloud_p_z = []
#     for point_p in point_cloud_p:
#         point_cloud_p_x.append(point_p[0])
#         point_cloud_p_y.append(point_p[1])
#         point_cloud_p_z.append(point_p[2])
#     p_centroid = point_cloud_centroid(point_cloud_p)
#     # plot the points
#     ax.scatter(point_cloud_p_x, point_cloud_p_y, point_cloud_p_z, c='r', marker='o')
#     # ax.scatter(p_centroid[0], p_centroid[1], p_centroid[2], c='b', marker='o') #uncomment to plot the centroid
#     # Set the axis limits and labels
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     # Show the plot
#     plt.show()


# p = [3, 1, -2, 1]
# q = [2, -1, 2, 3]

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
# # print(cross_product(p, q))
# print(quat_mult(p,q))
# print(conjugate(p))
# print(create_random_quat())
# print(create_random_translation_vector())

# point_p = point_p.astype(float)
left_curr = quat_mult(q, point_p)
right_curr = quat_mult(point_p, q_star)
point_p = [right_curr[1], right_curr[2], right_curr[3]]
point_p = [point_p[0]+b[0], point_p[1]+b[1], point_p[2]+b[2]]
point_cloud_p[i] = point_p

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

# point_cloud_q = np.array(list((matchDict.values())))
# point_cloud_q = point_cloud_q.reshape(point_cloud_p.shape)

# return
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

    # p_centroid = point_cloud_centroid_p(point_cloud_p)
    # p_prime = calc_single_prime(point_p, p_centroid)
    # point_q = matchDict[point_p]
    # q_prime = calc_single_prime(point_q, q_centroid)
    # left_curr = quat_mult(q, p_prime)
    # right_curr = quat_mult(left_curr, q_star)
    # Rp = [right_curr[1],right_curr[2],right_curr[3]]
    # rot = Rp[]
    # point_p = [rot[0]+b[0], rot[2]+b[1], rot[3]+b[2]]

    # if abs(p_centroid[0] - q_centroid[0]) < 0.001 and abs(p_centroid[1] - q_centroid[1]) < 0.001 and abs(p_centroid[2] - q_centroid[2]) < 0.001:
        #     point_p = [Rp[0], Rp[1], Rp[2]]
        # else:

    # dummy_q_centroid = create_random_centroid_vector()

    # left = quat_mult(quat, p_centroid)
    # right = quat_mult(left, quat_star)
    # Rp = right
    # Rp = [Rp[1], Rp[2], Rp[3]]
    # b = [dummy_q_centroid[0]-Rp[0], dummy_q_centroid[1]-Rp[1], dummy_q_centroid[2]-Rp[2]]

    # if abs(p_centroid[0] - q_centroid[0]) < 0.001 and abs(p_centroid[1] - q_centroid[1]) < 0.001 and abs(p_centroid[2] - q_centroid[2]) < 0.001:
        #     point_p = [Rp[0], Rp[1], Rp[2]]
        # else:

    # for i, point_p in enumerate(point_cloud_p):
#     point_p = tuple(point_p)
#     point_q = closest_point_on_cylinder(point_p, 12, .435, [0, 0, 12/2])
#     point_p = point_q
#     point_cloud_p[i] = point_p

# plot_single_point_cloud(point_cloud_p)
# plot_single_point_cloud(point_cloud_q)