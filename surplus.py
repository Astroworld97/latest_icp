# # def plot_single_point_cloud(point_cloud_p):
# #     fig = plt.figure()
# #     ax = fig.add_subplot(111, projection='3d')

# #     point_cloud_p_x = []
# #     point_cloud_p_y = []
# #     point_cloud_p_z = []
# #     for point_p in point_cloud_p:
# #         point_cloud_p_x.append(point_p[0])
# #         point_cloud_p_y.append(point_p[1])
# #         point_cloud_p_z.append(point_p[2])

# #     # plot the points
# #     ax.scatter(point_cloud_p_x, point_cloud_p_y, point_cloud_p_z, c='r', marker='o')

# #     # Set the axis labels
# #     ax.set_xlabel('X')
# #     ax.set_ylabel('Y')
# #     ax.set_zlabel('Z')
# #     # Set the x, y, and z limits
# #     # ax.set_xlim([-0.5, 0.5])
# #     # ax.set_ylim([-0.5, 0.5])
# #     # ax.set_zlim(-10, 10)

# #     # Show the plot
# #     plt.show()

# def plot_two_point_clouds(point_cloud_p, point_cloud_p_moved):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     point_cloud_p_x = []
#     point_cloud_p_y = []
#     point_cloud_p_z = []
#     for point_p in point_cloud_p:
#         point_cloud_p_x.append(point_p[0])
#         point_cloud_p_y.append(point_p[1])
#         point_cloud_p_z.append(point_p[2])

#     point_cloud_p_moved_x = []
#     point_cloud_p_moved_y = []
#     point_cloud_p_moved_z = []
#     for point_p in point_cloud_p_moved:
#         point_cloud_p_moved_x.append(point_p[0])
#         point_cloud_p_moved_y.append(point_p[1])
#         point_cloud_p_moved_z.append(point_p[2])

#     # plot the points
#     ax.scatter(point_cloud_p_x, point_cloud_p_y, point_cloud_p_z, c='r', marker='o')
#     ax.scatter(point_cloud_p_moved_x, point_cloud_p_moved_y, point_cloud_p_moved_z, c='b', marker='*')

#     # Set the axis limits and labels
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     # Show the plot
#     plt.show()


### from inside match ###
#     if it>0:
    #         p_centroid = point_cloud_centroid_p(point_cloud_p)
    #         p_prime = calc_single_prime(point_p, p_centroid)
    #         q_star = quat_conjugate(q)
    #         left = quat_mult(q, p_prime)
    #         right = quat_mult(left, q_star)
    #         Rp = [right[1], right[2], right[3]]
    #         point_p = [Rp[0]+q_centroid[0], Rp[1]+q_centroid[1], Rp[2]+q_centroid[2]]
    #         point_cloud_p[i] = point_p

    # def within_color_range(point_cloud_p, point_cloud_q, colorDictP, colorDictQ):
    # if abs(pt2[3]-pt1[3])<=10 and abs(pt2[4]-pt1[4])<=10 and abs(pt2[5]-pt1[5])<=10:
    #     return True
    # else:
    #     return False 

#     def is_negative():
#     rand_int = random.randint(0, 1)
#     rand_sign = -1 if rand_int == 0 else 1
#     return rand_sign

# # def shift(point_cloud):
# #     for i in range(len(point_cloud)):

# def is_point_on_cyl(point, rad, height):
#     if (point[0]**2 + point[1]**2 == rad**2) and point[2] > (-height/2) and point[2] < (height/2):
#         return True
#     else:
#         return False

    # left = round((x**2 + y**2), 4)
    # right = round((rad**2), 4)
    # assert left == right, print([x,y,z])

# def closest_point_on_cylinder(point, height, rad, origin, colorDictP, modelHuedRange):
#     #point is at arbitrary x, y, and z
#     x = point[0]/math.sqrt(point[0]**2 + point[1]**2)
#     y = point[1]/math.sqrt(point[0]**2 + point[1]**2)
#     x = x * rad
#     y = y * rad
    
#     if point[2]>=(height):
#         z = height
#         cylpoint = [x,y,z]
#         point_color = colorDictP[tuple(point)]
#         cylpoint_color = get_cylpoint_color(cylpoint, modelHuedRange)
#         if color_match(point_color, cylpoint_color):
#             return cylpoint
#         else:
#             for i in range(0, 1201, 1):
#                 to_subtract = i / 100.0
#                 z -= to_subtract
#                 cylpoint = [x,y,z]
#                 cylpoint_color = get_cylpoint_color(cylpoint, modelHuedRange)
#                 if color_match(point_color, cylpoint_color):
#                     return cylpoint
#     elif point[2]<=0:
#         z = 0
#         cylpoint = [x,y,z]
#         point_color = colorDictP[tuple(point)]
#         cylpoint_color = get_cylpoint_color(cylpoint, modelHuedRange)
#         if color_match(point_color, cylpoint_color):
#             return cylpoint
#         else:
#             for i in range(0, 1201, 1):
#                 to_add = i / 100.0
#                 z += to_add
#                 cylpoint = [x,y,z]
#                 cylpoint_color = get_cylpoint_color(cylpoint, modelHuedRange)
#                 if color_match(point_color, cylpoint_color):
#                     return cylpoint
#     else:
#         z = point[2]
#         cylpoint = [x,y,z]
#         point_color = colorDictP[tuple(point)]
#         cylpoint_color = get_cylpoint_color(cylpoint, modelHuedRange)
#         if color_match(point_color, cylpoint_color):
#             return cylpoint
#         else:
#             height_check = z
#             while(height_check!=height):
#                 for i in range(0, 1201, 1):
#                     to_add = i / 100.0
#                     z += to_add
#                     cylpoint = [x,y,z]
#                     cylpoint_color = get_cylpoint_color(cylpoint, modelHuedRange)
#                     if color_match(point_color, cylpoint_color):
#                         return cylpoint
#                     z -= to_add
#                     height_check += 0.01
#             for i in range(0, 1201, 1):
#                 to_subtract = i / 100.0
#                 z -= to_subtract
#                 cylpoint = [x,y,z]
#                 cylpoint_color = get_cylpoint_color(cylpoint, modelHuedRange)
#                 if color_match(point_color, cylpoint_color):
#                     return cylpoint
#                 z += to_subtract
#     return [x,y,z]