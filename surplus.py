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