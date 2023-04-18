# Test point cloud
point_cloud = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Expected centroid
expected_centroid = [4, 5, 6]

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

# Compute centroid using the function
computed_centroid = point_cloud_centroid(point_cloud)

# Check if the computed centroid is equal to the expected centroid
if computed_centroid == expected_centroid:
    print("Test passed!")
else:
    print("Test failed.")

# Test case 1 - Empty point cloud
assert point_cloud_centroid([]) == [0, 0, 0]

# Test case 2 - Single point cloud
assert point_cloud_centroid([(1,2,3)]) == [1, 2, 3]

# Test case 3 - Multiple point cloud with integer coordinates
assert point_cloud_centroid([(1,2,3), (4,5,6), (7,8,9)]) == [4, 5, 6]

# Test case 4 - Multiple point cloud with float coordinates
assert point_cloud_centroid([(1.5,2.5,3.5), (4.2,5.2,6.2), (7.8,8.8,9.8)]) == [4.5, 5.5, 6.5]

# Test case 5 - Multiple point cloud with negative coordinates
assert point_cloud_centroid([(-1,-2,-3), (-4,-5,-6), (-7,-8,-9)]) == [-4, -5, -6]

# Test case 6 - Multiple point cloud with mixed coordinates
print(point_cloud_centroid([(1,2.5,-3.2), (-4.2,5,-6), (7,8.8,9.9)]))
assert point_cloud_centroid([(1,2.5,-3.2), (-4.2,5,-6), (7,8.8,9.9)]) == [round(1.266666667,4), round(5.433333333,4), round(0.2333333333,4)]

print("all tests pass!")