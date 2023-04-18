def calc_single_prime(point, centroid):
    prime_x = round(point[0] - centroid[0], 4)
    prime_y = round(point[1] - centroid[1], 4)
    prime_z = round(point[2] - centroid[2], 4)
    return [prime_x, prime_y, prime_z]

# Test case 1
point = [1, 2, 3]
centroid = [2, 2, 2]
assert calc_single_prime(point, centroid) == [-1, 0, 1]

# Test case 2
point = [-3, 0, 1]
centroid = [-1, -1, -1]
assert calc_single_prime(point, centroid) == [-2, 1, 2]

# Test case 3
point = [4.2, 5.8, 7.9]
centroid = [4, 6, 8]
# print(calc_single_prime(point, centroid))
assert calc_single_prime(point, centroid) == [round(0.1999999999999993,4), round(-0.1999999999999993,4), round(-0.10000000000000142,4)]

#Test case 4
point = [1, 2, 3]
centroid = [4.0, 5.0, 6.0]
print(calc_single_prime(point, centroid))
assert calc_single_prime(point, centroid) == [-3.0, -3.0, -3.0]

print("all tests pass")
