import helpers_english_units
from helpers_english_units import *

# quaternions
p = [
    [4, 8, 16, 24],
    [3, 3, 3, 3],
    [8, 3, 9, 3],
    [2, 4, 3, 1]
 ]
q = [
    [1, 1, 1, 1],
    [2, 0, 2, 1],
    [9, 9, -9, -9],
    [-7, -2, 3, 2]
 ]

# what the results should be from each method
results = [
    { # [4, 8, 16, 24], [1, 1, 1, 1]
        "vect_from_quat": [[8,16,24], [1,1,1]],
        "vect_to_quat": [[0,8,16,24], [0,1,1,1]],
        "conjugate": [[4,-8,-16,-24], [1,-1,-1,-1]],
        "cross_product": [-8,16,-8],
        "dot_product": 48,
        "norm": [30.199, 2],
        "mult": [-44,4,36,20],
    },
    { # [3, 3, 3, 3], [2, 0, 2, 1]
        "vect_from_quat": [[3,3,3], [0,2,1]],
        "vect_to_quat": [[0,3,3,3], [0,0,2,1]],
        "conjugate": [[3,-3,-3,-3], [2,0,-2,-1]],
        "cross_product": [-3,-3,6],
        "dot_product": 9,
        "norm": [6, 3],
        "mult": [-3, 3, 9, 15],
    },
    { # [8, 3, 9, 3], [9, 9, -9, -9]
        "vect_from_quat": [[3,9,3], [9,-9,-9]],
        "vect_to_quat": [[0,3,9,3], [0,9,-9,-9]],
        "conjugate": [[8,-3,-9,-3], [9,-9,9,9]],
        "cross_product": [-54,54,-108],
        "dot_product": -81,
        "norm": [12.767, 18],
        "mult": [153, 45, 63, -153],
    },
    { # [2, 4, 3, 1], [-7, -2, 3, 2]
        "vect_from_quat": [[4,3,1], [-2,3,2]],
        "vect_to_quat": [[0,4,3,1], [0,-2,3,2]],
        "conjugate": [[2,-4,-3,-1], [-7,2,-3,-2]],
        "cross_product": [3,-10,18],
        "dot_product": 3,
        "norm": [5.477, 8.124],
        "mult": [-17,-29,-25, 15],
    },
]

for i in range(0,len(p)-1):
    # vect_from_quat
    pv = extract_vect_from_quat(p[i])
    qv = extract_vect_from_quat(q[i])
    if pv != results[i]["vect_from_quat"][0]:
        print("vect_from_quat: " + str(pv) + " != [" + str(results[i]["vect_from_quat"][0]))
    if qv != results[i]["vect_from_quat"][1]:
        print("vect_from_quat: " + str(qv) + " != " + str(results[i]["vect_from_quat"][1]))

    # vect_to_quat
    actual = vect_to_quat(pv)
    expected = results[i]["vect_to_quat"][0]
    if actual != expected:
        print("vect_to_quat: " + str(actual) + " != " + str(expected))
    actual = vect_to_quat(qv)
    expected = results[i]["vect_to_quat"][1]
    if actual != expected:
        print("vect_to_quat: " + str(actual) + " != " + str(expected))

    # conjugate
    actual = quat_conjugate(p[i])
    expected = results[i]["conjugate"][0]
    if actual != expected:
        print("conjugate: " + str(actual) + " != " + str(expected))
    actual = quat_conjugate(q[i])
    expected = results[i]["conjugate"][1]
    if actual != expected:
        print("conjugate: " + str(actual) + " != " + str(expected))

    # cross_product
    actual = quat_cross_product(p[i], q[i])
    expected = results[i]["cross_product"]
    if actual != expected:
        print("cross_product: " + str(actual) + " != " + str(expected))

    # dot_product
    actual = quat_dot_product(p[i], q[i])
    expected = results[i]["dot_product"]
    if actual != expected:
        print("dot_product: " + str(actual) + " != " + str(expected))

    # norm
    actual = quat_norm(p[i])
    expected = results[i]["norm"][0]
    if actual != expected:
        print("norm: " + str(actual) + " != " + str(expected))
    actual = quat_norm(q[i])
    expected = results[i]["norm"][1]
    if actual != expected:
        print("norm: " + str(actual) + " != " + str(expected))

    # mult
    actual = quat_mult(p[i], q[i])
    expected = results[i]["mult"]
    if actual != expected:
        print("mult: " + str(actual) + " != " + str(expected))

height = 12
rad = .435 #.87/2
origin = [0,0,height/2]

modelBlueRange = [height - 2, height] #section of analytical model with blue color tape
modelRedRange = [0.00, 2.00] #section of analytical model with red color tape

colorDictP = {
    (.435, 0, 0) : (0, 1, 1), #red at (.435, 0, 0)
    (.435, 0, 12) : (240/360, 1, 1), #blue at (.435, 0, 12)
    (100,100,100) : (0,1,1), #red at (100,100,100)
    (-10, 13, -2) : (240/360, 1, 1) #blue at (-10, 13, -2)
}

print(closest_point_on_cylinder((.435, 0, 0), height, rad, origin, colorDictP, modelBlueRange, modelRedRange)) #expected: (.435, 0, 0)
print(closest_point_on_cylinder((.435, 0, 12), height, rad, origin, colorDictP, modelBlueRange, modelRedRange)) #expected: (.435, 0, 12)
print(closest_point_on_cylinder((100,100,100), height, rad, origin, colorDictP, modelBlueRange, modelRedRange)) #expected: (0.308,0.308,2)
print(closest_point_on_cylinder((-10, 13, -2), height, rad, origin, colorDictP, modelBlueRange, modelRedRange)) #expected: (-0.265,0.345,10)

print(color_match((240/360, 1, 1), (0,1,1))) #expected: False
print(color_match((0,1,1), (0,1,1))) #expected: True
print(color_match((240/360, 1, 1), (240/360, 1, 1))) #expected: True

centroid = [0,0,6]

print(calc_single_prime((10,13,30), centroid)) #expected: (10,13,24)
print(calc_single_prime((99,99,99), centroid)) #expected: (99,99,93)
print(calc_single_prime((1,-1,6), centroid)) #expected: (1,-1,0)

# print(create_prime_matrix_p((10,13,24)))
# print(create_prime_matrix_p((99,99,93)))
# print(create_prime_matrix_p((1,-1,0)))

# print(create_prime_matrix_q((10,13,24)))
# print(create_prime_matrix_q((99,99,93)))
# print(create_prime_matrix_q((1,-1,0)))

# P = create_prime_matrix_p((10,13,24))
# Q = create_prime_matrix_q((99,99,93))
# print(calc_single_M(P, Q))

q_centroid = [0,0,6]
p_centroid = [10,10,10]
quat = [-3, 3, 9, 15]
quat_star = quat_conjugate(quat)
print(calc_b(q_centroid, p_centroid, quat, quat_star))

# M = create_prime_matrix_p((10,13,24))
# print(calc_quat(M))

