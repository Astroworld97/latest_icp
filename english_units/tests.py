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