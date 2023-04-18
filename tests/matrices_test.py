import numpy as np

p_prime = [-3.0, -3.0, -3.0]
q_prime = [-4.0, -3.0, -2.0]

def create_prime_matrix_p(prime_vector):
    prime_vector = [0, prime_vector[0], prime_vector[1], prime_vector[2]]
    values = [0, -1*prime_vector[1], -1*prime_vector[2], -1*prime_vector[3], prime_vector[1], 0, prime_vector[3], -1*prime_vector[2], prime_vector[2], -1*prime_vector[3], 0, prime_vector[1], prime_vector[3], prime_vector[2], -1*prime_vector[1], 0]
    return np.array(values).reshape((4, 4))

def create_prime_matrix_q(prime_vector):
    prime_vector = [0, prime_vector[0], prime_vector[1], prime_vector[2]]
    values = [0, -1*prime_vector[1], -1*prime_vector[2], -1*prime_vector[3], prime_vector[1], 0, -1*prime_vector[3], prime_vector[2], prime_vector[2], prime_vector[3], 0, -1*prime_vector[1], prime_vector[3], -1*prime_vector[2], prime_vector[1], 0]
    return np.array(values).reshape((4, 4))

print(create_prime_matrix_p(p_prime))
print(create_prime_matrix_q(q_prime))