def quat_dot_product(p, q): #inputs are quaternions. Remember the 0th element is irrelevant here.
    return p[1]*q[1] + p[2]*q[2] + p[3]*q[3] #returns a scalar

def quat_cross_product(p, q): #inputs are quaternions. Remember the 0th element is irrelevant here.
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

def quat_mult(p,q): #quaternion multiplication. Inputs p and q are quaternions.
    first_term = p[0]*q[0]
    dot_prod = quat_dot_product(p, q)
    first_term = first_term - dot_prod
    cross_prod = quat_cross_product(p, q)
    i_hat = p[0]*q[1] + q[0]*p[1] + cross_prod[0] 
    j_hat = p[0]*q[2] + q[0]*p[2] + cross_prod[1]
    k_hat = p[0]*q[3] + q[0]*p[3] + cross_prod[2]
    return [first_term, i_hat, j_hat, k_hat]

def calc_eqn_20(quat, p_prime, q_prime):
    left = quat_mult(quat, p_prime)
    right = quat_mult(q_prime, quat)
    dot = quat_dot_product(left, right)
    return dot

p_prime = [3,1,-2,1]
q_prime = [2,-1,2,3]
quat = [0.06, -0.66, -0.56, -0.48]
print(quat_mult(quat, p_prime))
print(quat_mult(q_prime, quat))
print(calc_eqn_20(quat, p_prime, q_prime))