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
