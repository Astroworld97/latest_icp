import numpy as np
def calc_single_M(P_i, Q_i):
    return np.matmul(P_i.T,Q_i)

P_0 = np.array([[0., 3., 3., 3.],
                [-3., 0., -3., 3.],
                [-3., 3., 0., -3.],
                [-3., -3., 3., 0.]])

Q_0 = np.array([[ 0., 4., 3., 2.],
                [-4., 0., 2., -3.],
                [-3., -2., 0., 4.],
                [-2., 3., -4., 0.]])

print(calc_single_M(P_0, Q_0))