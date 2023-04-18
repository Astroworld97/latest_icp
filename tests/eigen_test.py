from numpy.linalg import eig
import numpy as np

M_0 = [[ 27., -3., 6., -3.],
        [-3., -3., 21., 18.],
        [ 6., 21., -9., 15.],
        [-3., 18., 15., -15.]]

def calc_q(M):
    eigenvalues,eigenvectors=eig(M)
    max_eigval_idx = np.argmax(eigenvalues)
    max_eigvec = eigenvectors[:, max_eigval_idx]
    return max_eigvec

q = calc_q(M_0)
print(q)