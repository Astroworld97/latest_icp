import numpy as np
from numpy.linalg import eig

a = np.array([[3, 5], 
              [-1, -3]])
eigenvalues,eigenvectors=eig(a)
print('Eigenvalues: ', eigenvalues)
print('Eigenvectors: ', eigenvectors)
print(max(eigenvalues))
max_eigval_idx = np.argmax(eigenvalues)
print(max_eigval_idx)
max_eigvec = eigenvectors[:, max_eigval_idx]
print(max_eigvec)