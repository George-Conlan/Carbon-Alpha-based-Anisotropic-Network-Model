import numpy as np

def compute_msf(eigvals, eigvecs, N):
    msf = np.zeros(N)
    for k in range(len(eigvals)):
        v = eigvecs[:, k].reshape(N, 3)
        msf += (1.0 / eigvals[k]) * np.sum(v**2, axis=1)
    return msf

def compute_covariance(eigvals, eigvecs, N):
    cov = np.zeros((3*N,3*N))
    for k in range(len(eigvals)):
        v = eigvecs[:, k]
        cov+= (1.0/eigvals[k])*np.outer(v,v)
    return cov

def compute_collectivity(eigvecs, N):
    n_modes = eigvecs.shape[1]
    collectivity = np.zeros(n_modes)
    eps = 1e-12
    for k in range(n_modes):
        v = eigvecs[:, k].reshape(N, 3)
        sq = np.sum(v**2, axis=1)
        sq = sq / np.sum(sq)
        collectivity[k] = (1.0 / N) * np.exp(-np.sum(sq * np.log(sq + eps)))
    return collectivity
