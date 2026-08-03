from scipy.sparse.linalg import eigsh as eg
import numpy as np


def compute_modes(H, n_modes):
    eigvals, eigvecs= eg(H, k=n_modes, sigma= 1e-8, which='LM')
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs

def filter_modes(eigvals, eigvecs, tol, n_keep):
 nonzero_mask = eigvals>tol
 nonzero_vals= eigvals[nonzero_mask]   
 nonzero_vecs = eigvecs[:, nonzero_mask]
 zero_vals = eigvals[~nonzero_mask]
 zero_vecs = eigvecs[:, ~nonzero_mask]
 kept_vals = nonzero_vals[:n_keep]
 kept_vecs = nonzero_vecs[:, :n_keep]
 assert len(zero_vals) >= 6, "fewer than 6 near-zero eigenvalues found — increase n_modes"
 assert len(nonzero_vals) >= n_keep, "not enough nonzero modes to satisfy n_keep — increase n_modes"
 return kept_vals, kept_vecs, zero_vals, zero_vecs
