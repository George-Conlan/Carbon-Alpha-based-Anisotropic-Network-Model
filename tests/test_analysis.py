import numpy as np

from anm.contact_graph import build_contact_graph
from anm.hessian import build_hessian
from anm.modes import compute_modes, filter_modes
from anm.analysis import compute_msf, compute_covariance, compute_collectivity

COORDS = np.array([
    [0.0, 0.0, 0.0],
    [3.8, 0.0, 0.0],
    [0.0, 3.8, 0.0],
    [0.0, 0.0, 3.8],
])
GAMMA = 1.0
CUTOFF = 8.0
N = 4


def _nonzero_modes():
    _, contacts, _ = build_contact_graph(COORDS, CUTOFF)
    H = build_hessian(COORDS, contacts, GAMMA)
    eigvals, eigvecs = compute_modes(H, n_modes=10)
    kept_vals, kept_vecs, _, _ = filter_modes(eigvals, eigvecs, tol=1e-6, n_keep=4)
    return kept_vals, kept_vecs


def test_compute_msf_is_positive():
    eigvals, eigvecs = _nonzero_modes()
    msf = compute_msf(eigvals, eigvecs, N)
    assert msf.shape == (N,)
    assert np.all(msf > 0)


def test_compute_covariance_diagonal_matches_msf():
    # Tracing each residue's 3x3 diagonal block of the covariance matrix
    # must reproduce compute_msf exactly, since it's the same weighted
    # mode sum before/after collapsing to a per-residue scalar.
    eigvals, eigvecs = _nonzero_modes()
    msf = compute_msf(eigvals, eigvecs, N)
    cov = compute_covariance(eigvals, eigvecs, N)
    assert cov.shape == (3 * N, 3 * N)
    for i in range(N):
        block = cov[3 * i:3 * i + 3, 3 * i:3 * i + 3]
        np.testing.assert_allclose(np.trace(block), msf[i])


def test_compute_covariance_is_symmetric():
    eigvals, eigvecs = _nonzero_modes()
    cov = compute_covariance(eigvals, eigvecs, N)
    np.testing.assert_allclose(cov, cov.T)


def test_compute_collectivity_bounds():
    _, eigvecs = _nonzero_modes()
    collectivity = compute_collectivity(eigvecs, N)
    assert collectivity.shape == (eigvecs.shape[1],)
    # kappa ranges from 1/N (fully localized to one residue) to 1
    # (uniformly spread across all N residues).
    assert np.all(collectivity >= 1.0 / N - 1e-9)
    assert np.all(collectivity <= 1.0 + 1e-9)
