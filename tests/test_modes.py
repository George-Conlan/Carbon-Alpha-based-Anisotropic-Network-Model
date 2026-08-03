import numpy as np
import pytest

from anm.contact_graph import build_contact_graph
from anm.hessian import build_hessian
from anm.modes import compute_modes, filter_modes

COORDS = np.array([
    [0.0, 0.0, 0.0],
    [3.8, 0.0, 0.0],
    [0.0, 3.8, 0.0],
    [0.0, 0.0, 3.8],
])
GAMMA = 1.0
CUTOFF = 8.0


def _build_H():
    _, contacts, _ = build_contact_graph(COORDS, CUTOFF)
    return build_hessian(COORDS, contacts, GAMMA)


def test_compute_modes_sorted_ascending():
    eigvals, _ = compute_modes(_build_H(), n_modes=10)
    assert np.all(np.diff(eigvals) >= -1e-10)


def test_compute_modes_has_six_zero_modes():
    # 4 non-collinear points in 3D -> exactly 6 rigid-body (near-zero) modes:
    # 3 translational + 3 rotational.
    eigvals, _ = compute_modes(_build_H(), n_modes=10)
    assert np.sum(eigvals < 1e-6) == 6
    assert eigvals[6] > 1e-3  # first real vibrational mode is clearly nonzero


def test_filter_modes_splits_zero_and_nonzero():
    eigvals = np.array([0.0, 0.0, 1e-9, -1e-9, 1e-8, 0.0, 2.0, 3.0, 4.0])
    eigvecs = np.eye(9)
    kept_vals, kept_vecs, zero_vals, zero_vecs = filter_modes(
        eigvals, eigvecs, tol=1e-6, n_keep=3
    )
    assert len(zero_vals) == 6
    assert zero_vecs.shape == (9, 6)
    np.testing.assert_allclose(kept_vals, [2.0, 3.0, 4.0])
    assert kept_vecs.shape == (9, 3)


def test_filter_modes_raises_without_enough_zero_modes():
    eigvals = np.array([1.0, 2.0, 3.0, 4.0])
    eigvecs = np.eye(4)
    with pytest.raises(AssertionError):
        filter_modes(eigvals, eigvecs, tol=1e-6, n_keep=1)


def test_filter_modes_raises_without_enough_nonzero_modes():
    eigvals = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    eigvecs = np.eye(7)
    with pytest.raises(AssertionError):
        filter_modes(eigvals, eigvecs, tol=1e-6, n_keep=2)
