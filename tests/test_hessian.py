import numpy as np

from anm.contact_graph import build_contact_graph
from anm.hessian import pair_block, build_hessian

# Non-planar, non-collinear 4-point system (a tetrahedron) so the network
# has well-defined rigid-body modes but no accidental extra degeneracy.
COORDS = np.array([
    [0.0, 0.0, 0.0],
    [3.8, 0.0, 0.0],
    [0.0, 3.8, 0.0],
    [0.0, 0.0, 3.8],
])
GAMMA = 1.0
CUTOFF = 8.0  # large enough that every pair is in contact


def _build_H():
    _, contacts, _ = build_contact_graph(COORDS, CUTOFF)
    return build_hessian(COORDS, contacts, GAMMA)


def test_pair_block_known_values():
    coord_i = np.array([0.0, 0.0, 0.0])
    coord_j = np.array([1.0, 0.0, 0.0])
    block = pair_block(coord_i, coord_j, gamma=2.0)
    expected = np.array([
        [-2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    np.testing.assert_allclose(block, expected)


def test_pair_block_symmetric():
    block = pair_block(COORDS[0], COORDS[1], GAMMA)
    np.testing.assert_allclose(block, block.T)


def test_build_hessian_shape():
    H = _build_H()
    assert H.shape == (12, 12)


def test_build_hessian_symmetric():
    H = _build_H().toarray()
    np.testing.assert_allclose(H, H.T)


def test_build_hessian_translational_invariance():
    # A uniform rigid-body translation costs zero energy, so the Hessian
    # applied to a pure-translation displacement vector must vanish.
    H = _build_H().toarray()
    for axis in range(3):
        d = np.zeros(12)
        d[axis::3] = 1.0
        np.testing.assert_allclose(H @ d, np.zeros(12), atol=1e-10)
