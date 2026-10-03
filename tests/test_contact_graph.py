import numpy as np

from anm.contact_graph import build_contact_graph, check_connected, distance_matrix

LINE = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]])  # distances 5, 12, 13


def test_distance_matrix_known_values():
    D = distance_matrix(LINE)
    expected = np.array([[0, 5, 13], [5, 0, 12], [13, 12, 0]], dtype=float)
    np.testing.assert_allclose(D, expected)


def test_distance_matrix_symmetric_zero_diagonal():
    D = distance_matrix(np.random.default_rng(1).random((8, 3)) * 20)
    np.testing.assert_allclose(D, D.T)
    np.testing.assert_allclose(np.diag(D), 0.0)


def test_contacts_respect_cutoff():
    adj, contacts, D = build_contact_graph(LINE, cutoff=12.0)
    # 0-1 (5) and 1-2 (12, inclusive) are in contact; 0-2 (13) is not
    assert adj[0, 1] and adj[1, 2] and not adj[0, 2]
    assert contacts.tolist() == [[0, 1], [1, 2]]


def test_cutoff_is_inclusive_at_exact_distance():
    coords = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    assert build_contact_graph(coords, 8.0)[0][0, 1]
    assert not build_contact_graph(coords, 7.999)[0][0, 1]


def test_adjacency_symmetric_no_self_contacts_and_contacts_upper_triangle():
    rng = np.random.default_rng(2)
    adj, contacts, _ = build_contact_graph(rng.random((20, 3)) * 15, 8.0)
    assert (adj == adj.T).all()
    assert not adj.diagonal().any()
    assert (contacts[:, 0] < contacts[:, 1]).all()
    assert len(contacts) == adj.sum() // 2


def test_check_connected_true_and_false():
    adj, _, _ = build_contact_graph(LINE, cutoff=12.0)
    connected, comps = check_connected(adj)
    assert connected and len(comps) == 1

    adj, _, _ = build_contact_graph(LINE, cutoff=8.0)  # node 2 is 12+ A from the others
    connected, comps = check_connected(adj)
    assert not connected
    assert sorted(len(c) for c in comps) == [1, 2]
