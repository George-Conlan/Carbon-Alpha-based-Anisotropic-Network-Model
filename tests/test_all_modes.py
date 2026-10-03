import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

from anm import run_anm_pipeline
from anm.contact_graph import build_contact_graph
from anm.hessian import build_hessian
from anm.modes import compute_all_modes, filter_modes

from .helpers import atom_line, grid_coords, write_pdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_anm  # noqa: E402


def _grid_hessian():
    coords = grid_coords(3)
    _, contacts, _ = build_contact_graph(coords, 8.0)
    return coords, build_hessian(coords, contacts, 1.0)


def _grid_pdb(tmp_path):
    coords = grid_coords(3)
    lines = [atom_line(i + 1, "CA", "GLY", "A", i + 1, c, 10.0 + i) for i, c in enumerate(coords)]
    return write_pdb(tmp_path / "grid.pdb", lines)


def test_compute_all_modes_returns_full_sorted_spectrum():
    coords, H = _grid_hessian()
    vals, vecs = compute_all_modes(H)
    assert vals.shape == (3 * len(coords),) and vecs.shape == (81, 81)
    assert np.all(np.diff(vals) >= -1e-10)
    assert np.sum(vals < 1e-6) == 6  # 3 translations + 3 rotations


def test_filter_modes_n_keep_none_keeps_every_nonzero_mode():
    _, H = _grid_hessian()
    vals, vecs = compute_all_modes(H)
    kept_vals, kept_vecs, zero_vals, _ = filter_modes(vals, vecs, 1e-6, None)
    assert len(zero_vals) == 6
    assert len(kept_vals) == 81 - 6 and kept_vecs.shape == (81, 75)


def test_all_modes_msf_equals_pseudoinverse_diagonal(tmp_path):
    # Known answer: sum_k |v_k,i|^2 / lambda_k over all non-zero modes is the 3x3 diagonal
    # block trace of the Hessian pseudo-inverse.
    coords, H = _grid_hessian()
    pinv = np.linalg.pinv(H.toarray(), hermitian=True, rcond=1e-9)
    expected = np.array([np.trace(pinv[3 * i:3 * i + 3, 3 * i:3 * i + 3]) for i in range(len(coords))])
    res = run_anm_pipeline(_grid_pdb(tmp_path), n_modes="all", report_path=None)
    np.testing.assert_allclose(res.msf, expected, rtol=1e-7)
    assert len(res.kept_vals) == 3 * len(coords) - 6


def test_all_modes_case_insensitive_and_bad_string_rejected(tmp_path):
    path = _grid_pdb(tmp_path)
    assert len(run_anm_pipeline(path, n_modes="ALL", report_path=None).kept_vals) == 75
    with pytest.raises(ValueError, match="n_modes"):
        run_anm_pipeline(path, n_modes="everything", report_path=None)


def test_default_pipeline_behaviour_unchanged(tmp_path):
    res = run_anm_pipeline(_grid_pdb(tmp_path), report_path=None)  # n_modes=20 default
    assert len(res.kept_vals) == 14
    assert len(res.eigvals) == 20


def test_cli_n_modes_parser():
    assert run_anm._n_modes("all") == "all"
    assert run_anm._n_modes("ALL") == "all"
    assert run_anm._n_modes("30") == 30
    with pytest.raises(argparse.ArgumentTypeError):
        run_anm._n_modes("lots")
