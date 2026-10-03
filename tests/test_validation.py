import math
import warnings

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

from anm import run_anm_pipeline
from anm.validation import (bfactor_to_msf, compute_correlation, compute_residuals,
                            fit_scale, validate)

from .helpers import atom_line, grid_coords, write_pdb


def _stat(res):
    return float(res.statistic if hasattr(res, "statistic") else res[0])


# ---------------------------------------------------------------- known answers
def test_bfactor_to_msf_known_value():
    # B = 8 pi^2 <u^2> / 3  ->  <u^2> = 3B / (8 pi^2); B = 8 pi^2 / 3 gives exactly 1.
    assert bfactor_to_msf(np.array([8 * math.pi ** 2 / 3]))[0] == pytest.approx(1.0)
    b = np.array([10.0, 20.0, 40.0])
    np.testing.assert_allclose(bfactor_to_msf(b), b * 3 / (8 * math.pi ** 2))


def test_fit_scale_recovers_known_factor():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert fit_scale(pred, 2.5 * pred) == pytest.approx(2.5)


def test_correlation_perfect_positive_and_negative():
    x = np.arange(1.0, 11.0)
    p, s = compute_correlation(x, 3.0 * x + 7.0)
    assert _stat(p) == pytest.approx(1.0) and _stat(s) == pytest.approx(1.0)
    p, s = compute_correlation(x, -2.0 * x)
    assert _stat(p) == pytest.approx(-1.0) and _stat(s) == pytest.approx(-1.0)


def test_correlation_hand_computed_value():
    # dx = [-2,-1,0,1,2], dy = [-1,-2,1,0,2]: sum(dx*dy) = 8, sxx = syy = 10  ->  r = 0.8.
    # The ranks equal the values, so Spearman is also 0.8.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 1.0, 4.0, 3.0, 5.0])
    p, s = compute_correlation(x, y)
    assert _stat(p) == pytest.approx(0.8)
    assert _stat(s) == pytest.approx(0.8)


def test_spearman_is_rank_based_pearson_is_not():
    x = np.arange(1.0, 9.0)
    p, s = compute_correlation(x, x ** 5)  # monotonic but strongly nonlinear
    assert _stat(s) == pytest.approx(1.0)
    assert _stat(p) < 0.99


def test_correlation_is_scale_invariant():
    rng = np.random.default_rng(0)
    pred, exp = rng.random(30), rng.random(30)
    p1, s1 = compute_correlation(pred, exp)
    p2, s2 = compute_correlation(1000 * pred, exp)
    assert _stat(p1) == pytest.approx(_stat(p2))
    assert _stat(s1) == pytest.approx(_stat(s2))


def test_validate_end_to_end_known_answer():
    # Experimental B-factors that are exactly proportional to the prediction.
    pred = np.array([0.5, 1.0, 2.0, 4.0, 3.0, 1.5])
    c = 7.0
    bfactors = (c * pred) * 8 * math.pi ** 2 / 3        # so bfactor_to_msf(B) = c * pred
    scale, (p, s), resid = validate(pred, bfactors)
    assert scale == pytest.approx(c)
    assert _stat(p) == pytest.approx(1.0) and _stat(s) == pytest.approx(1.0)
    np.testing.assert_allclose(resid, 0.0, atol=1e-9)


def test_residuals_sign_and_value():
    pred, exp = np.array([1.0, 2.0]), np.array([3.0, 3.0])
    np.testing.assert_allclose(compute_residuals(pred, exp, 2.0), [-1.0, 1.0])  # scale*pred - exp


# ---------------------------------------------------------------- NaN / degenerate input
def test_nan_in_input_propagates_instead_of_giving_a_finite_number():
    # validate() does not mask NaNs: one NaN makes the scale and both
    # correlations NaN (never a plausible-looking finite value).
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bfac = np.array([10.0, np.nan, 30.0, 40.0, 50.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scale, (p, s), resid = validate(pred, bfac)
    assert np.isnan(scale)
    assert np.isnan(_stat(p)) and np.isnan(_stat(s))
    assert np.isnan(resid).all()


def test_shared_finite_mask_matches_clean_subset():
    # The benchmark's approach: drop non-finite residues from BOTH vectors with one mask.
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bfac = np.array([10.0, np.nan, 31.0, 38.0, 52.0, 59.0])
    ok = np.isfinite(pred) & np.isfinite(bfac)
    _, (p, s), _ = validate(pred[ok], bfac[ok])
    keep = [0, 2, 3, 4, 5]
    assert _stat(p) == pytest.approx(pearsonr(pred[keep], bfac[keep])[0])
    assert _stat(s) == pytest.approx(spearmanr(pred[keep], bfac[keep])[0])
    assert np.isfinite(_stat(p))


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        compute_correlation(np.arange(5.0), np.arange(4.0))


def test_constant_experimental_values_give_nan_not_a_number():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, s = compute_correlation(np.arange(5.0), np.full(5, 3.0))
    assert np.isnan(_stat(p)) and np.isnan(_stat(s))


# ---------------------------------------------------------------- missing residues + index alignment
def _grid_pdb(tmp_path, *, drop_ca=(), hetero=True, order=None):
    """27-residue cubic grid, one CA each. B-factor of residue k is 10 + k so every residue
    is identifiable. Residues in `drop_ca` keep an N atom but lose their CA."""
    coords = grid_coords(3)
    lines, serial = [], 1
    idx = list(range(27)) if order is None else list(order)
    for k in idx:
        resnum = k + 1
        if k in drop_ca:
            lines.append(atom_line(serial, "N", "GLY", "A", resnum, coords[k] + 0.5, 99.0))
        else:
            lines.append(atom_line(serial, "CA", "GLY", "A", resnum, coords[k], 10.0 + k))
        serial += 1
    if hetero:  # a calcium ion (atom name CA!) and a water, far from the protein
        lines.append(atom_line(serial, "CA", "CA", "A", 500, (100, 100, 100), 1.0, record="HETATM", element="CA"))
        lines.append(atom_line(serial + 1, "O", "HOH", "A", 501, (-50, 0, 0), 1.0, record="HETATM"))
    return write_pdb(tmp_path / "grid.pdb", lines)


def test_missing_ca_and_hetero_keep_vectors_aligned(tmp_path):
    dropped = {4, 20}
    res = run_anm_pipeline(_grid_pdb(tmp_path, drop_ca=dropped), n_modes="all", report_path=None)
    kept = [k for k in range(27) if k not in dropped]
    # one entry per residue that really has a CA; the HETATM calcium "CA" is excluded
    assert len(res.msf) == len(res.bfactors) == len(res.labels) == res.coords.shape[0] == len(kept)
    assert [lab[1] for lab in res.labels] == [k + 1 for k in kept]
    np.testing.assert_allclose(res.bfactors, [10.0 + k for k in kept])
    np.testing.assert_allclose(res.coords, grid_coords(3)[kept])


def test_msf_index_i_is_residue_i_physical_check(tmp_path):
    # In a symmetric lattice the buried centre residue is the stiffest and a corner the most flexible.
    res = run_anm_pipeline(_grid_pdb(tmp_path, hetero=False), n_modes="all", report_path=None)
    centre = 13  # (1,1,1) in x-major order
    corners = {0, 2, 6, 8, 18, 20, 24, 26}
    assert int(np.argmin(res.msf)) == centre
    assert int(np.argmax(res.msf)) in corners


def test_msf_follows_residue_order_in_file(tmp_path):
    # Writing the same structure in reversed residue order must reverse the MSF vector,
    # and the B-factor vector must be reversed with it (same index <-> same residue).
    fwd = run_anm_pipeline(_grid_pdb(tmp_path, hetero=False), n_modes="all", report_path=None)
    rev = run_anm_pipeline(_grid_pdb(tmp_path, hetero=False, order=range(26, -1, -1)),
                           n_modes="all", report_path=None)
    np.testing.assert_allclose(rev.msf, fwd.msf[::-1], rtol=1e-7)
    np.testing.assert_allclose(rev.bfactors, fwd.bfactors[::-1])
    assert [lab[1] for lab in rev.labels] == [lab[1] for lab in fwd.labels][::-1]


def test_pipeline_correlation_matches_manual_computation(tmp_path):
    res = run_anm_pipeline(_grid_pdb(tmp_path, hetero=False), n_modes="all", report_path=None)
    exp = bfactor_to_msf(res.bfactors)
    assert _stat(res.correlation[0]) == pytest.approx(pearsonr(res.msf, exp)[0])
    assert _stat(res.correlation[1]) == pytest.approx(spearmanr(res.msf, exp)[0])
    assert res.msf.shape == res.bfactors.shape
