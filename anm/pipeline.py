import numpy as np
from dataclasses import dataclass
from typing import Optional

from .structure_io import load_structure, extract_ca_records, records_to_arrays
from .contact_graph import build_contact_graph, check_connected
from .hessian import build_hessian
from .modes import compute_modes, filter_modes
from .analysis import compute_msf, compute_covariance, compute_collectivity
from .validation import validate

@dataclass
class ANMResult:
    coords: np.ndarray
    labels: list
    bfactors: np.ndarray
    adj: np.ndarray
    contacts: np.ndarray
    D: np.ndarray
    connected: bool
    H: object
    eigvals: np.ndarray
    eigvecs: np.ndarray
    kept_vals: np.ndarray
    kept_vecs: np.ndarray
    msf: np.ndarray
    collectivity: np.ndarray
    scale: float
    correlation: tuple
    residuals: np.ndarray
    cov: Optional[np.ndarray] = None

def run_anm_pipeline(
    pdb_path: str,
    chain_id: Optional[str] = None,
    model_id: int = 0,
    cutoff: float = 8.0,
    gamma: float = 1.0,
    n_modes: int = 20,
    n_keep: Optional[int] = None,
    tol: float = 1e-6,
    compute_cov: bool = False,
    report_path: Optional[str] = "anm_report.pdf",
) -> ANMResult:
    structure = load_structure(pdb_path)
    records = extract_ca_records(structure, chain_id=chain_id, model_id=model_id)
    coords, bfactors, labels = records_to_arrays(records)
    N = coords.shape[0]

    adj, contacts, D = build_contact_graph(coords, cutoff)
    connected, components = check_connected(adj)
    if not connected:
        raise ValueError(
            f"Contact network is disconnected into {len(components)} components "
            f"at cutoff={cutoff}; increase cutoff or handle components separately."
        )

    H = build_hessian(coords, contacts, gamma)

    if n_keep is None:
        n_keep = n_modes - 6

    # A PDF report includes the correlation heatmap, which needs the
    # covariance matrix -- force it on whenever a report will be written,
    # even if the caller passed compute_cov=False.
    if report_path is not None:
        compute_cov = True

    # Real structures don't always yield exactly 6 near-zero (rigid-body)
    # eigenvalues -- e.g. a loosely connected loop can add an extra
    # near-zero "mechanism" mode. If that leaves fewer nonzero modes than
    # n_keep requires, ask the eigensolver for more modes and try again,
    # instead of failing on the first attempt.
    max_possible_modes = 3 * N - 1  # eigsh requires k < matrix dimension
    current_n_modes = min(n_modes, max_possible_modes)
    last_error = None
    for _ in range(5):
        eigvals, eigvecs = compute_modes(H, current_n_modes)
        try:
            kept_vals, kept_vecs, zero_vals, zero_vecs = filter_modes(eigvals, eigvecs, tol, n_keep)
            last_error = None
            break
        except AssertionError as exc:
            last_error = exc
            if current_n_modes >= max_possible_modes:
                break
            current_n_modes = min(current_n_modes + 10, max_possible_modes)
    if last_error is not None:
        raise last_error

    msf = compute_msf(kept_vals, kept_vecs, N)
    collectivity = compute_collectivity(kept_vecs, N)

    cov = None
    if compute_cov:
        cov = compute_covariance(kept_vals, kept_vecs, N)

    scale, correlation, residuals = validate(msf, bfactors)

    result = ANMResult(
        coords=coords,
        labels=labels,
        bfactors=bfactors,
        adj=adj,
        contacts=contacts,
        D=D,
        connected=connected,
        H=H,
        eigvals=eigvals,
        eigvecs=eigvecs,
        kept_vals=kept_vals,
        kept_vecs=kept_vecs,
        msf=msf,
        collectivity=collectivity,
        scale=scale,
        correlation=correlation,
        residuals=residuals,
        cov=cov,
    )

    if report_path is not None:
        # Imported here, not at module level, to avoid a circular import:
        # report.py doesn't import pipeline.py, but if this import were at
        # the top of the file it would still run before ANMResult exists.
        from .report import generate_report
        generate_report(result, report_path)

    return result
