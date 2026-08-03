#!/usr/bin/env python
"""
Command-line entry point for the Carbon-alpha ANM pipeline.

Examples:
    python run_anm.py 1UBQ
    python run_anm.py 1UBQ --chain A --cutoff 10 --n-modes 30
    python run_anm.py path/to/structure.pdb --no-report
"""

import argparse
import os
import sys
import urllib.error
import urllib.request

import matplotlib
matplotlib.use("Agg")

import anm

RCSB_PDB_URL = "https://files.rcsb.org/download/{id}.pdb"


def _stat(res):
    """Pull the correlation coefficient out of a scipy pearsonr/spearmanr result,
    across scipy versions that return either a named result object or a plain tuple."""
    if hasattr(res, "statistic"):
        return res.statistic
    if hasattr(res, "correlation"):
        return res.correlation
    return res[0]


def resolve_structure_path(target, download_dir):
    if os.path.isfile(target):
        return target

    pdb_id = target.strip().upper()
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        raise SystemExit(
            f"'{target}' is neither an existing file nor a 4-character PDB ID "
            f"(e.g. 1UBQ)."
        )

    os.makedirs(download_dir, exist_ok=True)
    dest = os.path.join(download_dir, f"{pdb_id}.pdb")
    if not os.path.exists(dest):
        url = RCSB_PDB_URL.format(id=pdb_id)
        print(f"Downloading {pdb_id} from RCSB PDB...")
        try:
            urllib.request.urlretrieve(url, dest)
        except urllib.error.HTTPError as exc:
            raise SystemExit(
                f"Could not download '{pdb_id}' from RCSB (HTTP {exc.code}). "
                f"Check that the PDB ID is correct."
            )
        except urllib.error.URLError as exc:
            raise SystemExit(f"Network error while downloading '{pdb_id}': {exc.reason}")
    return dest


def main():
    parser = argparse.ArgumentParser(
        description="Run the Carbon-alpha-only Anisotropic Network Model pipeline on a protein structure."
    )
    parser.add_argument(
        "target",
        help="A 4-character PDB ID to download (e.g. 1UBQ), or a path to a local .pdb/.cif file.",
    )
    parser.add_argument("--chain", default=None, help="Restrict analysis to a single chain ID.")
    parser.add_argument("--cutoff", type=float, default=8.0, help="Contact cutoff distance in Angstroms (default: 8.0).")
    parser.add_argument("--gamma", type=float, default=1.0, help="Uniform spring constant (default: 1.0).")
    parser.add_argument("--n-modes", type=int, default=20, help="Number of low-frequency modes to solve for (default: 20).")
    parser.add_argument("--n-keep", type=int, default=None, help="Number of nonzero modes to keep (default: n-modes - 6).")
    parser.add_argument("--tol", type=float, default=1e-6, help="Eigenvalue tolerance for the rigid-body cutoff (default: 1e-6).")
    parser.add_argument("--no-report", action="store_true", help="Skip generating the PDF report.")
    parser.add_argument("--report-path", default=None, help="Output path for the PDF report (default: <structure>_report.pdf).")
    parser.add_argument("--animate", action="store_true", help="Also render an animated GIF of the lowest-frequency mode.")
    parser.add_argument("--animate-path", default=None, help="Output path for the mode animation GIF (default: <structure>_mode.gif).")
    parser.add_argument("--download-dir", default="structures", help="Directory downloaded PDB files are saved into (default: ./structures).")
    args = parser.parse_args()

    structure_path = resolve_structure_path(args.target, args.download_dir)

    if args.no_report:
        report_path = None
    elif args.report_path:
        report_path = args.report_path
    else:
        base = os.path.splitext(os.path.basename(structure_path))[0]
        report_path = f"{base}_report.pdf"

    try:
        result = anm.run_anm_pipeline(
            structure_path,
            chain_id=args.chain,
            cutoff=args.cutoff,
            gamma=args.gamma,
            n_modes=args.n_modes,
            n_keep=args.n_keep,
            tol=args.tol,
            report_path=report_path,
        )
    except ValueError as exc:
        raise SystemExit(f"Pipeline error: {exc}")

    pearson_r = _stat(result.correlation[0])
    spearman_r = _stat(result.correlation[1])

    print()
    print(f"Structure:                {structure_path}")
    print(f"Residues analyzed:        {result.coords.shape[0]}")
    print(f"Contact network connected: {result.connected}")
    print(f"Fit scale factor:         {result.scale:.4f}")
    print(f"Pearson r vs B-factors:   {pearson_r:.3f}")
    print(f"Spearman r vs B-factors:  {spearman_r:.3f}")
    if report_path:
        print(f"Report written to:       {report_path}")

    if args.animate:
        base = os.path.splitext(os.path.basename(structure_path))[0]
        animate_path = args.animate_path or f"{base}_mode.gif"
        anm.animate_mode(result.coords, result.kept_vecs[:, 0], animate_path)
        print(f"Mode animation written to: {animate_path}")

    print()


if __name__ == "__main__":
    main()
