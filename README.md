# Carbon-alpha Anisotropic Network Model (ANM)

A small Python pipeline that predicts protein flexibility from structure alone, using a Carbon-alpha-only Anisotropic Network Model, and validates the prediction against experimental B-factors.

Point it at a PDB ID or a local structure file and it will build the residue contact network, construct and diagonalize the ANM Hessian, derive per-residue flexibility (mean-square fluctuation), compare that prediction to the structure's own experimental B-factors, and generate a PDF report with all of it laid out — plus, optionally, an animated GIF of the backbone oscillating along its lowest-frequency normal mode.

## Background

An Anisotropic Network Model treats a protein as a mass-and-spring network: each Cα atom is a node, and any two Cα atoms within a cutoff distance are connected by a spring of uniform stiffness (`gamma`). The normal modes of this network — the eigenvectors/eigenvalues of its Hessian — describe the protein's natural low-frequency motions. Mean-square fluctuation (MSF) derived from those modes correlates, for most well-behaved structures, with the experimental B-factors already present in the PDB file, which is what this pipeline checks.

## Installation

Requires Python 3.9+.

```
git clone <this-repo-url>
cd "Protein Structure Math Modeling- Building a Carbon alpha only anisotropic nerwork model"
pip install -r requirements.txt
```

## Usage

Run it from a terminal, either with a 4-character PDB ID (auto-downloaded from RCSB) or a path to a local `.pdb`/`.cif` file:

```
python run_anm.py 1UBQ
python run_anm.py 6VXX --chain A --cutoff 10 --n-modes 30
python run_anm.py path/to/structure.pdb --no-report
python run_anm.py 1UBQ --animate
```

This prints a short summary to the terminal and (by default) writes `<structure>_report.pdf` alongside it — a single PDF containing the contact map, the predicted-vs-experimental MSF overlay, the mode correlation heatmap, 1D and 3D flexibility heatmaps, and a plain-language analysis page (correlation strength, most flexible/rigid residues, mode collectivity).

### CLI options

| Flag | Default | Meaning |
|---|---|---|
| `target` | — | 4-character PDB ID to download, or a path to a local structure file |
| `--chain` | all chains | Restrict analysis to a single chain ID |
| `--cutoff` | `8.0` | Contact cutoff distance, in Angstroms |
| `--gamma` | `1.0` | Uniform spring constant |
| `--n-modes` | `20` | Number of low-frequency modes to solve for |
| `--n-keep` | `n-modes - 6` | Number of nonzero (non-rigid-body) modes to keep |
| `--tol` | `1e-6` | Eigenvalue tolerance for the rigid-body mode cutoff |
| `--no-report` | off | Skip generating the PDF report |
| `--report-path` | `<structure>_report.pdf` | Custom output path for the PDF report |
| `--animate` | off | Also render an animated GIF of the lowest-frequency mode |
| `--animate-path` | `<structure>_mode.gif` | Custom output path for the mode animation |
| `--download-dir` | `structures/` | Where downloaded PDB files are saved |

### As a library

```python
import anm

result = anm.run_anm_pipeline("1ubq.pdb", cutoff=8.0, n_modes=20)

result.msf            # predicted mean-square fluctuation per residue
result.correlation     # (PearsonRResult, SpearmanrResult) vs experimental B-factors
result.connected        # whether the contact network was a single connected component
```

Every stage is also usable on its own — see `anm/__init__.py` for the full list of exported functions (`load_structure`, `build_contact_graph`, `build_hessian`, `compute_modes`, `compute_msf`, `plot_contact_map`, `generate_report`, etc.).

## Validation study

`validation/multi_protein_validation.ipynb` runs the pipeline against six real, well-characterized structures (crambin, ubiquitin, lysozyme, a ribosomal protein domain, T4 lysozyme, and adenylate kinase) and compares predicted flexibility to experimental B-factors across all of them, including a closer look at adenylate kinase's hinge motion. GitHub renders the notebook with its outputs already in place, so no setup is needed just to read it. To re-run it yourself:

```
pip install -r validation/requirements.txt
jupyter notebook validation/multi_protein_validation.ipynb
```

## Project structure

```
anm/
    structure_io.py     load a PDB/mmCIF file, extract Cα coordinates + B-factors
    contact_graph.py    build the residue contact network from a distance cutoff
    hessian.py           construct the ANM Hessian from contacts
    modes.py              diagonalize the Hessian, filter out rigid-body modes
    analysis.py           derive MSF, covariance, and mode collectivity
    validation.py        compare predicted MSF to experimental B-factors
    visualize.py          individual matplotlib plot functions + mode animation (GIF)
    report.py              assemble all plots + a written analysis into one PDF
    pipeline.py           orchestrates every stage above end to end
    __init__.py            public package API
run_anm.py                 command-line entry point
validation/                 multi-protein validation notebook (see below)
tests/                      pytest suite
```

## Testing

```
pytest
```

Test coverage currently exists for `analysis.py`, `hessian.py`, and `modes.py`. `structure_io.py`, `contact_graph.py`, `validation.py`, `visualize.py`, `pipeline.py`, and `report.py` don't have tests yet.

## License

MIT — see [LICENSE](LICENSE). Use it, modify it, ship it, just keep the copyright notice.
