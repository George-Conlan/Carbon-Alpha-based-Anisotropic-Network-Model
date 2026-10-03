"""Helpers for building small synthetic PDB files in tests."""
import itertools

import numpy as np


def atom_line(serial, name, resname, chain, resnum, xyz, bfactor=10.0, occ=1.0,
              altloc=" ", record="ATOM", element=None):
    """One fixed-width PDB ATOM/HETATM line."""
    element = element or name.strip()[:1]
    x, y, z = xyz
    return (f"{record:<6}{serial:>5} {' ' + name:<4}{altloc}{resname:>3} {chain}{resnum:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfactor:6.2f}          {element:>2}")


def write_pdb(path, lines):
    path.write_text("\n".join(list(lines) + ["END"]) + "\n")
    return str(path)


def grid_coords(n=3, spacing=3.8):
    """n x n x n cubic lattice of points, in a fixed (x-major) order."""
    return np.array([[i * spacing, j * spacing, k * spacing]
                     for i, j, k in itertools.product(range(n), repeat=3)], dtype=float)
