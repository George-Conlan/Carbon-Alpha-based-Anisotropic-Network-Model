import numpy as np
import pytest

from anm.structure_io import extract_ca_records, load_structure, records_to_arrays

from .helpers import atom_line, write_pdb


def _two_chain_pdb(tmp_path):
    lines = [
        atom_line(1, "N", "ALA", "A", 1, (0.0, 0.0, 0.0), 5.0),
        atom_line(2, "CA", "ALA", "A", 1, (1.0, 0.0, 0.0), 11.0),
        atom_line(3, "CA", "GLY", "A", 2, (2.0, 0.0, 0.0), 12.0),
        atom_line(4, "N", "SER", "A", 3, (3.0, 0.0, 0.0), 6.0),          # no CA -> skipped
        atom_line(5, "CA", "LEU", "A", 4, (4.0, 0.0, 0.0), 14.0),
        atom_line(6, "CA", "CA", "A", 100, (50.0, 0.0, 0.0), 1.0, record="HETATM", element="CA"),
        atom_line(7, "O", "HOH", "A", 101, (60.0, 0.0, 0.0), 1.0, record="HETATM"),
        atom_line(8, "CA", "VAL", "B", 1, (0.0, 5.0, 0.0), 21.0),
        atom_line(9, "CA", "VAL", "B", 2, (0.0, 6.0, 0.0), 22.0),
    ]
    return write_pdb(tmp_path / "two_chain.pdb", lines)


def test_load_structure_reads_pdb(tmp_path):
    structure = load_structure(_two_chain_pdb(tmp_path))
    assert [c.id for c in structure[0]] == ["A", "B"]


def test_load_structure_accepts_ent_extension(tmp_path):
    p = _two_chain_pdb(tmp_path)
    ent = tmp_path / "x.ent"
    ent.write_text(open(p).read())
    assert len(list(load_structure(str(ent))[0])) == 2


def test_load_structure_rejects_unknown_extension(tmp_path):
    f = tmp_path / "structure.xyz"
    f.write_text("nothing")
    with pytest.raises(ValueError, match="Unrecognized"):
        load_structure(str(f))


def test_extract_skips_hetero_water_and_residues_without_ca(tmp_path):
    recs = extract_ca_records(load_structure(_two_chain_pdb(tmp_path)), chain_id="A")
    assert [r["resnum"] for r in recs] == [1, 2, 4]          # 3 has no CA; 100/101 are HETATM
    assert [r["resname"] for r in recs] == ["ALA", "GLY", "LEU"]
    assert [r["bfactor"] for r in recs] == [11.0, 12.0, 14.0]  # the CA's B, not the N's
    np.testing.assert_allclose(recs[0]["coord"], [1.0, 0.0, 0.0])


def test_extract_chain_filter_and_all_chains(tmp_path):
    s = load_structure(_two_chain_pdb(tmp_path))
    assert {r["chain"] for r in extract_ca_records(s, chain_id="B")} == {"B"}
    both = extract_ca_records(s)
    assert [(r["chain"], r["resnum"]) for r in both] == [("A", 1), ("A", 2), ("A", 4), ("B", 1), ("B", 2)]
    assert extract_ca_records(s, chain_id="Z") == []


def test_extract_altloc_uses_highest_occupancy(tmp_path):
    lines = [
        atom_line(1, "CA", "ALA", "A", 1, (1.0, 0.0, 0.0), 20.0, occ=0.3, altloc="A"),
        atom_line(2, "CA", "ALA", "A", 1, (2.0, 0.0, 0.0), 30.0, occ=0.7, altloc="B"),
        atom_line(3, "CA", "GLY", "A", 2, (5.0, 0.0, 0.0), 12.0),
    ]
    recs = extract_ca_records(load_structure(write_pdb(tmp_path / "alt.pdb", lines)))
    assert len(recs) == 2
    np.testing.assert_allclose(recs[0]["coord"], [2.0, 0.0, 0.0])
    assert recs[0]["bfactor"] == 30.0


def test_records_to_arrays_shapes_and_labels(tmp_path):
    recs = extract_ca_records(load_structure(_two_chain_pdb(tmp_path)), chain_id="A")
    coords, bfactors, labels = records_to_arrays(recs)
    assert coords.shape == (3, 3) and coords.dtype == np.float64
    assert bfactors.shape == (3,)
    assert labels == [("A", 1, "ALA"), ("A", 2, "GLY"), ("A", 4, "LEU")]
    np.testing.assert_allclose(coords[:, 0], [1.0, 2.0, 4.0])
    np.testing.assert_allclose(bfactors, [11.0, 12.0, 14.0])
