from Bio.PDB import PDBParser, MMCIFParser
import numpy as np


def load_structure(path):
    if path.lower().endswith('.pdb') or path.lower().endswith('.ent'):
        parser = PDBParser(QUIET=True)
    elif path.lower().endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unrecognized structure file extension: {path}")

    structure = parser.get_structure('protein', path)
    return structure


def extract_ca_records(structure, chain_id = None, model_id = 0):
    records = []
    model = structure[model_id]
    for chain in model: 
        if chain_id and chain.id != chain_id:
            continue
        for residue in chain:
            if residue.id[0] != ' ':
               continue # skip herero groups / waters
            if 'CA' not in residue:
                continue # log this as a gap before skipping

            atom = residue['CA']
            if atom.is_disordered():
                atom =atom.disordered_get()

            records.append({
                'chain': chain.id,
                'resnum': residue.id[1],
                'resname': residue.resname,
                'coord': atom.get_coord(),
                'bfactor': atom.get_bfactor(),
            })

    return records


def records_to_arrays(records):
    coords = np.array([r['coord'] for r in records], dtype=np.float64)
    bfactors = np.array([r['bfactor'] for r in records], dtype=np.float64)
    labels = [(r['chain'], r['resnum'], r['resname']) for r in records]
    return coords, bfactors, labels
                      
