"""Shared helpers for the benchmark scripts: cached/retrying RCSB access and
per-chain structural bookkeeping read straight from PDB-format files."""
import json
import os
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))

def _default_cache_dir():
    """validation/cache, unless overridden with ANM_BENCH_CACHE or (on Windows)
    the repo path is so long that cache file paths would exceed MAX_PATH."""
    env = os.environ.get("ANM_BENCH_CACHE")
    if env:
        return env
    local = os.path.join(HERE, "cache")
    if os.name == "nt" and len(local) > 170:
        return os.path.join(os.path.expanduser("~"), ".anm_benchmark_cache")
    return local


CACHE_DIR = _default_cache_dir()
PDB_DIR = os.path.join(CACHE_DIR, "pdb")
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
PDB_URL = "https://files.rcsb.org/download/{}.pdb"


EXT_PREFIX = "\\\\?\\"  # Windows extended-length path prefix: \\?\


def lp(path):
    """Extended-length path on Windows, so outputs still work when the repo
    lives in a deeply nested folder (MAX_PATH = 260). No-op elsewhere."""
    path = os.path.abspath(path)
    if os.name == "nt" and not path.startswith(EXT_PREFIX):
        return EXT_PREFIX + path
    return path


def _retry(fn, attempts=5, what="request"):
    """Run fn(); retry with exponential backoff on network / 5xx errors."""
    last = None
    for k in range(attempts):
        try:
            return fn()
        except (requests.RequestException, RuntimeError) as exc:
            last = exc
            time.sleep(min(2 ** k, 30))
    raise RuntimeError(f"{what} failed after {attempts} attempts: {last}")


def cached_json(name, producer):
    """Return JSON cached at cache/<name>.json, calling producer() on a miss."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name + ".json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    data = producer()
    with open(path, "w") as fh:
        json.dump(data, fh)
    return data


def post_json(url, payload):
    def go():
        r = requests.post(url, json=payload, timeout=90)
        if r.status_code >= 500:
            raise RuntimeError(f"HTTP {r.status_code}")
        r.raise_for_status()
        return r.json()
    return _retry(go, what=f"POST {url}")


def download_pdb(pdb_id, pdb_dir=PDB_DIR):
    """Download (once) and return the path of a PDB-format file, or None if
    RCSB has no PDB-format file for the entry (e.g. very large structures)."""
    os.makedirs(pdb_dir, exist_ok=True)
    path = os.path.join(pdb_dir, pdb_id.lower() + ".pdb")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    def go():
        r = requests.get(PDB_URL.format(pdb_id.upper()), timeout=90)
        if r.status_code == 404:
            return None
        if r.status_code >= 500:
            raise RuntimeError(f"HTTP {r.status_code}")
        r.raise_for_status()
        return r.text

    text = _retry(go, what=f"download {pdb_id}")
    if text is None:
        return None
    with open(path, "w") as fh:
        fh.write(text)
    return path


_AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
        "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE"}


def chain_stats(pdb_path, chain):
    """Structural bookkeeping for one chain of a PDB-format file, from the
    SEQRES/HELIX/SHEET/ATOM records (model 1 only). Returns a dict."""
    seqres = 0
    helix = set()
    sheet = set()
    strands = []  # (sense, residues)
    ca = []       # (resseq, icode, bfactor)
    seen = set()
    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6]
            if rec == "SEQRES" and line[11] == chain:
                seqres = int(line[13:17])
            elif rec == "HELIX " and line[19] == chain and line[31] == chain:
                a, b = int(line[21:25]), int(line[33:37])
                helix.update(range(a, b + 1))
            elif rec == "SHEET " and line[21] == chain and line[32] == chain:
                a, b = int(line[22:26]), int(line[33:37])
                sense = int(line[38:40]) if line[38:40].strip() else 0
                res = set(range(a, b + 1))
                sheet.update(res)
                strands.append((sense, len(res)))
            elif rec == "ENDMDL":
                break
            elif rec == "ATOM  " and line[21] == chain and line[12:16] == " CA ":
                if line[16] not in (" ", "A"):
                    continue
                key = (int(line[22:26]), line[26])
                if key in seen:
                    continue
                seen.add(key)
                ca.append((key[0], key[1], float(line[60:66])))
    n = len(ca)
    resnums = [c[0] for c in ca]
    gaps = [b - a - 1 for a, b in zip(resnums, resnums[1:]) if b - a > 1]
    obs = set(resnums)
    bf = [c[2] for c in ca]
    anti = sum(l for s, l in strands if s == -1)
    par = sum(l for s, l in strands if s == 1)
    return {
        "n_ca": n,
        "seqres_len": seqres,
        "n_gap_residues": int(sum(gaps)),
        "max_gap": max(gaps) if gaps else 0,
        "helix_frac": len(helix & obs) / n if n else 0.0,
        "sheet_frac": len(sheet & obs) / n if n else 0.0,
        "parallel_frac": par / (par + anti) if (par + anti) else 0.0,
        "mean_b": sum(bf) / n if n else float("nan"),
        "b_std": (sum((x - sum(bf) / n) ** 2 for x in bf) / n) ** 0.5 if n else 0.0,
    }
