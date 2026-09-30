"""Build the 100-protein benchmark list (validation/protein_benchmark.csv).

Selection uses ONLY structure-quality / diversity criteria. The ANM is never
run here, so no ANM result can influence which proteins are chosen.

Pipeline
  1. RCSB Search API: X-ray, resolution <= 2.5 A, one protein entity and one
     deposited chain, 40-500 residues, clustered at 30 % sequence identity
     (one representative per cluster -> no homologous pairs above 30 % id).
  2. RCSB GraphQL: metadata for a seeded random sample of those cluster
     representatives (name, organism, keywords, EC, assembly oligomeric state).
  3. Download PDB files; compute observed-residue coverage, internal gaps,
     secondary-structure fractions (from PDB HELIX/SHEET records), B-factor
     spread; apply quality filters; assign fold class and functional class.
  4. Seeded, stratified selection over size bin x fold class, balancing
     functional class and resolution.

Run:  python validation/build_benchmark.py
"""
import argparse
import os
import random
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu  # noqa: E402

SEED = 20260930
N_TARGET = 100
POOL_SAMPLE = 1600          # cluster representatives whose metadata we fetch
MAX_RES = 2.5
SIZE_BINS = [("40-99", 40, 99, 20), ("100-199", 100, 199, 30),
             ("200-349", 200, 349, 30), ("350-500", 350, 500, 20)]
FOLDS = ["all-alpha", "all-beta", "alpha/beta", "alpha+beta"]
# keyword classes dominated by close homologs, non-natural or membrane proteins
EXCLUDE_KEYWORDS = ("IMMUNE SYSTEM", "DE NOVO", "ANTIBODY", "VIRUS", "UNKNOWN FUNCTION",
                    "MEMBRANE PROTEIN", "TOXIN")


def t(attr, op, value):
    return {"type": "terminal", "service": "text",
            "parameters": {"attribute": attr, "operator": op, "value": value}}


def search_representatives():
    nodes = [
        t("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
        t("rcsb_entry_info.resolution_combined", "less_or_equal", MAX_RES),
        t("rcsb_entry_info.deposited_polymer_entity_instance_count", "equals", 1),
        t("rcsb_entry_info.polymer_entity_count_protein", "equals", 1),
        t("entity_poly.rcsb_sample_sequence_length", "range",
          {"from": 40, "to": 500, "include_lower": True, "include_upper": True}),
    ]
    ids, start = [], 0
    while True:
        q = {"query": {"type": "group", "logical_operator": "and", "nodes": nodes},
             "return_type": "polymer_entity",
             "request_options": {"paginate": {"start": start, "rows": 1000},
                                 "group_by": {"aggregation_method": "sequence_identity",
                                              "similarity_cutoff": 30},
                                 "group_by_return_type": "representatives"}}
        res = bu.post_json(bu.SEARCH_URL, q)
        ids += [r["identifier"] for r in res["result_set"]]
        start += 1000
        if start >= res["group_by_count"]:
            return ids


GQL = """query($ids:[String!]!){ polymer_entities(entity_ids:$ids){
 rcsb_id
 entity_poly{rcsb_sample_sequence_length}
 rcsb_polymer_entity{pdbx_description rcsb_ec_lineage{id}}
 rcsb_polymer_entity_container_identifiers{auth_asym_ids}
 rcsb_entity_source_organism{ncbi_scientific_name}
 entry{ exptl{method} struct_keywords{pdbx_keywords} refine{ls_R_factor_R_free}
  rcsb_entry_info{resolution_combined deposited_polymer_monomer_count}
  assemblies{rcsb_struct_symmetry{oligomeric_state kind}} }
}}"""


def fetch_metadata(entity_ids):
    out = []
    for i in range(0, len(entity_ids), 100):
        res = bu.post_json(bu.GRAPHQL_URL, {"query": GQL, "variables": {"ids": entity_ids[i:i + 100]}})
        out += res["data"]["polymer_entities"] or []
        print(f"  metadata {min(i + 100, len(entity_ids))}/{len(entity_ids)}", end="\r")
    print()
    return out


def functional_class(keywords, has_ec):
    k = (keywords or "").upper()
    if has_ec or any(w in k for w in ("HYDROLASE", "TRANSFERASE", "OXIDOREDUCTASE", "LYASE",
                                      "ISOMERASE", "LIGASE", "KINASE", "PROTEASE", "SYNTHASE")):
        return "enzyme"
    if "TRANSPORT" in k or "ELECTRON" in k:
        return "transport"
    if any(w in k for w in ("SIGNAL", "CYTOKINE", "HORMONE", "GROWTH", "CELL CYCLE", "APOPTOSIS")):
        return "signaling"
    if any(w in k for w in ("STRUCTURAL", "CONTRACTILE", "ADHESION", "MOTOR", "CYTOSKELETON")):
        return "structural"
    if any(w in k for w in ("BINDING", "TRANSCRIPTION", "LECTIN", "CHAPERONE", "REGULATOR")):
        return "binding/regulatory"
    return "other"


def fold_class(h, s, par):
    if s < 0.05 and h >= 0.25:
        return "all-alpha"
    if h < 0.10 and s >= 0.20:
        return "all-beta"
    if h >= 0.10 and s >= 0.10:
        return "alpha/beta" if par >= 0.5 else "alpha+beta"
    return "other/low-SS"


def build_candidates(meta):
    rows, skipped = [], Counter()
    for m in meta:
        e = m["entry"]
        pdb_id, _ = m["rcsb_id"].split("_")
        kw = (e["struct_keywords"] or {}).get("pdbx_keywords") or ""
        if any(x in kw.upper() for x in EXCLUDE_KEYWORDS):
            skipped["excluded keyword class"] += 1
            continue
        states = [s["oligomeric_state"] for a in (e["assemblies"] or [])[:1]
                  for s in (a["rcsb_struct_symmetry"] or [])]
        if states and states[0] != "Monomer":
            skipped["assembly not monomeric"] += 1
            continue
        chains = m["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"]
        if len(chains) != 1:
            skipped["not a single chain"] += 1
            continue
        path = bu.download_pdb(pdb_id)
        if path is None:
            skipped["no PDB-format file"] += 1
            continue
        st = bu.chain_stats(path, chains[0])
        res = min(e["rcsb_entry_info"]["resolution_combined"])
        seqlen = m["entity_poly"]["rcsb_sample_sequence_length"]
        if st["n_ca"] < 40 or st["n_ca"] > 500:
            skipped["observed residues outside 40-500"] += 1
            continue
        if st["n_ca"] / max(seqlen, 1) < 0.90:
            skipped["< 90% of sequence observed"] += 1
            continue
        if st["max_gap"] > 5 or st["n_gap_residues"] > 0.05 * st["n_ca"]:
            skipped["internal gaps (>5 consecutive or >5% total)"] += 1
            continue
        if st["b_std"] < 1e-6:
            skipped["constant B-factors"] += 1
            continue
        ec_lin = (m["rcsb_polymer_entity"] or {}).get("rcsb_ec_lineage") or []
        ec = [x["id"] for x in ec_lin]
        org = ((m.get("rcsb_entity_source_organism") or [{}])[0] or {}).get("ncbi_scientific_name") or ""
        rfree = next((r["ls_R_factor_R_free"] for r in (e["refine"] or [])
                      if r.get("ls_R_factor_R_free")), None)
        rows.append({
            "PDB_ID": pdb_id, "chain": chains[0],
            "protein_name": m["rcsb_polymer_entity"]["pdbx_description"],
            "organism": org, "keywords": kw,
            "residue_count": st["n_ca"], "seqres_length": seqlen,
            "n_missing_internal": st["n_gap_residues"],
            "resolution": res, "experimental_method": e["exptl"][0]["method"],
            "r_free": rfree,
            "fold_class": fold_class(st["helix_frac"], st["sheet_frac"], st["parallel_frac"]),
            "helix_frac": round(st["helix_frac"], 3),
            "sheet_frac": round(st["sheet_frac"], 3),
            "functional_class": functional_class(kw, bool(ec)),
            "ec_number": ec[-1] if ec else "",
            "mean_bfactor": round(st["mean_b"], 2),
        })
    print("Candidates passing filters:", len(rows))
    for k, v in skipped.most_common():
        print(f"  dropped: {v:4d}  {k}")
    return pd.DataFrame(rows)


def select(df, rng):
    """Stratified greedy pick: fill each size bin round-robin across fold
    classes; within a fold class prefer the least-represented functional class
    and resolution band, ties broken by the seeded shuffle order."""
    df = df[df.fold_class.isin(FOLDS)].copy()
    df["size_bin"] = None
    for name, lo, hi, _ in SIZE_BINS:
        df.loc[df.residue_count.between(lo, hi), "size_bin"] = name
    df["res_bin"] = pd.cut(df.resolution, [0, 1.5, 2.0, 2.5], labels=["<=1.5", "1.5-2.0", "2.0-2.5"]).astype(str)
    chosen, fn_count, res_count, org_count = [], Counter(), Counter(), Counter()
    for name, _, _, quota in SIZE_BINS:
        pools = {f: df[(df.size_bin == name) & (df.fold_class == f)].sample(
                     frac=1, random_state=rng.randrange(10 ** 6)) for f in FOLDS}
        picked, guard = 0, 0
        while picked < quota and guard < 10 * quota:
            for f in FOLDS:
                guard += 1
                pool = pools[f]
                if picked >= quota or pool.empty:
                    continue
                score = pool.apply(lambda r: fn_count[r.functional_class] + res_count[r.res_bin]
                                   + 3 * (org_count[r.organism] > 0), axis=1)
                idx = score.idxmin()  # first minimum == seeded-random among ties
                r = pool.loc[idx]
                chosen.append(r)
                fn_count[r.functional_class] += 1
                res_count[r.res_bin] += 1
                org_count[r.organism] += 1
                pools[f] = pool.drop(idx)
                picked += 1
    out = pd.DataFrame(chosen)
    if len(out) < N_TARGET:  # top up from leftovers if some strata were thin
        rest = df[~df.PDB_ID.isin(out.PDB_ID)].sample(frac=1, random_state=SEED)
        out = pd.concat([out, rest.head(N_TARGET - len(out))])
    return out.drop(columns=["res_bin"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(bu.HERE, "protein_benchmark.csv"))
    args = ap.parse_args()
    rng = random.Random(SEED)

    print("1/4 searching RCSB for cluster representatives ...")
    reps = bu.cached_json("search_representatives", search_representatives)
    print(f"  {len(reps)} cluster representatives (30% identity, X-ray, <= {MAX_RES} A)")
    sample = sorted(reps)
    rng.shuffle(sample)
    sample = sample[:POOL_SAMPLE]

    print("2/4 fetching metadata ...")
    meta = bu.cached_json("metadata_sample", lambda: fetch_metadata(sample))

    print("3/4 downloading structures and applying quality filters ...")
    cand = build_candidates(meta)
    cand.to_csv(bu.lp(os.path.join(bu.CACHE_DIR, "candidates.csv")), index=False)

    print("4/4 stratified selection ...")
    sel = select(cand, rng)
    sel = sel.sort_values(["residue_count", "PDB_ID"]).reset_index(drop=True)
    sel.to_csv(bu.lp(args.out), index=False)
    print(f"Wrote {len(sel)} proteins to {args.out}")
    for col in ("size_bin", "fold_class", "functional_class"):
        print(sel[col].value_counts().to_string(), "\n")


if __name__ == "__main__":
    main()
