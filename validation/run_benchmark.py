"""Run the Calpha ANM over the benchmark list and produce every output.

    python validation/run_benchmark.py

Reads   validation/protein_benchmark.csv   (made by build_benchmark.py)
Writes  validation/results_100_proteins.csv     per-protein results (failures kept)
        validation/summary_statistics.json      headline statistics
        validation/association_analysis.csv     r vs protein properties
        validation/group_analysis.csv           r by fold / functional class
        validation/figures/*.png                figures
        validation/VALIDATION_REPORT.md         report (numbers filled from the data)

The same ANM parameters are used for every protein (see ANM_PARAMS); nothing
is tuned per protein. PDB files are cached in validation/cache/pdb/.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import bench_utils as bu  # noqa: E402
from anm import run_anm_pipeline  # noqa: E402

SEED = 20260930
N_BOOT = 10000
FIG_DIR = os.path.join(HERE, "figures")
# Identical for every protein -- the library defaults, stated explicitly.
ANM_PARAMS = dict(cutoff=8.0, gamma=1.0, n_modes=20, n_keep=None, tol=1e-6)
N_KEEP_EFFECTIVE = ANM_PARAMS["n_modes"] - 6  # pipeline default n_keep = n_modes - 6
# Secondary configuration: identical except MSF uses every non-zero mode
# (dense diagonalization). Its results go to separate files; the 14-mode
# results above are never overwritten by an all-modes run.
ALL_MODES_PARAMS = dict(ANM_PARAMS, n_modes="all")
RESULTS_FILE = "results_100_proteins.csv"
RESULTS_ALL_FILE = "results_100_proteins_all_modes.csv"
SUMMARY_ALL_FILE = "summary_statistics_all_modes.json"


# --------------------------------------------------------------------------
# 1. Run ANM on every protein
# --------------------------------------------------------------------------
def analyze_one(row, params=ANM_PARAMS):
    """Run the pipeline on one benchmark row. Always returns a dict; on any
    failure `status` is 'failed' and `failure_reason` says why."""
    out = {"PDB_ID": row.PDB_ID, "chain": row.chain, "protein_name": row.protein_name,
           "resolution": row.resolution, "status": "failed", "failure_reason": "",
           "warnings": ""}
    warns = []
    try:
        path = bu.download_pdb(row.PDB_ID)
        if path is None:
            raise RuntimeError("no PDB-format file available from RCSB")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = run_anm_pipeline(path, chain_id=row.chain, report_path=None, **params)
        n = len(res.msf)
        out["n_residues"] = n
        out["connected"] = bool(res.connected)
        if n != row.residue_count:
            warns.append(f"residue count {n} != benchmark list {row.residue_count}")

        msf, bf = np.asarray(res.msf, float), np.asarray(res.bfactors, float)
        ok = np.isfinite(msf) & np.isfinite(bf)        # same mask applied to BOTH arrays
        out["n_usable_residues"] = int(ok.sum())
        if ok.sum() < n:
            warns.append(f"{n - ok.sum()} residues with non-finite MSF/B dropped")
        if ok.sum() < 10:
            raise RuntimeError("fewer than 10 usable residues")
        if np.ptp(bf[ok]) == 0:
            raise RuntimeError("constant B-factors; correlation undefined")
        pr, sr = stats.pearsonr(msf[ok], bf[ok]), stats.spearmanr(msf[ok], bf[ok])
        # cross-check against the repository's own validate() output
        if ok.all() and not np.isclose(pr[0], res.correlation[0][0], atol=1e-9):
            raise RuntimeError("internal inconsistency: Pearson differs from pipeline validate()")
        out.update(pearson_r=float(pr[0]), pearson_p=float(pr[1]),
                   spearman_rho=float(sr[0]), spearman_p=float(sr[1]))
        out["lead_mode_collectivity"] = float(res.collectivity[0])
        out["mean_collectivity_kept"] = float(np.mean(res.collectivity))
        out["n_modes_kept"] = int(len(res.kept_vals))
        out["n_zero_modes"] = int(np.sum(res.eigvals <= params["tol"]))
        out["lowest_kept_eigenvalue"] = float(res.kept_vals[0])
        if out["n_zero_modes"] != 6:
            warns.append(f"{out['n_zero_modes']} near-zero modes (6 rigid-body + "
                         f"{out['n_zero_modes'] - 6} floppy, all excluded)")
        if not res.kept_vals.min() > params["tol"]:
            raise RuntimeError("a rigid-body mode leaked into the kept modes")
        out["mean_bfactor_obs"] = float(bf[ok].mean())
        out["status"] = "ok"
    except Exception as exc:  # recorded, never silently dropped
        out["failure_reason"] = f"{type(exc).__name__}: {exc}"
    out["warnings"] = "; ".join(warns)
    return out


def run_all(bench, limit=None, params=ANM_PARAMS):
    rows = []
    for i, row in enumerate(bench.itertuples(index=False), 1):
        if limit and i > limit:
            break
        r = analyze_one(row, params)
        tag = f"r={r['pearson_r']:+.3f}" if r["status"] == "ok" else f"FAILED ({r['failure_reason']})"
        print(f"[{i:3d}/{len(bench)}] {row.PDB_ID}_{row.chain}  N={row.residue_count:<4d} {tag}")
        rows.append(r)
    res = pd.DataFrame(rows)
    extra = bench[["PDB_ID", "chain", "residue_count", "fold_class", "functional_class",
                   "helix_frac", "sheet_frac", "organism", "r_free"]]
    return res.merge(extra, on=["PDB_ID", "chain"], how="left")


# --------------------------------------------------------------------------
# 2. Statistics
# --------------------------------------------------------------------------
def boot_ci(x, fn=np.mean, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    bs = np.array([fn(x[rng.integers(0, len(x), len(x))]) for _ in range(N_BOOT)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def describe(x):
    x = np.asarray(x, float)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    lo, hi = boot_ci(x)
    return {"N": int(len(x)), "mean": float(x.mean()), "median": float(med),
            "sd": float(x.std(ddof=1)), "min": float(x.min()), "max": float(x.max()),
            "q25": float(q1), "q75": float(q3), "iqr": float(q3 - q1),
            "ci95_mean_low": lo, "ci95_mean_high": hi,
            "ci95_median_low": boot_ci(x, np.median)[0], "ci95_median_high": boot_ci(x, np.median)[1]}


def summarize(ok):
    r = ok.pearson_r.values
    summ = {"pearson": describe(r), "spearman": describe(ok.spearman_rho.values),
            "pct_pearson_gt_0": float(100 * np.mean(r > 0))}
    for t in (0.3, 0.5, 0.7):
        summ[f"pct_pearson_ge_{t}"] = float(100 * np.mean(r >= t))
    summ["pct_pearson_p_lt_0.05_and_positive"] = float(
        100 * np.mean((ok.pearson_p.values < 0.05) & (r > 0)))
    return summ


def build_summ(results, ok, params, n_modes_kept, qc):
    summ = summarize(ok)
    summ.update(n_attempted=int(len(results)), n_valid=int(len(ok)),
                anm_params=params, n_modes_kept=n_modes_kept,
                weakest=ok.loc[ok.pearson_r.idxmin(), ["PDB_ID", "chain", "protein_name", "pearson_r"]].to_dict(),
                strongest=ok.loc[ok.pearson_r.idxmax(), ["PDB_ID", "chain", "protein_name", "pearson_r"]].to_dict(),
                qc=qc)
    return summ


def association_table(ok):
    preds = {"residue_count": "Residue count", "resolution": "Resolution (A)",
             "mean_bfactor_obs": "Mean experimental B-factor (A^2)",
             "helix_frac": "Helix fraction", "sheet_frac": "Sheet fraction",
             "lead_mode_collectivity": "Lead-mode collectivity"}
    rows = []
    for col, label in preds.items():
        d = ok[[col, "pearson_r"]].dropna()
        pr, sr = stats.pearsonr(d[col], d.pearson_r), stats.spearmanr(d[col], d.pearson_r)
        rows.append({"variable": label, "n": len(d), "pearson_r_vs_anm_r": pr[0], "pearson_p": pr[1],
                     "spearman_rho_vs_anm_r": sr[0], "spearman_p": sr[1]})
    return pd.DataFrame(rows)


def group_table(ok):
    rows, tests = [], {}
    for col in ("fold_class", "functional_class"):
        groups = {g: d.pearson_r.values for g, d in ok.groupby(col)}
        for g, v in groups.items():
            rows.append({"grouping": col, "group": g, "n": len(v), "mean_r": v.mean(),
                         "median_r": np.median(v), "sd_r": v.std(ddof=1) if len(v) > 1 else np.nan})
        big = [v for v in groups.values() if len(v) >= 3]
        tests[col] = (float(stats.kruskal(*big).pvalue) if len(big) > 1 else None)
    return pd.DataFrame(rows), tests


# --------------------------------------------------------------------------
# 3. Figures
# --------------------------------------------------------------------------
def make_figures(ok, res_all, bench, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(bu.lp(FIG_DIR), exist_ok=True)
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelsize": 12, "axes.titlesize": 13, "font.family": "DejaVu Sans"})
    BLUE, ORANGE, GREY, RED = "#2a6fbb", "#e08a1e", "#7a7a7a", "#c0392b"
    made = []

    def save(fig, name):
        fig.tight_layout()
        p = bu.lp(os.path.join(FIG_DIR, name))
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    r, rho, n = ok.pearson_r.values, ok.spearman_rho.values, len(ok)
    P, S = summ["pearson"], summ["spearman"]

    def hist(x, label, name, color, st):
        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        ax.hist(x, bins=np.linspace(-1, 1, 41), color=color, edgecolor="white")
        ax.axvline(0, color="black", lw=0.8)
        ax.axvline(st["mean"], color=RED, ls="--", lw=1.6, label=f"mean = {st['mean']:.2f}")
        ax.axvline(st["median"], color="black", ls=":", lw=1.6, label=f"median = {st['median']:.2f}")
        ax.set_xlabel(label); ax.set_ylabel("Number of proteins")
        ax.set_title(f"{label.split(' (')[0]} across {n} proteins")
        ax.legend(frameon=False)
        save(fig, name)

    hist(r, "Pearson r (ANM MSF vs B-factor)", "fig01_pearson_histogram.png", BLUE, P)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    data = [r, rho]
    vp = ax.violinplot(data, showextrema=False, widths=0.8)
    for b, c in zip(vp["bodies"], (BLUE, ORANGE)):
        b.set_facecolor(c); b.set_alpha(0.35)
    ax.boxplot(data, widths=0.22, showfliers=False, medianprops=dict(color="black"))
    rng = np.random.default_rng(SEED)
    for i, d in enumerate(data, 1):
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(d)), d, s=9, color=GREY, alpha=0.6, zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Pearson r", "Spearman rho"])
    ax.set_ylabel("Correlation with experimental B-factors")
    ax.set_title("Distribution of per-protein correlations")
    save(fig, "fig02_pearson_spearman_box_violin.png")

    hist(rho, "Spearman rho (ANM MSF vs B-factor)", "fig03_spearman_histogram.png", ORANGE, S)

    def scatter(x, xlabel, name, logx=False):
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        ax.scatter(x, r, s=26, color=BLUE, alpha=0.75, edgecolor="white", lw=0.4)
        ax.axhline(0, color="black", lw=0.8)
        if logx:
            ax.set_xscale("log")
        rs = stats.spearmanr(x, r)
        ax.set_xlabel(xlabel); ax.set_ylabel("Pearson r")
        ax.set_title(f"Spearman rho = {rs[0]:+.2f} (p = {rs[1]:.2g}, n = {len(x)})", fontsize=11)
        save(fig, name)

    scatter(ok.residue_count.values, "Residue count (chain)", "fig04_r_vs_size.png", logx=True)
    scatter(ok.resolution.values, "X-ray resolution (A)", "fig05_r_vs_resolution.png")
    scatter(ok.lead_mode_collectivity.values, "Lead-mode collectivity (kappa)", "fig06_r_vs_collectivity.png")

    order = ok.sort_values("pearson_r").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(range(len(order)), order.pearson_r, color=[BLUE if v >= 0 else RED for v in order.pearson_r])
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(P["median"], color="black", ls=":", lw=1, label=f"median {P['median']:.2f}")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{a}_{b}" for a, b in zip(order.PDB_ID, order.chain)], rotation=90, fontsize=6.5)
    ax.set_xlim(-1, len(order)); ax.set_ylabel("Pearson r")
    ax.set_title(f"Pearson r for every analysed protein (n = {len(order)}, sorted)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig07_sorted_pearson_bar.png")

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.scatter(r, rho, s=28, color=BLUE, alpha=0.75, edgecolor="white", lw=0.4)
    ax.plot([-1, 1], [-1, 1], color=GREY, lw=1, ls="--")
    ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="black", lw=0.6)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
    ax.set_xlabel("Pearson r"); ax.set_ylabel("Spearman rho")
    ax.set_title(f"Pearson vs Spearman (agreement r = {stats.pearsonr(r, rho)[0]:.2f})", fontsize=11)
    save(fig, "fig08_pearson_vs_spearman.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    sizes = bench.residue_count.values
    ax.hist(sizes, bins=np.arange(40, 541, 40), color=GREY, edgecolor="white")
    ax.set_xlabel("Residues per analysed chain"); ax.set_ylabel("Number of proteins")
    ax.set_title(f"Protein sizes in the benchmark (n = {len(sizes)}, median {int(np.median(sizes))})")
    save(fig, "fig09_size_distribution.png")

    # case studies: best, nearest-to-median, worst
    med_idx = (ok.pearson_r - P["median"]).abs().idxmin()
    picks = [("Highest r", ok.pearson_r.idxmax()), ("Near-median r", med_idx), ("Lowest r", ok.pearson_r.idxmin())]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for ax, (lab, idx) in zip(axes, picks):
        row = ok.loc[idx]
        path = bu.download_pdb(row.PDB_ID)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = run_anm_pipeline(path, chain_id=row.chain, report_path=None, **ANM_PARAMS)
        x = np.arange(len(res.msf))
        resnum = [l[1] for l in res.labels]
        ax.plot(x, res.bfactors * 3 / (8 * np.pi ** 2), color=BLUE, lw=1.6, label="Experimental (from B-factor)")
        ax.plot(x, res.scale * res.msf, color=ORANGE, lw=1.6, label="ANM prediction (least-squares scaled)")
        step = max(1, len(x) // 10)
        ax.set_xticks(x[::step]); ax.set_xticklabels([resnum[i] for i in x[::step]])
        ax.set_xlabel("Residue number"); ax.set_ylabel("MSF (A^2)")
        ax.set_title(f"{lab}: {row.PDB_ID}_{row.chain} ({str(row.protein_name).title()[:45]}), "
                     f"N = {int(row.n_residues)}, r = {row.pearson_r:.2f}, rho = {row.spearman_rho:.2f}", fontsize=11)
        ax.legend(frameon=False, loc="upper right", fontsize=9)
    save(fig, "fig10_case_studies.png")

    # extras: r vs mean B, r by fold class
    scatter(ok.mean_bfactor_obs.values, "Mean experimental B-factor (A^2)", "fig11_r_vs_mean_bfactor.png")
    fig, ax = plt.subplots(figsize=(7, 4.6))
    folds = [f for f in ["all-alpha", "all-beta", "alpha/beta", "alpha+beta"] if f in set(ok.fold_class)]
    ax.boxplot([ok[ok.fold_class == f].pearson_r for f in folds], showfliers=True)
    ax.set_xticklabels([f"{f}\n(n={int((ok.fold_class == f).sum())})" for f in folds])
    ax.axhline(0, color="black", lw=0.8); ax.set_ylabel("Pearson r")
    ax.set_title("Pearson r by (heuristic) fold class")
    save(fig, "fig12_r_by_fold_class.png")
    return made


# --------------------------------------------------------------------------
# 4. Quality-control checks (assertions on the final outputs)
# --------------------------------------------------------------------------
def quality_control(bench, results):
    qc = []
    assert len(bench) == bench[["PDB_ID", "chain"]].drop_duplicates().shape[0], "duplicate PDB-chain pairs"
    qc.append("no duplicate PDB-chain pairs")
    assert bench.PDB_ID.is_unique, "duplicate PDB IDs"
    qc.append("no duplicate PDB IDs")
    assert set(results.PDB_ID) == set(bench.PDB_ID), "some proteins missing from results"
    qc.append(f"all {len(bench)} benchmark proteins present in results (failures included)")
    ok = results[results.status == "ok"]
    num = ok[["pearson_r", "spearman_rho", "lead_mode_collectivity"]]
    assert np.isfinite(num.values).all(), "NaN/inf in valid results"
    qc.append("no NaN/inf in pearson_r, spearman_rho, lead_mode_collectivity of valid proteins")
    assert (ok.n_zero_modes >= 6).all(), "fewer than 6 rigid-body modes removed"
    assert (ok.lowest_kept_eigenvalue > ANM_PARAMS["tol"]).all(), "rigid-body mode kept"
    n_extra = int((ok.n_zero_modes > 6).sum())
    qc.append(f"6 rigid-body modes removed in all {len(ok)} valid proteins; {n_extra} have extra "
              "near-zero (floppy-terminus) modes, also excluded and flagged in `warnings`")
    return qc


def compare_configs(ok14, ok_all, assoc14, assoc_all):
    """Paired comparison of the two configurations over proteins valid in both."""
    m = ok14.merge(ok_all, on=["PDB_ID", "chain"], suffixes=("_14", "_all"))
    out = {"n_paired": int(len(m))}
    for key, col in (("pearson", "pearson_r"), ("spearman", "spearman_rho")):
        d = (m[col + "_all"] - m[col + "_14"]).values
        out[key] = {"mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
                    "n_all_higher": int((d > 0).sum()), "n_all_lower": int((d < 0).sum()),
                    "wilcoxon_p": float(stats.wilcoxon(d).pvalue)}
    out["lead_collectivity_max_abs_diff"] = float(
        (m.lead_mode_collectivity_all - m.lead_mode_collectivity_14).abs().max())
    for tag, a in (("14", assoc14), ("all", assoc_all)):
        row = a.set_index("variable").loc["Lead-mode collectivity"]
        out[f"collectivity_{tag}"] = {"spearman_rho": float(row.spearman_rho_vs_anm_r),
                                      "spearman_p": float(row.spearman_p),
                                      "pearson_r": float(row.pearson_r_vs_anm_r),
                                      "pearson_p": float(row.pearson_p)}
    return out


def load_all_modes_ctx(bench, ok14, assoc14):
    """Context for the all-modes configuration, read from its separate results
    file; None if that file doesn't exist (all-modes never run)."""
    path = bu.lp(os.path.join(HERE, RESULTS_ALL_FILE))
    if not os.path.exists(path):
        return None
    results = pd.read_csv(path, dtype={"chain": str})
    ok = results[results.status == "ok"].reset_index(drop=True)
    summ = build_summ(results, ok, ALL_MODES_PARAMS, None, quality_control(bench, results))
    summ["n_modes_kept_min"] = int(ok.n_modes_kept.min())
    summ["n_modes_kept_median"] = float(ok.n_modes_kept.median())
    summ["n_modes_kept_max"] = int(ok.n_modes_kept.max())
    assoc = association_table(ok)
    return {"results": results, "ok": ok, "summ": summ, "assoc": assoc,
            "cmp": compare_configs(ok14, ok, assoc14, assoc)}


def print_comparison(summ14, ctx):
    s14, sa, c = summ14, ctx["summ"], ctx["cmp"]
    print("\n=== 14 kept modes vs all non-zero modes ===")
    print(f"valid proteins: {s14['n_valid']} (14 modes), {sa['n_valid']} (all modes); paired: {c['n_paired']}")
    print(f"{'':22s}{'14 modes':>10s}{'all modes':>11s}")
    for lab, k, f in (("Pearson r mean", "pearson", "mean"), ("Pearson r median", "pearson", "median"),
                      ("Spearman rho mean", "spearman", "mean"), ("Spearman rho median", "spearman", "median")):
        print(f"{lab:22s}{s14[k][f]:10.3f}{sa[k][f]:11.3f}")
    for k in ("pearson", "spearman"):
        d = c[k]
        print(f"paired {k}: mean diff (all - 14) {d['mean_diff']:+.3f}, all higher in {d['n_all_higher']}, "
              f"lower in {d['n_all_lower']}, Wilcoxon p = {d['wilcoxon_p']:.3g}")
    for tag in ("14", "all"):
        d = c[f"collectivity_{tag}"]
        print(f"lead-mode collectivity vs Pearson r ({tag}): Spearman rho = {d['spearman_rho']:+.3f}, "
              f"p = {d['spearman_p']:.3g}; Pearson = {d['pearson_r']:+.3f}, p = {d['pearson_p']:.3g}")


def main_all_modes(args, bench):
    """--n-modes all: run the all-modes configuration into its own files, then
    regenerate the report from the untouched 14-mode results plus these."""
    res_all_path = bu.lp(os.path.join(HERE, RESULTS_ALL_FILE))
    if args.from_results:
        results = pd.read_csv(res_all_path, dtype={"chain": str})
    else:
        results = run_all(bench, args.limit, ALL_MODES_PARAMS)
    if args.limit:
        bench = bench[bench.PDB_ID.isin(results.PDB_ID)]
    results.to_csv(res_all_path, index=False)
    res14 = pd.read_csv(bu.lp(os.path.join(HERE, RESULTS_FILE)), dtype={"chain": str})
    ok14 = res14[res14.status == "ok"].reset_index(drop=True)
    summ14 = build_summ(res14, ok14, ANM_PARAMS, N_KEEP_EFFECTIVE, quality_control(bench, res14))
    assoc14 = association_table(ok14)
    groups14, tests14 = group_table(ok14)
    summ14["group_kruskal_p"] = tests14
    ctx = load_all_modes_ctx(bench, ok14, assoc14)
    with open(bu.lp(os.path.join(HERE, SUMMARY_ALL_FILE)), "w") as fh:
        json.dump({"summary": ctx["summ"], "comparison_vs_14_modes": ctx["cmp"]}, fh, indent=2)
    print(f"\n{ctx['summ']['n_valid']}/{len(results)} analysed successfully in all-modes configuration")
    if not args.no_report:
        from write_report import write_report, FIG_CAPTIONS
        write_report(bench, res14, ok14, summ14, assoc14, groups14, list(FIG_CAPTIONS), ctx)
    print_comparison(summ14, ctx)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=os.path.join(HERE, "protein_benchmark.csv"))
    ap.add_argument("--limit", type=int, default=None, help="only run the first N (debugging)")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--from-results", action="store_true",
                    help="reuse results_100_proteins.csv instead of re-running the ANM")
    ap.add_argument("--n-modes", choices=["20", "all"], default="20",
                    help="'20' (default): the 14-kept-mode benchmark. 'all': use every non-zero mode; "
                         "writes results_100_proteins_all_modes.csv and leaves the 14-mode results untouched")
    args = ap.parse_args()
    np.random.seed(SEED)

    bench = pd.read_csv(bu.lp(args.benchmark), dtype={"chain": str})
    if args.n_modes == "all":
        return main_all_modes(args, bench)
    res_path = bu.lp(os.path.join(HERE, RESULTS_FILE))
    if args.from_results:
        results = pd.read_csv(res_path, dtype={"chain": str})
    else:
        results = run_all(bench, args.limit)
    if args.limit:
        bench = bench[bench.PDB_ID.isin(results.PDB_ID)]
    results.to_csv(res_path, index=False)  # save before QC
    qc = quality_control(bench, results)

    ok = results[results.status == "ok"].reset_index(drop=True)
    print(f"\n{len(ok)}/{len(results)} analysed successfully; {len(results) - len(ok)} failed")
    if ok.empty:
        sys.exit("no valid proteins; nothing to summarise")
    summ = build_summ(results, ok, ANM_PARAMS, N_KEEP_EFFECTIVE, qc)
    assoc = association_table(ok)
    groups, group_tests = group_table(ok)
    summ["group_kruskal_p"] = group_tests
    assoc.to_csv(bu.lp(os.path.join(HERE, "association_analysis.csv")), index=False)
    groups.to_csv(bu.lp(os.path.join(HERE, "group_analysis.csv")), index=False)
    with open(bu.lp(os.path.join(HERE, "summary_statistics.json")), "w") as fh:
        json.dump(summ, fh, indent=2)

    figs = make_figures(ok, results, bench, summ)
    print("figures:", ", ".join(figs))
    if not args.no_report:
        from write_report import write_report
        ctx = None if args.limit else load_all_modes_ctx(bench, ok, assoc)
        write_report(bench, results, ok, summ, assoc, groups, figs, ctx)
    print(json.dumps({k: summ[k] for k in ("n_attempted", "n_valid")}, indent=1))
    print(json.dumps(summ["pearson"], indent=1))


if __name__ == "__main__":
    main()
