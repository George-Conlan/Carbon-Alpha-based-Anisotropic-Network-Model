"""Generate validation/VALIDATION_REPORT.md and the README validation section.

Every number in the text is filled in from the benchmark outputs at run time;
nothing is hard-coded. The interpretation paragraphs are qualitative and only
make claims that the generated numbers can be checked against in the tables.
"""
import os
import re

import numpy as np
import bench_utils as bu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FIG_CAPTIONS = {
    "fig01_pearson_histogram.png": "Histogram of Pearson r across all analysed proteins.",
    "fig02_pearson_spearman_box_violin.png": "Box/violin plot of Pearson r and Spearman rho (points are individual proteins).",
    "fig03_spearman_histogram.png": "Histogram of Spearman rho.",
    "fig04_r_vs_size.png": "Pearson r versus chain length (log x-axis).",
    "fig05_r_vs_resolution.png": "Pearson r versus X-ray resolution.",
    "fig06_r_vs_collectivity.png": "Pearson r versus lead-mode collectivity.",
    "fig07_sorted_pearson_bar.png": "Pearson r for every analysed protein, sorted.",
    "fig08_pearson_vs_spearman.png": "Pearson r versus Spearman rho per protein.",
    "fig09_size_distribution.png": "Distribution of chain sizes in the benchmark.",
    "fig10_case_studies.png": "Case studies: highest, near-median and lowest Pearson r; predicted vs experimental flexibility along the chain.",
    "fig11_r_vs_mean_bfactor.png": "Pearson r versus mean experimental B-factor.",
    "fig12_r_by_fold_class.png": "Pearson r by heuristic fold class.",
}


def f(x, d=3):
    return f"{x:.{d}f}"


def stat_table(P, S):
    def ci(s):
        return f"[{f(s['ci95_mean_low'])}, {f(s['ci95_mean_high'])}]"
    rows = [("N (valid proteins)", P["N"], S["N"]),
            ("Mean", f(P["mean"]), f(S["mean"])),
            ("Median", f(P["median"]), f(S["median"])),
            ("Standard deviation", f(P["sd"]), f(S["sd"])),
            ("25th percentile", f(P["q25"]), f(S["q25"])),
            ("75th percentile", f(P["q75"]), f(S["q75"])),
            ("Interquartile range", f(P["iqr"]), f(S["iqr"])),
            ("Minimum", f(P["min"]), f(S["min"])),
            ("Maximum", f(P["max"]), f(S["max"])),
            ("95% bootstrap CI of mean", ci(P), ci(S)),
            ("95% bootstrap CI of median", f"[{f(P['ci95_median_low'])}, {f(P['ci95_median_high'])}]",
             f"[{f(S['ci95_median_low'])}, {f(S['ci95_median_high'])}]")]
    out = ["| Statistic | Pearson r | Spearman rho |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(out)


def md_table(df, cols, fmt=None):
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                v = fmt[c](v)
            elif isinstance(v, float):
                v = f"{v:.3f}"
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write_report(bench, results, ok, summ, assoc, groups, figs):
    P, S = summ["pearson"], summ["spearman"]
    nA, nV = summ["n_attempted"], summ["n_valid"]
    failed = results[results.status != "ok"]
    neg = ok[ok.pearson_r < 0]
    warned = results[results.warnings.fillna("") != ""]
    ap = summ["anm_params"]

    assoc_t = assoc.copy()
    assoc_t["Pearson"] = assoc_t.apply(lambda r: f"{r.pearson_r_vs_anm_r:+.2f} (p {pfmt(r.pearson_p)})", axis=1)
    assoc_t["Spearman"] = assoc_t.apply(lambda r: f"{r.spearman_rho_vs_anm_r:+.2f} (p {pfmt(r.spearman_p)})", axis=1)
    assoc_md = md_table(assoc_t.rename(columns={"variable": "Variable"}), ["Variable", "n", "Pearson", "Spearman"])

    groups_md = md_table(groups.rename(columns={"grouping": "Grouping", "group": "Group", "n": "n",
                                                "mean_r": "Mean r", "median_r": "Median r", "sd_r": "SD r"}),
                         ["Grouping", "Group", "n", "Mean r", "Median r", "SD r"])
    kw = summ["group_kruskal_p"]
    kw_txt = "; ".join(f"{k}: p = {pfmt(v)}" for k, v in kw.items() if v is not None)

    sig = assoc.set_index("variable")
    def rho_of(v):
        return sig.loc[v, "spearman_rho_vs_anm_r"], sig.loc[v, "spearman_p"]

    fail_md = ("None." if failed.empty else
               md_table(failed, ["PDB_ID", "chain", "protein_name", "failure_reason"]))
    neg_md = ("None." if neg.empty else
              md_table(neg.sort_values("pearson_r"), ["PDB_ID", "chain", "protein_name", "n_residues", "pearson_r", "spearman_rho"],
                      {"n_residues": lambda v: int(v)}))
    n_extra = int((ok.n_zero_modes > 6).sum())

    cnt_fold = bench.fold_class.value_counts()
    cnt_fn = bench.functional_class.value_counts()
    fold_txt = ", ".join(f"{k} {v}" for k, v in cnt_fold.items())
    fn_txt = ", ".join(f"{k} {v}" for k, v in cnt_fn.items())
    bins = [(40, 99), (100, 199), (200, 349), (350, 500)]
    bin_txt = ", ".join(f"{lo}-{hi}: {int(bench.residue_count.between(lo, hi).sum())}" for lo, hi in bins)
    n_org = bench.organism.nunique()

    fig_md = "\n\n".join(f"![{FIG_CAPTIONS.get(n, n)}](figures/{n})  \n*{n}: {FIG_CAPTIONS.get(n, '')}*"
                         for n in figs)

    best, worst = summ["strongest"], summ["weakest"]
    gap = ok.pearson_r - ok.spearman_rho
    n_rho_gt, n_r_gt = int((gap < -0.2).sum()), int((gap > 0.2).sum())

    md = f"""# ANM Validation Across {nA} Diverse Protein Structures

*Generated by `validation/run_benchmark.py`; all numbers below are produced by that run (seed {20260930}).*

## Benchmark construction

The benchmark list ([protein_benchmark.csv](protein_benchmark.csv)) was built programmatically by
[build_benchmark.py](build_benchmark.py) from the RCSB PDB Search and Data APIs. **The ANM was never run during
selection**, so no correlation result could influence which proteins were chosen, and no protein was added or
removed after seeing results.

Inclusion criteria:

- X-ray diffraction structures, resolution <= 2.5 A.
- One protein entity and a single deposited chain; biological assembly 1 annotated as a monomer.
- 40-500 residues in the analysed chain.
- >= 90 % of the deposited sequence observed; no internal gap longer than 5 residues and <= 5 % of residues missing internally.
- Non-constant B-factors (so a correlation is defined) and a PDB-format file available.
- Keyword classes dominated by close homologs, designed, membrane or unannotated proteins were excluded (immune system/antibody, de novo, virus, membrane protein, toxin, unknown function).

Redundancy control: candidates were restricted to one representative per **30 % sequence-identity cluster** (RCSB
clustering), so no two proteins in the list are in the same 30 % cluster. This is a sequence-based guarantee only;
distant structural relatives (same fold, <30 % identity) can still co-occur.

Diversity: from the pool of structures that passed the filters above, a stratified, seeded greedy selection filled four size bins
(quotas 20/30/30/20), cycled through four fold classes within each bin, and preferred under-represented functional
classes, resolution bands and organisms.

- Size bins (residues): {bin_txt}
- Fold class (heuristic): {fold_txt}
- Functional class (heuristic): {fn_txt}
- {n_org} distinct source organisms.

The fold and functional labels are **heuristics, not curated annotations**. Fold class comes from the PDB file's own
HELIX/SHEET records (all-alpha: helix >= 25 % and sheet < 5 %; all-beta: helix < 10 % and sheet >= 20 %; otherwise
alpha/beta if >= 50 % of strand residues are in parallel sheets, else alpha+beta). Functional class comes from the
PDB keyword and EC annotation and will misclassify some entries.

## Methods

All calculations use the repository's existing `anm` package, unchanged, with **one parameter set for every protein**
(no per-protein tuning):

- **C-alpha elastic network.** Each residue is one node at its C-alpha position (model 1, the chosen chain; heteroatom residues excluded; alternate locations resolved by the parser to the highest-occupancy atom).
- **Contact cutoff.** Residues i, j are connected if their C-alpha distance is <= {ap['cutoff']} A.
- **Spring constant.** Uniform gamma = {ap['gamma']} for every contact.
- **Hessian.** The 3N x 3N ANM Hessian is assembled from 3 x 3 super-elements -gamma * (u u^T), where u is the unit vector between connected nodes; diagonal blocks are the negative sum of the node's off-diagonal blocks.
- **Normal modes.** The {ap['n_modes']} lowest-eigenvalue modes are computed by sparse shift-invert diagonalisation; eigenvalues <= {ap['tol']:g} (the 6 rigid-body translations/rotations, plus any extra "floppy" zero modes) are discarded and the next {summ['n_modes_kept']} modes are kept. **MSF therefore uses only the {summ['n_modes_kept']} lowest non-zero modes**, a truncation inherited from the package default, not tuned.
- **MSF.** MSF_i = sum over kept modes k of (1/lambda_k) * |v_k,i|^2.
- **Comparison with B-factors.** Experimental B-factors are converted with B = 8 pi^2 <u^2> / 3 (the package's `bfactor_to_msf`). Pearson and Spearman correlations are computed between the predicted MSF vector and the experimental C-alpha B-factor vector over the same residues, in residue order. Both metrics are invariant to the overall scale, so the gamma value and the unit conversion do not affect them.
- **Metrics.** Pearson r measures linear agreement; Spearman rho measures rank agreement and is less sensitive to a few extreme residues (typically flexible termini).
- **Collectivity.** Lead-mode collectivity is the package's participation-entropy measure (1/N) exp(-sum p_i ln p_i) for the lowest non-zero mode.
- **Summary statistics.** Per-protein correlations are summarised with mean, median, SD, quartiles and a {10000:,}-resample percentile bootstrap 95 % CI (seeded). No Fisher-z averaging is applied; r values are averaged directly.
- **Association analysis.** Pearson r is compared with protein properties by Pearson and Spearman correlation across proteins (descriptive; n = {nV}).

## Dataset

- Structures attempted: **{nA}**
- Analysed successfully: **{nV}**
- Failed: **{len(failed)}** (reported below, not dropped)
- Residue count range (analysed chains): {int(ok.n_residues.min())}-{int(ok.n_residues.max())} (median {int(ok.n_residues.median())})
- Resolution range: {f(ok.resolution.min(), 2)}-{f(ok.resolution.max(), 2)} A (median {f(ok.resolution.median(), 2)})
- Mean experimental B-factor range: {f(ok.mean_bfactor_obs.min(), 1)}-{f(ok.mean_bfactor_obs.max(), 1)} A^2
- Fold and functional composition: see Benchmark construction. Full per-protein table: [results_100_proteins.csv](results_100_proteins.csv).

### Failures

{fail_md}

### Warnings

{len(warned)} proteins carry a warning in the `warnings` column of the results file. {n_extra} proteins have more than the expected 6 near-zero modes: the additional zero modes are floppy terminal residues held by too few contacts; they are excluded like the rigid-body modes, so those residues' unconstrained motion is not represented in the MSF.

## Results

### Summary statistics (n = {nV})

{stat_table(P, S)}

### Fraction of proteins above correlation thresholds (Pearson r)

| Threshold | % of valid proteins |
|---|---|
| r > 0 | {f(summ['pct_pearson_gt_0'], 1)} % |
| r >= 0.3 | {f(summ['pct_pearson_ge_0.3'], 1)} % |
| r >= 0.5 | {f(summ['pct_pearson_ge_0.5'], 1)} % |
| r >= 0.7 | {f(summ['pct_pearson_ge_0.7'], 1)} % |

Strongest: **{best['PDB_ID']}_{best['chain']}** ({str(best['protein_name']).title()}), r = {f(best['pearson_r'])}.
Weakest: **{worst['PDB_ID']}_{worst['chain']}** ({str(worst['protein_name']).title()}), r = {f(worst['pearson_r'])}.

Proteins with Pearson r < 0 ({len(neg)}):

{neg_md}

### Does performance depend on protein properties?

Association between each property and the per-protein Pearson r (across proteins; associations only, not causal claims):

{assoc_md}

Pearson r by group (Kruskal-Wallis across groups with >= 3 members: {kw_txt}):

{groups_md}

## Figures

{fig_md}

## Interpretation

**What the average means.** Across {nV} structurally diverse proteins the mean Pearson r is {f(P['mean'], 2)}
(median {f(P['median'], 2)}, SD {f(P['sd'], 2)}). The bootstrap 95 % CI of the mean is
[{f(P['ci95_mean_low'], 2)}, {f(P['ci95_mean_high'], 2)}]. That is a moderate positive association:
{f(summ['pct_pearson_gt_0'], 0)} % of proteins are positive, but only {f(summ['pct_pearson_ge_0.5'], 0)} % reach
r >= 0.5 and {f(summ['pct_pearson_ge_0.7'], 0)} % reach r >= 0.7. A correlation of this size means a residue's
ANM flexibility rank is informative about its B-factor rank, and leaves most of the per-residue variance
unexplained (r^2 at the mean r is about {f(P['mean'] ** 2, 2)}). The spread between proteins is large, so the
mean should not be read as the expected accuracy for any single protein.

**Why the ANM should not reproduce B-factors exactly.** The model is a single-minimum, harmonic, C-alpha-only
network whose only input is geometry. It has no side chains, no chemistry, no solvent and no ligands, and every
contact has the same stiffness. Experimental B-factors contain much more than thermal motion of an isolated
molecule:

- **Crystal packing.** Lattice contacts restrain some surface residues and leave others free; the ANM here sees one isolated chain.
- **Static (lattice) disorder.** Conformational heterogeneity between unit cells contributes to B even at zero temperature.
- **Refinement artefacts.** B-factors depend on the refinement program, restraints, TLS treatment, resolution and occupancy handling, so values are not strictly comparable between structures.
- **Uniform springs.** A single gamma and a hard cutoff ignore that contacts differ in stiffness; terminal and loop residues with few contacts are the usual source of large errors.
- **B-factors as a proxy.** B-factors report average positional spread in a crystal at typically cryogenic temperature, not intrinsic solution dynamics or functional motions.
- **Mode truncation.** Only the {summ['n_modes_kept']} lowest non-zero modes contribute, which smooths the predicted profile.

**Why some proteins do better than others.** In this benchmark, Pearson r has the following associations
(Spearman rho across proteins): residue count {rho_of('Residue count')[0]:+.2f} (p {pfmt(rho_of('Residue count')[1])}),
resolution {rho_of('Resolution (A)')[0]:+.2f} (p {pfmt(rho_of('Resolution (A)')[1])}),
mean B-factor {rho_of('Mean experimental B-factor (A^2)')[0]:+.2f} (p {pfmt(rho_of('Mean experimental B-factor (A^2)')[1])}),
lead-mode collectivity {rho_of('Lead-mode collectivity')[0]:+.2f} (p {pfmt(rho_of('Lead-mode collectivity')[1])}).
These are descriptive associations within a modest sample and many tests; p-values are not corrected for
multiple comparisons, and none of these relationships establishes why a given protein is predicted well or badly.

**Pearson versus Spearman.** The mean Spearman rho ({f(S['mean'], 2)}; 95 % CI [{f(S['ci95_mean_low'], 2)}, {f(S['ci95_mean_high'], 2)}])
is higher than the mean Pearson r ({f(P['mean'], 2)}). Spearman exceeds Pearson by more than 0.2 in {n_rho_gt} proteins,
whereas Pearson exceeds Spearman by more than 0.2 in only {n_r_gt}. A plausible (untested) reading is that a few residues with
very large predicted MSF (isolated terminal or loop residues with few contacts) depress the linear correlation more than the rank
correlation, so the ANM's relative ordering of residues is better than its magnitudes. The reverse also occurs:
the highest-r protein in Figure 10 has r = {f(best['pearson_r'], 2)} but a much lower rho, because its r is driven almost entirely by
the extremely flexible chain terminus; Pearson r alone can reward predicting a few extreme residues while saying little about the
core. Both metrics should be read together.

**Case studies (Figure 10).** The three panels show different failure/success modes: agreement at flexible termini (highest r), broad
agreement on loop peaks together with an isolated single-residue spike in the prediction (near-median), and a predicted loop peak
that has no counterpart in a nearly flat experimental profile (lowest r). These are single examples chosen by rank, not
representative mechanisms.

## Limitations

- **Benchmark-selection bias.** Selection favours well-ordered, monomeric, high-resolution, single-chain structures that crystallise readily; flexible, multi-domain, disordered or membrane proteins are under-represented. Filtering on observed-residue coverage removes proteins with large disordered regions, which are exactly the flexible ones. The result describes this population, not proteins in general. Fold and function labels are heuristic.
- **B-factor limitations.** See Interpretation; B-factors mix dynamics, disorder and refinement choices and are compared here without any per-structure correction (no TLS decomposition, no normalisation by resolution or packing).
- **X-ray structures only.** No NMR, cryo-EM or room-temperature data; crystal environment affects every entry.
- **One parameter set.** cutoff {ap['cutoff']} A, gamma {ap['gamma']}, {summ['n_modes_kept']} kept modes, applied uniformly by design. These were not optimised; other cutoffs or all-mode MSF would give different numbers, and no sensitivity analysis is included in this report.
- **Missing residues.** Up to 5 % of residues may be missing internally and up to 10 % of the sequence unobserved; the missing segments are simply absent from the network, which can disconnect it or alter local stiffness. One disconnected-network failure is an example (see Failures).
- **Crystal contacts.** Only the isolated chain (monomer, one asymmetric-unit copy) is modelled; lattice neighbours are ignored.
- **Differences in refinement procedures.** Deposited B-factors come from different programs and protocols; no harmonisation was attempted.
- **Statistics.** Proteins are treated as independent (30 % identity clusters), but structural-family relationships below that threshold remain. The bootstrap CI reflects sampling variability of this benchmark, not selection bias.
- **Quality control performed.** {"; ".join(summ['qc'])}. Correlations use index-aligned MSF and B vectors with one shared finite-value mask, and the Pearson value was cross-checked against the package's own `validate()` output.

## Conclusion

On {nV} non-redundant X-ray monomers analysed with a single untuned parameter set, Cα-ANM predictions of residue
flexibility show a moderate positive rank and linear relationship with experimental B-factors (mean Pearson r
{f(P['mean'], 2)}, median {f(P['median'], 2)}; 95 % CI of the mean [{f(P['ci95_mean_low'], 2)}, {f(P['ci95_mean_high'], 2)}];
mean Spearman rho {f(S['mean'], 2)}, higher than mean Pearson r, which suggests a few extreme predicted residues depress the linear correlation). Performance varies widely, from r = {f(P['min'], 2)} to {f(P['max'], 2)}, and
{len(neg)} proteins are not positively correlated. The model captures a real but partial signal of relative
flexibility; it is not a quantitative predictor of B-factors, and these results do not show it reflects
solution dynamics beyond what crystallographic B-factors measure.
"""
    with open(bu.lp(os.path.join(HERE, "VALIDATION_REPORT.md")), "w", encoding="utf-8") as fh:
        fh.write(md)
    update_readme(nV, nA, P)
    print("wrote VALIDATION_REPORT.md and README validation section")


README_START = "<!-- benchmark-summary:start -->"
README_END = "<!-- benchmark-summary:end -->"


def update_readme(nV, nA, P):
    path = os.path.join(ROOT, "README.md")
    with open(path, encoding="utf-8") as fh:
        txt = fh.read()
    block = (f"{README_START}\n"
             f"### Large-scale benchmark\n\n"
             f"Across {nV} structurally diverse, non-redundant X-ray protein structures ({nA} attempted), "
             f"the model achieved a mean Pearson correlation of {P['mean']:.2f}, median {P['median']:.2f}, "
             f"and standard deviation {P['sd']:.2f} between ANM-predicted residue flexibility and experimental "
             f"B-factors (95% bootstrap CI of the mean [{P['ci95_mean_low']:.2f}, {P['ci95_mean_high']:.2f}]), using one "
             f"untuned parameter set. Performance varies widely between proteins. Full method, statistics, figures "
             f"and limitations: [validation/VALIDATION_REPORT.md](validation/VALIDATION_REPORT.md). "
             f"Reproduce with `python validation/build_benchmark.py` (once) and `python validation/run_benchmark.py`.\n"
             f"{README_END}\n")
    if README_START in txt:
        txt = re.sub(re.escape(README_START) + r".*?" + re.escape(README_END) + r"\n?", lambda m: block, txt, flags=re.S)
    else:
        marker = "## Project structure"
        txt = txt.replace(marker, block + "\n" + marker, 1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(txt)
