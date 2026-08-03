import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .visualize import (
    plot_contact_map,
    plot_msf_overlay,
    plot_correlation_heatmap,
    plot_flexibility_heatmap_1d,
    plot_flexibility_heatmap_3d,
)
from .validation import bfactor_to_msf


def _corr_stat(res):
    if hasattr(res, "statistic"):
        return res.statistic
    if hasattr(res, "correlation"):
        return res.correlation
    return res[0]


def _build_analysis_page(result):
    N = result.coords.shape[0]
    pearson, spearman = result.correlation
    pearson_r = _corr_stat(pearson)
    spearman_r = _corr_stat(spearman)

    order = np.argsort(result.msf)
    most_rigid = order[:5]
    most_flexible = order[::-1][:5]

    def describe(idx):
        chain, resnum, resname = result.labels[idx]
        return f"{resname}{resnum} (chain {chain}): MSF={result.msf[idx]:.4f}"

    if pearson_r >= 0.7:
        agreement = "strong agreement with experimental B-factors"
    elif pearson_r >= 0.3:
        agreement = "moderate agreement with experimental B-factors"
    else:
        agreement = "weak agreement with experimental B-factors"

    dominant_collectivity = float(result.collectivity[0]) if len(result.collectivity) else float("nan")
    if dominant_collectivity >= 0.5:
        motion_desc = "broadly collective, spread across much of the structure"
    else:
        motion_desc = "relatively localized to a subset of residues"

    lines = [
        "ANM Analysis Summary",
        "",
        f"Residues (N): {N}",
        f"Connected contact network: {result.connected}",
        f"Pearson r: {pearson_r:.3f}   Spearman r: {spearman_r:.3f}   ({agreement})",
        f"Lowest-frequency kept mode collectivity: {dominant_collectivity:.3f} ({motion_desc})",
        "",
        "Most flexible residues:",
        *[f"  {describe(i)}" for i in most_flexible],
        "",
        "Most rigid residues:",
        *[f"  {describe(i)}" for i in most_rigid],
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
    return fig


def generate_report(result, output_path):
    with PdfPages(output_path) as pdf:
        fig = plt.figure()
        plot_contact_map(result.adj)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        exp_msf = bfactor_to_msf(result.bfactors)
        plot_msf_overlay(result.msf * result.scale, exp_msf)
        pdf.savefig(fig)
        plt.close(fig)

        if result.cov is not None:
            fig = plt.figure()
            N = result.coords.shape[0]
            plot_correlation_heatmap(result.cov, N)
            pdf.savefig(fig)
            plt.close(fig)

        fig = plt.figure()
        plot_flexibility_heatmap_1d(result.msf)
        pdf.savefig(fig)
        plt.close(fig)

        # plot_flexibility_heatmap_3d creates its own figure internally
        # (unlike the other plot_* functions, which draw on the current
        # figure via gca()), so grab the figure back off the returned axes
        # instead of pre-creating one.
        ax3d = plot_flexibility_heatmap_3d(result.coords, result.msf)
        fig3d = ax3d.get_figure()
        pdf.savefig(fig3d)
        plt.close(fig3d)

        fig = _build_analysis_page(result)
        pdf.savefig(fig)
        plt.close(fig)
