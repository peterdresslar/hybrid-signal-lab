"""
Figure 7 (manuscript): Intra-family type responsiveness, Qwen 3.5 family,
balanced attention-contribution mode.

Regenerated from 022-balanced-attention-hybrid data.
Output goes directly to manuscript figures folder (cas-capstone-dresslar/figures/figure7.png).

Run from repo root:
    python -m docs.analysis.figure_generation.scripts.make_figure7_manuscript
"""
from __future__ import annotations

import shutil
from pathlib import Path

from docs.figurelib.common import REPO_ROOT
from docs.figurelib.type_responsiveness import TypeResponsivenessSeries, plot_type_responsiveness_lines


TYPE_ORDER = [
    "code_comprehension",
    "reasoning_numerical",
    "algorithmic",
    "reasoning_tracking",
    "structural_copying",
    "syntactic_pattern",
    "factual_recall",
    "factual_retrieval",
    "domain_knowledge",
    "cultural_memorized",
    "long_range_retrieval",
]

TYPE_LABELS = [
    "COD",
    "RNU",
    "ALG",
    "RTR",
    "STC",
    "SYN",
    "FAR",
    "FTR",
    "DOM",
    "CUL",
    "LRR",
]

# Output paths
ARCHIVE_PATH = REPO_ROOT / "docs" / "figures" / "archive" / "figure_intra_family_type_responsiveness_qwen_balanced_attention.png"
MANUSCRIPT_PATH = REPO_ROOT.parent / "cas-capstone-dresslar" / "figures" / "figure7.png"


if __name__ == "__main__":
    base = REPO_ROOT / "data" / "022-balanced-attention-hybrid"

    # Balanced data ylim -- magnitudes are ~3-5x smaller than unbalanced
    plot_type_responsiveness_lines(
        series=[
            TypeResponsivenessSeries(
                path=base / "2B" / "analysis" / "analysis_type_gain_summary.csv",
                label="Qwen 3.5 2B (dense)",
                color="#E67E5F",
                marker="o",
            ),
            TypeResponsivenessSeries(
                path=base / "9B" / "analysis" / "analysis_type_gain_summary.csv",
                label="Qwen 3.5 9B (dense)",
                color="#4C78A8",
                marker="s",
            ),
            TypeResponsivenessSeries(
                path=base / "35B" / "analysis" / "analysis_type_gain_summary.csv",
                label="Qwen 3.5 35B (MoE)",
                color="#3F2B63",
                marker="D",
            ),
        ],
        type_order=TYPE_ORDER,
        type_labels=TYPE_LABELS,
        output_path=ARCHIVE_PATH,
        mode_label="Attention-Contribution (balanced)",
        ylim=(-0.08, 0.08),  # tighter ylim for balanced magnitudes
        figsize=(7, 6),
        annotate_clusters=True,
    )

    # Copy to manuscript figures
    if MANUSCRIPT_PATH.parent.exists():
        shutil.copy2(ARCHIVE_PATH, MANUSCRIPT_PATH)
        print(f"Copied to {MANUSCRIPT_PATH}")
    else:
        print(f"Manuscript figures dir not found at {MANUSCRIPT_PATH.parent}")
        print(f"Archive copy saved at {ARCHIVE_PATH}")

    print("Done.")
