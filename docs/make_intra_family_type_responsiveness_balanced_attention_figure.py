from __future__ import annotations

from figurelib.common import REPO_ROOT
from figurelib.type_responsiveness import TypeResponsivenessSeries, plot_type_responsiveness_lines


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


if __name__ == "__main__":
    base = REPO_ROOT / "data" / "022-balanced-attention-hybrid"
    plot_type_responsiveness_lines(
        series=[
            TypeResponsivenessSeries(
                path=base / "2B" / "analysis" / "analysis_type_gain_summary.csv",
                label="2B",
                color="#E67E5F",
                marker="o",
            ),
            TypeResponsivenessSeries(
                path=base / "9B" / "analysis" / "analysis_type_gain_summary.csv",
                label="9B",
                color="#4C78A8",
                marker="s",
            ),
            TypeResponsivenessSeries(
                path=base / "35B" / "analysis" / "analysis_type_gain_summary.csv",
                label="35B (MoE)",
                color="#3F2B63",
                marker="D",
            ),
        ],
        type_order=TYPE_ORDER,
        type_labels=TYPE_LABELS,
        output_path=REPO_ROOT / "docs" / "figure_intra_family_type_responsiveness_qwen_balanced_attention.png",
        mode_label="Attention-Contribution",
        ylim=(-0.20, 0.22),
        figsize=(7, 6),
        annotate_clusters=True,
    )
