from __future__ import annotations

from docs.figurelib.common import REPO_ROOT
from docs.figurelib.task_dependent_gain import plot_task_dependent_gain_response


if __name__ == "__main__":
    data_dir = REPO_ROOT / "data" / "intervention_modes" / "b4_021_attn_contr" / "9B"
    plot_task_dependent_gain_response(
        type_gain_summary_path=data_dir / "analysis" / "analysis_type_gain_summary.csv",
        meta_path=data_dir / "_meta.json",
        output_combined=REPO_ROOT / "docs" / "figures" / "archive" / "figure_task_dependent_gain_response_qwen9b.png",
        output_top=REPO_ROOT / "docs" / "figures" / "archive" / "figure_task_dependent_gain_response_qwen9b_top.png",
        output_bottom=REPO_ROOT / "docs" / "figures" / "archive" / "figure_task_dependent_gain_response_qwen9b_heatmap.png",
        caption_path=REPO_ROOT / "docs" / "figures" / "archive" / "figure_task_dependent_gain_response_qwen9b_caption.md",
    )
