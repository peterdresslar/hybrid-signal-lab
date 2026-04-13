from __future__ import annotations

from docs.figurelib.common import REPO_ROOT
from docs.figurelib.oracle_headroom import plot_oracle_headroom_distribution_two_panel


if __name__ == "__main__":
    base = REPO_ROOT / "data" / "022-balanced-attention-hybrid"
    out_png = REPO_ROOT / "docs" / "figures" / "archive" / "figure_oracle_headroom_distribution_balanced_hybrid.png"
    out_md = REPO_ROOT / "docs" / "figures" / "archive" / "figure_oracle_headroom_distribution_balanced_hybrid_caption.md"

    plot_oracle_headroom_distribution_two_panel(
        qwen_prompt_winners_path=base / "9B" / "analysis" / "analysis_prompt_winners.csv",
        qwen_type_gain_summary_path=base / "9B" / "analysis" / "analysis_type_gain_summary.csv",
        olmo_prompt_winners_path=base / "OLMO" / "analysis" / "analysis_prompt_winners.csv",
        olmo_type_gain_summary_path=base / "OLMO" / "analysis" / "analysis_type_gain_summary.csv",
        output_path=out_png,
        mode_label="Attention-Contribution",
        xlim=(-0.05, 0.85),
    )

    out_md.write_text(
        "Figure X. Per-prompt oracle headroom by prompt type for the balanced attention-contribution "
        "sweep in Qwen 3.5 9B and Olmo Hybrid 7B. Each point is a prompt's best observed Δp across "
        "the balanced kitchen-sink profile set; boxes summarize the within-type distribution and "
        "orange diamonds mark the mean Δp of the best constant profile for each type. Qwen retains "
        "substantial oracle headroom in computational prompt families, while Olmo's headroom shifts "
        "toward narrower, lower-magnitude niches despite comparable prompt-level spread in several types.\n"
    )
