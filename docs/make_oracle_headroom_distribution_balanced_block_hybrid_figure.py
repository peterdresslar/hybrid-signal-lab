from __future__ import annotations

from figurelib.common import REPO_ROOT
from figurelib.oracle_headroom import plot_oracle_headroom_distribution_two_panel


if __name__ == "__main__":
    base = REPO_ROOT / "data" / "022-balanced-block-hybrid"
    out_png = REPO_ROOT / "docs" / "figure_oracle_headroom_distribution_balanced_block_hybrid.png"
    out_md = REPO_ROOT / "docs" / "figure_oracle_headroom_distribution_balanced_block_hybrid_caption.md"

    plot_oracle_headroom_distribution_two_panel(
        qwen_prompt_winners_path=base / "9B" / "analysis" / "analysis_prompt_winners.csv",
        qwen_type_gain_summary_path=base / "9B" / "analysis" / "analysis_type_gain_summary.csv",
        olmo_prompt_winners_path=base / "OLMO" / "analysis" / "analysis_prompt_winners.csv",
        olmo_type_gain_summary_path=base / "OLMO" / "analysis" / "analysis_type_gain_summary.csv",
        output_path=out_png,
        mode_label="Block Intervention",
        xlim=(-0.05, 1.05),
    )

    out_md.write_text(
        "Figure X. Per-prompt oracle headroom by prompt type for the balanced block-output sweep in "
        "Qwen 3.5 9B and Olmo Hybrid 7B. Each point is a prompt's best observed Δp across the "
        "balanced kitchen-sink profile set; boxes summarize the within-type distribution and orange "
        "diamonds mark the mean Δp of the best constant profile for each type. Under block-output "
        "intervention, Qwen retains a small productive window concentrated in computational tasks, "
        "while Olmo's headroom remains narrower and more heterogeneous across prompt families.\n"
    )
