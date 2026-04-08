from __future__ import annotations

from figurelib.common import REPO_ROOT
from figurelib.oracle_headroom import plot_oracle_headroom_distribution


if __name__ == "__main__":
    base = REPO_ROOT / "data" / "intervention_modes" / "b4_021_attn_contr" / "9B" / "analysis"
    out_png = REPO_ROOT / "docs" / "figure_oracle_headroom_distribution_by_type_qwen9b.png"
    out_md = REPO_ROOT / "docs" / "figure_oracle_headroom_distribution_by_type_qwen9b_caption.md"
    plot_oracle_headroom_distribution(
        prompt_winners_path=base / "analysis_prompt_winners.csv",
        type_gain_summary_path=base / "analysis_type_gain_summary.csv",
        output_path=out_png,
    )
    out_md.write_text(
        "Figure X. Per-prompt oracle headroom by prompt type for Qwen3.5-9B under "
        "attention-contribution intervention. Each point is a prompt's best observed Δp "
        "across all tested profiles; boxes summarize the within-type distribution. Orange "
        "diamonds mark the mean Δp of the best constant profile for each type, showing that "
        "substantial prompt-level headroom remains even in several weak-average categories.\n"
    )
