from __future__ import annotations

from docs.figurelib.common import REPO_ROOT
from docs.figurelib.head_entropy_correlations import plot_head_entropy_correlation_heatmaps


if __name__ == "__main__":
    analysis = REPO_ROOT / "data" / "intervention_modes" / "b4_021_attn_contr" / "9B" / "analysis"
    out_png = REPO_ROOT / "docs" / "figures" / "archive" / "figure_head_entropy_correlations_qwen9b.png"
    out_md = REPO_ROOT / "docs" / "figures" / "archive" / "figure_head_entropy_correlations_qwen9b_caption.md"

    plot_head_entropy_correlation_heatmaps(
        correlations_path=analysis / "analysis_head_correlations.json",
        output_path=out_png,
        profiles=[
            "edges_narrow",
            "late_boost_4.0",
            "shifted_ramp_down",
            "tent_steep",
        ],
    )

    out_md.write_text(
        "Figure X. Per-head correlation between baseline attention entropy and intervention Δp "
        "for the four router-selected profiles in Qwen 3.5 9B (1,070 prompts). Each cell shows "
        "Pearson r for one attention head at one hybrid attention layer; outlined cells mark "
        "|r| > 0.30. The earliest and deepest attention layers carry the strongest signal across "
        "profiles, with individual scout heads reaching |r| above 0.4. The sign structure varies "
        "by profile, showing that the predictive head-level baseline signal is profile-specific. "
        "These headwise patterns are the raw structure compressed by PCA in the subsequent figure.\n"
    )
