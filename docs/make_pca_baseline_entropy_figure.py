from __future__ import annotations

from figurelib.common import REPO_ROOT
from figurelib.pca_baseline_entropy import plot_pca_baseline_entropy


if __name__ == "__main__":
    analysis = REPO_ROOT / "data" / "intervention_modes" / "b4_021_attn_contr" / "9B" / "analysis"
    out_png = REPO_ROOT / "docs" / "figure_pca_baseline_attention_entropy_qwen9b.png"
    out_md = REPO_ROOT / "docs" / "figure_pca_baseline_attention_entropy_qwen9b_caption.md"

    stats = plot_pca_baseline_entropy(
        pca_path=analysis / "analysis_baseline_attn_pca.json",
        joined_long_path=analysis / "analysis_joined_long.csv",
        output_path=out_png,
        route_profiles=[
            "edges_narrow",
            "late_boost_4.0",
            "shifted_ramp_down",
            "tent_steep",
        ],
    )

    out_md.write_text(
        "Figure X. PCA projection of per-head baseline attention entropy for Qwen 3.5 9B "
        f"(1,070 prompts). (A) Colored by prompt type, with PC1 ({stats['pc1_pct']:.1f}%) "
        "strongly separating the manual task categories. (B) Colored by oracle routing "
        "assignment for the deployed four-profile router set plus `off`; routing structure "
        f"is weak on PC1 (η² = {stats['eta_pc1_by_route']:.3f}) but stronger on PC2 "
        f"(η² = {stats['eta_pc2_by_route']:.3f}), where intervention-responsive prompts "
        f"have mean PC2 = {stats['mean_pc2_on']:+.2f} and `off` prompts have mean PC2 = "
        f"{stats['mean_pc2_off']:+.2f} (Cohen's d = {stats['cohen_d_pc2_on_vs_off']:.2f}). "
        "The routing-relevant structure is therefore partly orthogonal to the coarse type axis, "
        "consistent with the prompt-level separability seen elsewhere in the sweep.\n"
    )
