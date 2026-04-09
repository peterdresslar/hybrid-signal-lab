from __future__ import annotations

from figurelib.common import REPO_ROOT
from figurelib.head_entropy_correlations import plot_head_entropy_shared_vs_specific


if __name__ == "__main__":
    analysis = REPO_ROOT / "data" / "intervention_modes" / "b4_021_attn_contr" / "9B" / "analysis"
    out_png = REPO_ROOT / "docs" / "figure_head_entropy_shared_vs_specific_qwen9b.png"
    out_md = REPO_ROOT / "docs" / "figure_head_entropy_shared_vs_specific_qwen9b_caption.md"

    plot_head_entropy_shared_vs_specific(
        correlations_path=analysis / "analysis_head_correlations.json",
        output_path=out_png,
        shared_profiles=[
            "edges_narrow",
            "late_boost_4.0",
            "shifted_ramp_down",
            "tent_steep",
        ],
        contrast_profiles=[
            "edges_narrow",
            "tent_steep",
        ],
    )

    out_md.write_text(
        "Figure X. Headwise baseline-entropy correlations decomposed into a shared intervention "
        "scaffold and profile-specific residuals for the four router profiles in Qwen 3.5 9B. "
        "Left: the mean Pearson-r map across the four profiles, showing a common set of scout heads "
        "that predict whether prompts are broadly intervention-responsive. Middle and right: profile "
        "deviations from that shared scaffold for `edges_narrow` and `tent_steep`. The residual maps "
        "are much smaller in magnitude, indicating that routing distinctions sit on top of a dominant "
        "shared susceptibility signal rather than arising from wholly disjoint head sets.\n"
    )
