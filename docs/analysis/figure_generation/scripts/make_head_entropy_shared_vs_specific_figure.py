from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.figurelib.common import REPO_ROOT
from docs.figurelib.head_entropy_correlations import plot_head_entropy_shared_vs_specific


if __name__ == "__main__":
    data_dir = REPO_ROOT / "data" / "022-balanced-attention-hybrid"
    analysis = data_dir / "9B" / "analysis"
    out_png = REPO_ROOT / "docs" / "figures" / "archive" / "figure_head_entropy_shared_vs_specific_qwen9b.png"
    out_md = REPO_ROOT / "docs" / "figures" / "archive" / "figure_head_entropy_shared_vs_specific_qwen9b_caption.md"

    plot_head_entropy_shared_vs_specific(
        correlations_path=analysis / "analysis_head_correlations.json",
        output_path=out_png,
        shared_profiles=[
            "constant_2.6",
            "edges_narrow_bal_0.55",
            "late_boost_bal_0.60",
            "triad_odd_bal_0.45",
        ],
    )

    out_md.write_text(
        "Figure X. Headwise baseline-attention-entropy structure for the four benchmarked "
        "Qwen 3.5 9B intervention profiles in the balanced attention-contribution sweep. "
        "Left: mean Pearson correlation between each head's baseline entropy and per-prompt "
        "`Δp` across the four selected profiles (`constant_2.6`, `edges_narrow_bal_0.55`, "
        "`late_boost_bal_0.60`, `triad_odd_bal_0.45`). Right: recurrence map showing, for "
        "each head, in how many of the four profiles that head appears among the top-12 "
        "absolute head-entropy correlations. The repeated scout-head regions indicate that "
        "baseline head entropy captures a shared intervention-susceptibility scaffold. "
        "Because the same heads recur across profiles, this headwise view alone does not "
        "fully resolve which profile is best for a given prompt.\n"
    )
