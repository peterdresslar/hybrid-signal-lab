from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figurelib.common import REPO_ROOT
from figurelib.head_entropy_correlations import plot_head_entropy_bistate


if __name__ == "__main__":
    data_dir = REPO_ROOT / "data" / "022-balanced-attention-hybrid"
    analysis = data_dir / "9B" / "analysis"
    out_png = REPO_ROOT / "docs" / "figure_head_entropy_bookend_bistate_qwen9b.png"
    out_md = REPO_ROOT / "docs" / "figure_head_entropy_bookend_bistate_qwen9b_caption.md"

    plot_head_entropy_bistate(
        data_dir=data_dir,
        model_key="9B",
        correlations_path=analysis / "analysis_head_correlations.json",
        joined_path=analysis / "analysis_joined_long.csv",
        output_path=out_png,
        profile_name="bookend_high_bal_0.40",
    )

    out_md.write_text(
        "Figure X. Headwise baseline-attention-entropy structure for the sparse specialist "
        "profile `bookend_high_bal_0.40` in the balanced 9B attention-contribution sweep. "
        "Left: Pearson correlation between each head's baseline entropy and per-prompt `Δp` "
        "under `bookend_high_bal_0.40`. Right: mean baseline-entropy difference between prompts "
        "for which the profile is beneficial (`Δp > 0`) and prompts best left off (`Δp ≤ 0`). "
        "Compared with the broad bistate `constant_2.3` figure, this profile's signal should be "
        "read as a narrower specialist regime rather than a global intervention-susceptibility axis.\n"
    )
