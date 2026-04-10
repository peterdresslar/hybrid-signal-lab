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
    out_png = REPO_ROOT / "docs" / "figure_head_entropy_shared_vs_specific_qwen9b.png"
    out_md = REPO_ROOT / "docs" / "figure_head_entropy_shared_vs_specific_qwen9b_caption.md"

    plot_head_entropy_bistate(
        data_dir=data_dir,
        model_key="9B",
        correlations_path=analysis / "analysis_head_correlations.json",
        joined_path=analysis / "analysis_joined_long.csv",
        output_path=out_png,
        profile_name="constant_2.3",
    )

    out_md.write_text(
        "Figure X. Headwise baseline-attention-entropy structure for the bistate 9B "
        "attention-contribution router under `constant_2.3`. Left: Pearson correlation "
        "between each head's baseline entropy and per-prompt `Δp` under `constant_2.3`. "
        "Right: mean baseline-entropy difference between prompts for which `constant_2.3` "
        "is beneficial (`Δp > 0`) and prompts best left off (`Δp ≤ 0`). The same scout-head "
        "regions recur in both panels, indicating that the bistate router is exploiting a "
        "single dominant intervention-susceptibility signal rather than separating among "
        "multiple profile-specific regimes.\n"
    )
