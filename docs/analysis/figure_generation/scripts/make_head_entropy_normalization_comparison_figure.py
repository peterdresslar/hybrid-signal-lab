from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.figurelib.common import REPO_ROOT
from docs.figurelib.head_entropy_correlations import plot_head_entropy_normalization_comparison


if __name__ == "__main__":
    data_dir = REPO_ROOT / "data" / "022-balanced-attention-hybrid"
    analysis = data_dir / "9B" / "analysis"
    out_png = REPO_ROOT / "docs" / "figures" / "archive" / "figure_head_entropy_normalization_comparison_qwen9b.png"

    plot_head_entropy_normalization_comparison(
        data_dir=data_dir,
        model_key="9B",
        correlations_path=analysis / "analysis_head_correlations.json",
        joined_path=analysis / "analysis_joined_long.csv",
        output_path=out_png,
        profiles=["oracle_any_constant_positive", "constant_response_polarity"],
    )
