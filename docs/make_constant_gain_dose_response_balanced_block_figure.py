from __future__ import annotations

from figurelib.common import REPO_ROOT
from figurelib.constant_gain import ConstantGainPanel, plot_constant_gain_dose_response


OUT_PNG = REPO_ROOT / "docs" / "figure_constant_gain_dose_response_balanced_block_hybrid.png"


if __name__ == "__main__":
    plot_constant_gain_dose_response(
        panels=[
            ConstantGainPanel(
                path=REPO_ROOT / "data" / "022-balanced-block-hybrid" / "9B" / "analysis" / "analysis_type_gain_summary.csv",
                title="Qwen 3.5 9B (pre-norm)",
            ),
            ConstantGainPanel(
                path=REPO_ROOT / "data" / "022-balanced-block-hybrid" / "OLMO" / "analysis" / "analysis_type_gain_summary.csv",
                title="Olmo Hybrid 7B (post-norm)",
            ),
        ],
        output_path=OUT_PNG,
        mode_label="Block Intervention",
        xlim=(0.4, 3.1),
        ylim=(-0.80, 0.40),
    )
