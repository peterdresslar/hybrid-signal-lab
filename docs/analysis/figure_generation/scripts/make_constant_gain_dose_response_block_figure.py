from __future__ import annotations

from docs.figurelib.common import REPO_ROOT
from docs.figurelib.constant_gain import ConstantGainPanel, plot_constant_gain_dose_response


OUT_PNG = REPO_ROOT / "docs" / "figures" / "archive" / "figure_constant_gain_dose_response_by_type_block_output.png"


if __name__ == "__main__":
    plot_constant_gain_dose_response(
        panels=[
            ConstantGainPanel(
                path=REPO_ROOT / "data" / "intervention_modes" / "b4_010_ks1_backend_default" / "9B" / "analysis" / "analysis_type_gain_summary.csv",
                title="Qwen 3.5 9B (pre-norm, block-output)",
            ),
            ConstantGainPanel(
                path=REPO_ROOT / "data" / "intervention_modes" / "b4_010_ks1_backend_default" / "OLMO" / "analysis" / "analysis_type_gain_summary.csv",
                title="Olmo Hybrid 7B (post-norm, block-output)",
            ),
        ],
        output_path=OUT_PNG,
        xlim=(0.2, 2.05),
        ylim=(-0.80, 0.40),
    )
