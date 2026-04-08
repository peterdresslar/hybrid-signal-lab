from __future__ import annotations

from figurelib.common import REPO_ROOT
from figurelib.constant_gain import ConstantGainPanel, plot_constant_gain_dose_response


OUT_PNG = REPO_ROOT / "docs" / "figure_constant_gain_dose_response_by_type.png"


if __name__ == "__main__":
    plot_constant_gain_dose_response(
        panels=[
            ConstantGainPanel(
                path=REPO_ROOT / "data" / "intervention_modes" / "b4_021_attn_contr" / "9B" / "analysis" / "analysis_type_gain_summary.csv",
                title="Qwen 3.5 9B (pre-norm)",
            ),
            ConstantGainPanel(
                path=REPO_ROOT / "data" / "intervention_modes" / "b4_021_attn_contr" / "OLMO" / "analysis" / "analysis_type_gain_summary.csv",
                title="Olmo Hybrid 7B (post-norm)",
            ),
        ],
        output_path=OUT_PNG,
        xlim=(0.4, 3.1),
        ylim=(-0.45, 0.40),
    )
