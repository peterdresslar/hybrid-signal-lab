from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.figurelib.common import REPO_ROOT
from docs.figurelib.head_entropy_diagnostics import plot_head_entropy_diagnostics


if __name__ == "__main__":
    metrics_path = REPO_ROOT / "docs" / "analysis" / "head_entropy" / "outputs" / "qwen9b" / "raw" / "metrics_summary.json"
    output_path = REPO_ROOT / "docs" / "figures" / "diagnostics" / "figure_head_entropy_diagnostics_qwen9b.png"
    plot_head_entropy_diagnostics(
        metrics_path=metrics_path,
        output_path=output_path,
    )
