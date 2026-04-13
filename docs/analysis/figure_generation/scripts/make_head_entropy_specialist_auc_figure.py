from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.figurelib.common import REPO_ROOT
from docs.figurelib.head_entropy_diagnostics import plot_shared_vs_specialist_auc


if __name__ == "__main__":
    shared_metrics_path = (
        REPO_ROOT / "docs" / "analysis" / "head_entropy" / "outputs" / "qwen9b" / "length_resid" / "metrics_summary.json"
    )
    specialist_metrics_path = (
        REPO_ROOT
        / "docs"
        / "analysis"
        / "head_entropy"
        / "outputs"
        / "qwen9b"
        / "length_resid"
        / "specialist_vs_anchor_summary.json"
    )
    output_path = REPO_ROOT / "docs" / "figures" / "manuscript" / "figure5b.png"
    plot_shared_vs_specialist_auc(
        shared_metrics_path=shared_metrics_path,
        specialist_metrics_path=specialist_metrics_path,
        output_path=output_path,
    )
