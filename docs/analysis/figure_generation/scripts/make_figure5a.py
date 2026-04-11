from __future__ import annotations

from pathlib import Path

from docs.figurelib.head_entropy_diagnostics import plot_figure5a_variance_vs_residual_auc


REPO_ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    raw_metrics = REPO_ROOT / "docs" / "analysis" / "head_entropy" / "outputs" / "qwen9b" / "raw" / "metrics_summary.json"
    resid_metrics = (
        REPO_ROOT
        / "docs"
        / "analysis"
        / "head_entropy"
        / "outputs"
        / "qwen9b"
        / "length_resid_full"
        / "metrics_summary.json"
    )
    output_path = REPO_ROOT / "docs" / "figures" / "manuscript" / "figure5a.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_figure5a_variance_vs_residual_auc(
        raw_metrics_path=raw_metrics,
        length_resid_metrics_path=resid_metrics,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
