from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from docs.figurelib.common import REPO_ROOT, configure_matplotlib

FIGURE_DIR = REPO_ROOT / "docs" / "figures" / "archive"
LEFT_IMAGE = FIGURE_DIR / "figure_intra_family_type_responsiveness_qwen_balanced_attention_top8.png"
RIGHT_IMAGE = FIGURE_DIR / "figure_intra_family_type_responsiveness_qwen_balanced_attention.png"
OUTPUT_IMAGE = FIGURE_DIR / "figure_intra_family_type_responsiveness_qwen_balanced_attention_comparison.png"


def main() -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)

    left = mpimg.imread(LEFT_IMAGE)
    right = mpimg.imread(RIGHT_IMAGE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, image in zip(axes, [left, right], strict=True):
        ax.imshow(image)
        ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.075, wspace=0.04)

    fig.text(0.25, 0.033, "Top 8 profiles", ha="center", va="center", fontsize=13)
    fig.text(0.75, 0.033, "All profiles", ha="center", va="center", fontsize=13)

    fig.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
