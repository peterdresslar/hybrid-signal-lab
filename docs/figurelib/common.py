from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


DOCS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = DOCS_ROOT.parent


def prettify_type(name: str) -> str:
    return name.replace("_", " ")


def configure_matplotlib(*, font_family: str = "sans-serif", font_size: int = 11) -> None:
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": font_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )


def qualitative_11_color_map(type_order: list[str]) -> dict[str, str]:
    tab10 = list(plt.get_cmap("tab10").colors)
    extra = ["#4D4D4D"]
    palette = tab10 + extra
    return {prompt_type: palette[idx] for idx, prompt_type in enumerate(type_order)}

