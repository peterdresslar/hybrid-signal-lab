from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from .common import add_mode_inset, configure_matplotlib, prettify_type, qualitative_11_color_map


@dataclass(frozen=True)
class ConstantGainPanel:
    path: Path
    title: str


def parse_gain(profile_name: str) -> float:
    return float(profile_name.split("_", 1)[1])


def load_constant_gain_data(path: Path) -> dict[str, list[tuple[float, float]]]:
    rows = list(csv.DictReader(path.open()))
    data: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        profile = row["g_profile"]
        if not profile.startswith("constant_"):
            continue
        prompt_type = row["type"]
        gain = parse_gain(profile)
        try:
            mean_delta_p = float(row["mean_delta_target_prob"])
        except (TypeError, ValueError):
            continue
        data.setdefault(prompt_type, []).append((gain, mean_delta_p))

    for prompt_type in data:
        if not any(abs(gain - 1.0) < 1e-9 for gain, _ in data[prompt_type]):
            data[prompt_type].append((1.0, 0.0))
        data[prompt_type].sort(key=lambda item: item[0])
    return data


def get_type_order_by_peak(data: dict[str, list[tuple[float, float]]]) -> list[str]:
    peaks = []
    for prompt_type, points in data.items():
        peak = max(delta for _, delta in points)
        peaks.append((prompt_type, peak))
    peaks.sort(key=lambda item: item[1], reverse=True)
    return [prompt_type for prompt_type, _ in peaks]


DEFAULT_GAIN_TICKS = [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0]


def _format_gain_tick(value: float, _: float) -> str:
    if abs(value - 1.0) < 1e-9:
        return "1.0"
    return f"{value:g}"


def plot_constant_gain_dose_response(
    *,
    panels: list[ConstantGainPanel],
    output_path: Path,
    mode_label: str | None = None,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    legend_columns: int = 4,
    figsize: tuple[float, float] = (14, 5),
    facecolor: str = "white",
    xticks: list[float] | None = None,
) -> None:
    model_data = [load_constant_gain_data(panel.path) for panel in panels]
    type_order = get_type_order_by_peak(model_data[0])
    colors = qualitative_11_color_map(type_order)
    gain_ticks = xticks if xticks is not None else DEFAULT_GAIN_TICKS

    configure_matplotlib(font_family="sans-serif", font_size=11)

    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharey=True)
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    else:
        axes = [axes]

    for ax, panel, data in zip(axes, panels, model_data):
        for prompt_type in type_order:
            points = data[prompt_type]
            x = [gain for gain, _ in points]
            y = [delta for _, delta in points]
            ax.plot(
                x,
                y,
                color=colors[prompt_type],
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=prettify_type(prompt_type),
            )

        ax.set_title(panel.title)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Constant gain factor (g)")
        ax.xaxis.set_major_locator(FixedLocator(gain_ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_gain_tick))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.5)
        ax.axvline(1.0, color="#888888", linestyle="--", linewidth=0.5)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.3)
        if mode_label is not None:
            add_mode_inset(ax, mode_label)

    axes[0].set_ylabel("Mean Δp (target probability)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=legend_columns,
        frameon=True,
        fancybox=False,
        edgecolor="#BBBBBB",
        framealpha=1.0,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=False, facecolor=facecolor)
    plt.close(fig)
