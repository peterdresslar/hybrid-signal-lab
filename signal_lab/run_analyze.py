"""Batch analysis for a sweep *collection* directory with one model run per subfolder.

Typical layout::

    [DATA_DIR]/outputs/signal_lab/runs/<collection>/
        35B_20260322_123428/
            main.jsonl
            _meta.json
            ...
        OLMO_20260322_152121/
            ...

Each model directory is analyzed with ``signal_lab.sweep_analyze`` (same as running
``--run-dir`` on that folder). If ``--input-dir`` itself contains ``main.jsonl``,
a single run is analyzed (same as ``sweep_analyze``).

After each successful analysis (when files are written), ``signal_lab.sweep_plot_analyze``
runs on that model's analysis directory. When two or more models are present and
``--no-compare`` is not set, all pairwise ``signal_lab.sweep_compare`` and
``signal_lab.sweep_plot_compare`` steps run (compare output defaults under the
collection when layouts match ``runs/<collection>/``; with ``--output-parent``,
comparisons go under ``<output-parent>/_comparisons/``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from signal_lab.paths import DATA_DIR_ENV_VAR, default_compare_output_dir


def discover_model_run_dirs(root: Path) -> list[Path]:
    """Return sweep run directories (each must contain main.jsonl)."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    if (root / "main.jsonl").is_file():
        return [root]
    runs: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "main.jsonl").is_file():
            runs.append(child)
    return runs


def analysis_dir_for(run_dir: Path, output_parent: Path | None) -> Path:
    """Directory containing sweep_analyze artifacts (``analysis_joined_long.csv``, etc.)."""
    if output_parent is not None:
        return (output_parent / run_dir.name / "analysis").resolve()
    return (run_dir / "analysis").resolve()


def compare_bundle_dir(
    run_a: Path,
    run_b: Path,
    *,
    output_parent: Path | None,
) -> Path | None:
    """Explicit compare dir under --output-parent (matches sweep_compare name order: run-a vs run-b)."""
    if output_parent is not None:
        return (output_parent / "_comparisons" / f"{run_a.name}_vs_{run_b.name}").resolve()
    return None


def build_sweep_analyze_cmd(
    run_dir: Path,
    *,
    data_dir: str | None,
    output_dir: Path | None,
    prefix: str,
    no_write_files: bool,
    json_out: Path | None,
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        "-m",
        "signal_lab.sweep_analyze",
        "--run-dir",
        str(run_dir),
        "--prefix",
        prefix,
    ]
    if data_dir is not None:
        cmd.extend(["--data-dir", data_dir])
    if output_dir is not None:
        cmd.extend(["--output-dir", str(output_dir)])
    if no_write_files:
        cmd.append("--no-write-files")
    if json_out is not None:
        cmd.extend(["--json-out", str(json_out)])
    return cmd


def build_sweep_plot_analyze_cmd(
    analysis_dir: Path,
    *,
    data_dir: str | None,
    prefix: str,
    x_metric: str,
    x_metrics: list[str] | None,
    intervention_folders: bool,
    best_interventions_top_n: int,
    plot_dpi: int,
    label_top_n: int,
    min_family_points: int,
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        "-m",
        "signal_lab.sweep_plot_analyze",
        "--analysis-dir",
        str(analysis_dir),
        "--prefix",
        prefix,
        "--dpi",
        str(plot_dpi),
        "--label-top-n",
        str(label_top_n),
        "--min-family-points",
        str(min_family_points),
        "--best-interventions-top-n",
        str(best_interventions_top_n),
    ]
    if data_dir is not None:
        cmd.extend(["--data-dir", data_dir])
    if x_metrics:
        cmd.append("--x-metrics")
        cmd.extend(x_metrics)
    else:
        cmd.extend(["--x-metric", x_metric])
    if intervention_folders:
        cmd.append("--intervention-folders")
    return cmd


def build_sweep_compare_cmd(
    analysis_a: Path,
    analysis_b: Path,
    *,
    label_a: str,
    label_b: str,
    data_dir: str | None,
    output_dir: Path | None,
    compare_prefix: str,
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        "-m",
        "signal_lab.sweep_compare",
        "--run-a",
        str(analysis_a),
        "--run-b",
        str(analysis_b),
        "--label-a",
        label_a,
        "--label-b",
        label_b,
        "--prefix",
        compare_prefix,
    ]
    if data_dir is not None:
        cmd.extend(["--data-dir", data_dir])
    if output_dir is not None:
        cmd.extend(["--output-dir", str(output_dir)])
    return cmd


def build_sweep_plot_compare_cmd(
    compare_dir: Path,
    *,
    data_dir: str | None,
    compare_prefix: str | None,
    intervention_folders: bool,
    best_interventions_top_n: int,
    disagreement_top_n: int,
    plot_dpi: int,
    label_top_n: int,
    min_family_points: int,
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        "-m",
        "signal_lab.sweep_plot_compare",
        "--compare-dir",
        str(compare_dir),
        "--dpi",
        str(plot_dpi),
        "--label-top-n",
        str(label_top_n),
        "--min-family-points",
        str(min_family_points),
        "--best-interventions-top-n",
        str(best_interventions_top_n),
        "--disagreement-top-n",
        str(disagreement_top_n),
    ]
    if data_dir is not None:
        cmd.extend(["--data-dir", data_dir])
    if compare_prefix is not None:
        cmd.extend(["--prefix", compare_prefix])
    if intervention_folders:
        cmd.append("--intervention-folders")
    return cmd


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print(" ", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one or more sweep model runs: either a single directory with "
            "main.jsonl, or a collection directory whose subfolders each hold a model run. "
            "Optionally generates per-model plots and all pairwise cross-model comparisons."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help=(
            "Sweep run directory (contains main.jsonl) or a parent directory whose "
            "immediate subdirectories are model runs (each with main.jsonl)."
        ),
    )
    parser.add_argument(
        "--output-parent",
        type=str,
        default=None,
        help=(
            "Optional parent directory. Each model run is written under "
            "<output-parent>/<run-folder-name>/analysis/ instead of the default "
            "<run-dir>/analysis/. Pairwise comparisons use "
            "<output-parent>/_comparisons/<a>_vs_<b>/."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"Forwarded to sweep_analyze, sweep_plot_analyze, sweep_compare, sweep_plot_compare. {DATA_DIR_ENV_VAR} supported.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="analysis",
        help="Forwarded to sweep_analyze and sweep_plot_analyze (default: analysis).",
    )
    parser.add_argument(
        "--no-write-files",
        action="store_true",
        help="Forwarded to sweep_analyze. Skips plots and comparisons (nothing to read).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help=(
            "Forwarded to sweep_analyze for a single discovered run. "
            "If multiple runs are discovered, use --json-out-dir instead."
        ),
    )
    parser.add_argument(
        "--json-out-dir",
        type=str,
        default=None,
        help=(
            "When multiple model runs are analyzed, write one JSON report per run to "
            "<json-out-dir>/<run-folder-name>.json"
        ),
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip pairwise sweep_compare and sweep_plot_compare (per-model plots still run).",
    )
    parser.add_argument(
        "--compare-prefix",
        type=str,
        default="compare",
        help="Filename prefix for sweep_compare / sweep_plot_compare (default: compare).",
    )
    parser.add_argument(
        "--x-metric",
        type=str,
        default="tokens_approx",
        help="Primary x-axis for sweep_plot_analyze when --x-metrics is not set.",
    )
    parser.add_argument(
        "--x-metrics",
        nargs="+",
        default=None,
        help="Optional batch of x-axis metrics for sweep_plot_analyze.",
    )
    intervention_group = parser.add_mutually_exclusive_group()
    intervention_group.add_argument(
        "--intervention-folders",
        dest="intervention_folders",
        action="store_true",
        help=(
            "Write intervention-folder plot bundles (interventions/, baseline/, best_interventions/) "
            "for sweep_plot_analyze and sweep_plot_compare. Enabled by default."
        ),
    )
    intervention_group.add_argument(
        "--no-intervention-folders",
        dest="intervention_folders",
        action="store_false",
        help="Disable intervention-folder plot bundles.",
    )
    parser.set_defaults(intervention_folders=True)
    parser.add_argument(
        "--best-interventions-top-n",
        type=int,
        default=12,
        help="Forwarded to plot commands (default: 12).",
    )
    parser.add_argument(
        "--disagreement-top-n",
        type=int,
        default=12,
        help="Forwarded to sweep_plot_compare (default: 12).",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=220,
        help="Forwarded to plot commands (default: 220).",
    )
    parser.add_argument(
        "--label-top-n",
        type=int,
        default=3,
        help="Forwarded to plot commands (default: 3).",
    )
    parser.add_argument(
        "--min-family-points",
        type=int,
        default=50,
        help="Forwarded to plot commands (default: 50).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print subprocess commands without executing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir)
    runs = discover_model_run_dirs(root)
    if not runs:
        print(f"No sweep runs found under {root.expanduser().resolve()}", file=sys.stderr)
        print(
            "Expected either main.jsonl in the given directory, or subfolders each "
            "containing main.jsonl.",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(runs) > 1 and args.json_out:
        print(
            "Use --json-out-dir when analyzing multiple runs (not --json-out).",
            file=sys.stderr,
        )
        sys.exit(2)

    json_out_dir = Path(args.json_out_dir).expanduser() if args.json_out_dir else None
    output_parent = Path(args.output_parent).expanduser() if args.output_parent else None
    do_plots = not args.no_write_files
    do_compare = do_plots and len(runs) >= 2 and not args.no_compare

    for idx, run_dir in enumerate(runs, start=1):
        name = run_dir.name
        if output_parent is not None:
            out = output_parent / name / "analysis"
            out.mkdir(parents=True, exist_ok=True)
        else:
            out = None

        json_path: Path | None = None
        if args.json_out:
            json_path = Path(args.json_out).expanduser()
        elif json_out_dir is not None:
            json_out_dir.mkdir(parents=True, exist_ok=True)
            json_path = json_out_dir / f"{name}.json"

        analyze_cmd = build_sweep_analyze_cmd(
            run_dir,
            data_dir=args.data_dir,
            output_dir=out,
            prefix=args.prefix,
            no_write_files=args.no_write_files,
            json_out=json_path,
        )

        print(f"[{idx}/{len(runs)}] {name} — sweep_analyze")
        run_cmd(analyze_cmd, dry_run=args.dry_run)

        if do_plots:
            adir = analysis_dir_for(run_dir, output_parent)
            plot_cmd = build_sweep_plot_analyze_cmd(
                adir,
                data_dir=args.data_dir,
                prefix=args.prefix,
                x_metric=args.x_metric,
                x_metrics=args.x_metrics,
                intervention_folders=args.intervention_folders,
                best_interventions_top_n=args.best_interventions_top_n,
                plot_dpi=args.plot_dpi,
                label_top_n=args.label_top_n,
                min_family_points=args.min_family_points,
            )
            print(f"[{idx}/{len(runs)}] {name} — sweep_plot_analyze")
            run_cmd(plot_cmd, dry_run=args.dry_run)

    if do_compare:
        print("Pairwise comparisons (sweep_compare + sweep_plot_compare)")
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                ra, rb = runs[i], runs[j]
                la, lb = ra.name, rb.name
                analysis_a = analysis_dir_for(ra, output_parent)
                analysis_b = analysis_dir_for(rb, output_parent)
                cmp_out = compare_bundle_dir(ra, rb, output_parent=output_parent)
                compare_cmd = build_sweep_compare_cmd(
                    analysis_a,
                    analysis_b,
                    label_a=la,
                    label_b=lb,
                    data_dir=args.data_dir,
                    output_dir=cmp_out,
                    compare_prefix=args.compare_prefix,
                )
                print(f"  {la} vs {lb} — sweep_compare")
                run_cmd(compare_cmd, dry_run=args.dry_run)

                if cmp_out is not None:
                    compare_dir = cmp_out
                else:
                    try:
                        compare_dir = default_compare_output_dir(analysis_a, analysis_b)
                    except ValueError:
                        print(
                            f"  [skip sweep_plot_compare] Could not resolve default comparison directory "
                            f"for {la} vs {lb}. Use --output-parent so comparisons live under "
                            f"<output-parent>/_comparisons/.",
                            file=sys.stderr,
                        )
                        continue

                plot_cmp_cmd = build_sweep_plot_compare_cmd(
                    compare_dir,
                    data_dir=args.data_dir,
                    compare_prefix=args.compare_prefix,
                    intervention_folders=args.intervention_folders,
                    best_interventions_top_n=args.best_interventions_top_n,
                    disagreement_top_n=args.disagreement_top_n,
                    plot_dpi=args.plot_dpi,
                    label_top_n=args.label_top_n,
                    min_family_points=args.min_family_points,
                )
                print(f"  {la} vs {lb} — sweep_plot_compare")
                run_cmd(plot_cmp_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
