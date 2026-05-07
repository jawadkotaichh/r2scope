import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot as result_plot


DEFAULT_METRIC = "test_battle_won_mean"
DEFAULT_THRESHOLD = 0.0


def first_crossing(info, metric_name, threshold):
    x, y = result_plot.get_metric(info, metric_name)

    if x is None or y is None:
        return None

    x, y = result_plot.deduplicate_xy(x, y)

    if len(x) == 0:
        return None

    for timestep, value in zip(x, y):
        if value > threshold:
            return {
                "first_win_t": float(timestep),
                "first_win_value": float(value),
                "max_t": float(x[-1]),
                "max_value": float(y[-1]),
                "point_count": int(len(x)),
            }

    return {
        "first_win_t": None,
        "first_win_value": None,
        "max_t": float(x[-1]),
        "max_value": float(y[-1]),
        "point_count": int(len(x)),
    }


def metric_label(metric_name):
    labels = {
        "test_battle_won_mean": "Test win rate",
        "battle_won_mean": "Train win rate",
    }

    return labels.get(metric_name, metric_name)


def collect_rows_for_map(map_name, run_dirs, metric_name, threshold):
    run_records, skipped_runs = result_plot.load_run_records(run_dirs)
    grouped_records = result_plot.records_by_algorithm(run_records)
    rows = []
    map_display = result_plot.display_map_name(map_name)

    for algorithm, records in grouped_records.items():
        for record in records:
            crossing = first_crossing(record["info"], metric_name, threshold)

            if crossing is None:
                skipped_runs.append(
                    {
                        "run_name": record["run_name"],
                        "reason": f"Metric '{metric_name}' was not found.",
                    }
                )
                continue

            first_win_t = crossing["first_win_t"]
            max_t = crossing["max_t"]

            rows.append(
                {
                    "map": map_display,
                    "map_key": map_name,
                    "algorithm": algorithm,
                    "algorithm_display": result_plot.ALGORITHM_DISPLAY_NAMES.get(
                        algorithm, algorithm
                    ),
                    "seed": int(record["seed"]),
                    "run_name": record["run_name"],
                    "metric": metric_name,
                    "threshold": float(threshold),
                    "first_win": first_win_t is not None,
                    "first_win_t": first_win_t,
                    "first_win_million_steps": (
                        first_win_t / 1e6 if first_win_t is not None else None
                    ),
                    "first_win_value": crossing["first_win_value"],
                    "max_t": max_t,
                    "max_million_steps": max_t / 1e6 if max_t is not None else None,
                    "max_value": crossing["max_value"],
                    "point_count": crossing["point_count"],
                    "attempts": record["attempt_labels"],
                }
            )

    return rows, skipped_runs


def finite_first_win_times(rows):
    values = [
        row["first_win_t"]
        for row in rows
        if row["first_win"] and row["first_win_t"] is not None
    ]

    return np.asarray(values, dtype=float)


def median_or_none(values):
    if len(values) == 0:
        return None

    return float(np.median(values))


def summarize(rows, skipped):
    by_map_alg = {}

    for row in rows:
        key = (row["map"], row["algorithm"])
        by_map_alg.setdefault(key, []).append(row)

    summaries = []

    for (map_name, algorithm), group in sorted(by_map_alg.items()):
        times = finite_first_win_times(group)
        total = len(group)
        wins = int(sum(1 for row in group if row["first_win"]))

        summaries.append(
            {
                "map": map_name,
                "algorithm": algorithm,
                "algorithm_display": result_plot.ALGORITHM_DISPLAY_NAMES.get(
                    algorithm, algorithm
                ),
                "seeds": sorted(int(row["seed"]) for row in group),
                "runs": total,
                "first_win_runs": wins,
                "first_win_rate": wins / total if total else None,
                "median_first_win_t": median_or_none(times),
                "median_first_win_million_steps": (
                    median_or_none(times) / 1e6 if len(times) else None
                ),
                "min_first_win_t": float(np.min(times)) if len(times) else None,
                "max_first_win_t": float(np.max(times)) if len(times) else None,
            }
        )

    return {
        "rows": rows,
        "summaries": summaries,
        "skipped": skipped,
    }


def write_outputs(outdir, payload):
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "time_to_first_win.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    return json_path


def grouped_rows_by_algorithm(rows):
    grouped = {}

    for row in rows:
        grouped.setdefault(row["algorithm"], []).append(row)

    return {
        algorithm: grouped[algorithm]
        for algorithm in result_plot.TARGET_ALGORITHMS
        if algorithm in grouped
    }


def plot_map_time_to_first_win(map_name, rows, out_file, metric_name, threshold):
    grouped = grouped_rows_by_algorithm(rows)

    if not grouped:
        return False

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    algorithms = list(grouped)
    x_positions = np.arange(len(algorithms), dtype=float)
    algorithm_order = list(grouped)
    max_seen = 0.0
    plotted = False

    for idx, algorithm in enumerate(algorithms):
        group = sorted(grouped[algorithm], key=lambda row: row["seed"])
        color = result_plot.algorithm_color(algorithm, algorithm_order)
        times = finite_first_win_times(group)
        med = median_or_none(times)

        if med is not None:
            ax.bar(
                idx,
                med / 1e6,
                width=0.52,
                color=color,
                alpha=0.26,
                edgecolor=color,
                linewidth=1.2,
                label="Median first win" if idx == 0 else None,
            )
            max_seen = max(max_seen, med)
            plotted = True

        for seed_idx, row in enumerate(group):
            jitter = (seed_idx - (len(group) - 1) / 2.0) * 0.055

            if row["first_win"]:
                y_value = row["first_win_t"] / 1e6
                marker = "o"
                scatter_kwargs = {
                    "color": color,
                    "edgecolor": "white",
                    "linewidth": 0.8,
                }
                alpha = 0.95
                max_seen = max(max_seen, row["first_win_t"])
            else:
                y_value = row["max_t"] / 1e6
                marker = "x"
                scatter_kwargs = {
                    "color": color,
                    "linewidth": 1.4,
                }
                alpha = 0.9
                max_seen = max(max_seen, row["max_t"])

            ax.scatter(
                idx + jitter,
                y_value,
                s=42,
                marker=marker,
                alpha=alpha,
                zorder=3,
                **scatter_kwargs,
            )

    if not plotted and max_seen == 0:
        plt.close(fig)
        return False

    ax.set_title(
        f"{result_plot.display_map_name(map_name)}: Time to first win",
        fontsize=11,
    )
    ax.set_ylabel("Timestep (million env steps)", fontsize=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [result_plot.ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm) for algorithm in algorithms],
        fontsize=9,
    )
    ax.set_xlim(-0.55, len(algorithms) - 0.45)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.5, linewidth=0.8)
    ax.tick_params(axis="y", labelsize=9)

    subtitle = f"{metric_label(metric_name)} > {threshold:g}; x marker means no first win by last eval"
    ax.text(
        0.0,
        -0.24,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return True


def plot_summary_by_map(summaries, out_file):
    map_names = sorted({row["map"] for row in summaries})

    if not map_names:
        return False

    algorithms = [
        algorithm
        for algorithm in result_plot.TARGET_ALGORITHMS
        if any(row["algorithm"] == algorithm for row in summaries)
    ]

    if not algorithms:
        return False

    lookup = {(row["map"], row["algorithm"]): row for row in summaries}
    x = np.arange(len(map_names), dtype=float)
    width = min(0.34, 0.72 / max(len(algorithms), 1))

    fig_width = max(6.0, len(map_names) * 0.62)
    fig, ax = plt.subplots(figsize=(fig_width, 3.2))

    for alg_idx, algorithm in enumerate(algorithms):
        offset = (alg_idx - (len(algorithms) - 1) / 2.0) * width
        values = []

        for map_name in map_names:
            row = lookup.get((map_name, algorithm))
            values.append(
                np.nan
                if row is None or row["median_first_win_t"] is None
                else row["median_first_win_t"] / 1e6
            )

        color = result_plot.algorithm_color(algorithm, algorithms)
        ax.bar(
            x + offset,
            values,
            width=width,
            color=color,
            alpha=0.72,
            label=result_plot.ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm),
        )

    ax.set_title("Median time to first win by map", fontsize=11)
    ax.set_ylabel("Timestep (million env steps)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [result_plot.display_map_name(map_name) for map_name in map_names],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.5, linewidth=0.8)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return True


def build_map_to_run_dirs(args, root):
    if args.runs:
        run_dirs = [root / run_name for run_name in args.runs]
        missing = [str(run_dir) for run_dir in run_dirs if not run_dir.is_dir()]

        if missing:
            raise SystemExit("These run folders do not exist:\n" + "\n".join(missing))

        maps = sorted(
            {
                map_name
                for map_name in (result_plot.run_dir_map_name(run_dir) for run_dir in run_dirs)
                if map_name is not None
            }
        )

        if args.map_name:
            requested_map = result_plot.clean_map_name(args.map_name)

            if maps and maps != [requested_map]:
                raise SystemExit(
                    f"The provided --runs belong to map(s) {', '.join(maps)}, "
                    f"not '{requested_map}'."
                )

            return {requested_map: run_dirs}

        if len(maps) != 1:
            raise SystemExit(
                "When using --runs without map_name, all runs must belong to exactly one map."
            )

        return {maps[0]: run_dirs}

    if args.map_name:
        map_name = result_plot.clean_map_name(args.map_name)
        return {map_name: result_plot.discover_run_dirs(root, map_name)}

    return result_plot.discover_run_dirs_by_map(root)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute time to first win from Sacred info.json metrics and plot "
            "R2SCOPE vs RODE comparisons."
        )
    )
    parser.add_argument(
        "map_name",
        nargs="?",
        default=None,
        help="Optional SC2 map name, e.g. 10m_vs_11m. If omitted, all maps are scanned.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Root directory containing run folders. Defaults to this script's directory.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help=(
            "Output directory. Defaults to <root>/plots/time_to_first_win. "
            "Per-map plots are saved below this directory."
        ),
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Optional explicit run folder names to include.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=(
            "Metric used to detect the first win. Defaults to test_battle_won_mean. "
            "Use battle_won_mean for training-window wins."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "First-win threshold. Defaults to 0, so any non-zero win rate counts. "
            "For example, --threshold 0.5 finds the first evaluation above 50%%."
        ),
    )

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).expanduser().resolve() if args.root else script_dir
    outdir = Path(args.outdir).expanduser() if args.outdir else root / "plots" / "time_to_first_win"
    outdir = outdir.resolve()

    map_to_run_dirs = build_map_to_run_dirs(args, root)

    if not map_to_run_dirs:
        raise SystemExit(f"No run folders found under {root}")

    all_rows = []
    all_skipped = []
    plotted_maps = []

    for map_name, run_dirs in map_to_run_dirs.items():
        if not run_dirs:
            all_skipped.append(
                {
                    "map": map_name,
                    "run_name": None,
                    "reason": f"No run folders found for map '{map_name}' under {root}",
                }
            )
            continue

        print(f"Evaluating map {map_name} ({len(run_dirs)} run folder(s))")
        rows, skipped = collect_rows_for_map(
            map_name=map_name,
            run_dirs=run_dirs,
            metric_name=args.metric,
            threshold=args.threshold,
        )
        all_rows.extend(rows)
        all_skipped.extend(
            [{"map": map_name, **skipped_run} for skipped_run in skipped]
        )

        if rows:
            map_outdir = outdir / result_plot.safe_folder_name(map_name)
            map_outdir.mkdir(parents=True, exist_ok=True)
            plotted = plot_map_time_to_first_win(
                map_name=map_name,
                rows=rows,
                out_file=map_outdir / "time_to_first_win.png",
                metric_name=args.metric,
                threshold=args.threshold,
            )

            if plotted:
                plotted_maps.append(map_name)
                print(f"Saved map plot to: {map_outdir.resolve()}")

    if not all_rows:
        raise SystemExit(
            f"No usable rows found for metric '{args.metric}'. Check the metric name."
        )

    payload = summarize(all_rows, all_skipped)
    json_path = write_outputs(outdir, payload)

    summary_plot = outdir / "median_time_to_first_win_by_map.png"
    if plot_summary_by_map(payload["summaries"], summary_plot):
        print(f"Saved summary plot to: {summary_plot.resolve()}")

    print(f"Wrote JSON: {json_path.resolve()}")

    if plotted_maps:
        print("Plotted maps: " + ", ".join(plotted_maps))


if __name__ == "__main__":
    main()
