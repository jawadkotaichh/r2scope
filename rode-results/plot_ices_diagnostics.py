import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot as result_plot


ICES_ALGORITHM = "rode_ices"

DIFFICULTY_GROUPS = {
    "easy": [
        "1c3s5z",
        "2s3z",
        "2s_vs_1sc",
        "3s5z",
        "10m_vs_11m",
    ],
    "hard": [
        "2c_vs_64zg",
        "3s_vs_5z",
        "5m_vs_6m",
        "MMM2",
    ],
    "super_hard": [
        "3s5z_vs_3s6z",
        "6h_vs_8z",
        "corridor",
        "27m_vs_30m",
    ],
}

METRICS = {
    "test_win": "test_battle_won_mean",
    "explorer_usage": "ices_fraction_explore_actions",
    "intrinsic_mean": "ices_intrinsic_mean",
    "intrinsic_std": "ices_intrinsic_std",
    "explorer_entropy": "ices_entropy",
}


def median_iqr(ys):
    return (
        np.nanmedian(ys, axis=0),
        np.nanpercentile(ys, 25, axis=0),
        np.nanpercentile(ys, 75, axis=0),
    )


def smooth(y, window=3):
    return result_plot.smooth_curve(np.asarray(y, dtype=float), window=window)


def aggregate(records, metric_name, n_points=260):
    return result_plot.aggregate_metric(records, metric_name, n_points=n_points)


def curve_stats(records, metric_name):
    x, ys = aggregate(records, metric_name)

    if x is None:
        return None

    mid, low, high = median_iqr(ys)
    return {
        "x": x,
        "mid": smooth(mid),
        "low": smooth(low),
        "high": smooth(high),
        "n": len(records),
    }


def plot_iqr_line(ax, stats, color, label, linewidth=1.8, linestyle="-", alpha=0.16):
    if stats is None:
        return False

    x_plot = stats["x"] / 1e6
    ax.plot(
        x_plot,
        stats["mid"],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )

    if stats["n"] > 1:
        ax.fill_between(
            x_plot,
            stats["low"],
            stats["high"],
            color=color,
            alpha=alpha,
            linewidth=0,
        )

    return True


def load_grouped_records(root):
    records_by_map = {}
    skipped = []

    for map_name, run_dirs in result_plot.discover_run_dirs_by_map(root).items():
        run_records, skipped_runs = result_plot.load_run_records(run_dirs)
        grouped = result_plot.records_by_algorithm(run_records)
        skipped.extend(skipped_runs)

        if ICES_ALGORITHM in grouped:
            records_by_map[map_name] = grouped

    known_maps = {map_name for maps in DIFFICULTY_GROUPS.values() for map_name in maps}
    unknown_maps = sorted(map_name for map_name in records_by_map if map_name not in known_maps)

    if unknown_maps:
        DIFFICULTY_GROUPS["other"] = unknown_maps

    return records_by_map, skipped


def set_missing(ax):
    ax.text(
        0.5,
        0.5,
        "missing",
        ha="center",
        va="center",
        fontsize=8,
        color="0.45",
        transform=ax.transAxes,
    )


def plot_test_win(ax, grouped, show_legend):
    plotted = False
    algorithm_order = [alg for alg in result_plot.TARGET_ALGORITHMS if alg in grouped]

    for algorithm in algorithm_order:
        stats = curve_stats(grouped[algorithm], METRICS["test_win"])
        color = result_plot.algorithm_color(algorithm, algorithm_order)
        plotted = plot_iqr_line(
            ax,
            stats,
            color=color,
            label=result_plot.ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm),
            alpha=0.14,
        ) or plotted

    if not plotted:
        set_missing(ax)

    ax.set_ylim(-0.04, 1.04)

    if show_legend and plotted:
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")


def plot_usage(ax, records, show_legend):
    stats = curve_stats(records, METRICS["explorer_usage"])
    plotted = plot_iqr_line(
        ax,
        stats,
        color=result_plot.algorithm_color(ICES_ALGORITHM, [ICES_ALGORITHM]),
        label="Usage",
        alpha=0.16,
    )

    if not plotted:
        set_missing(ax)
        return

    upper = max(0.12, float(np.nanmax(stats["high"])) * 1.15)
    ax.set_ylim(0, min(1.0, upper))

    if show_legend:
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")


def plot_intrinsic(ax, records, show_legend):
    mean_stats = curve_stats(records, METRICS["intrinsic_mean"])
    std_stats = curve_stats(records, METRICS["intrinsic_std"])
    color = result_plot.algorithm_color(ICES_ALGORITHM, [ICES_ALGORITHM])

    plotted = plot_iqr_line(
        ax,
        mean_stats,
        color=color,
        label="Mean",
        alpha=0.16,
    )
    plotted = plot_iqr_line(
        ax,
        std_stats,
        color="tab:orange",
        label="Std",
        linewidth=1.3,
        linestyle="--",
        alpha=0.08,
    ) or plotted

    if not plotted:
        set_missing(ax)

    if show_legend and plotted:
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")


def plot_entropy(ax, records, show_legend):
    stats = curve_stats(records, METRICS["explorer_entropy"])
    plotted = plot_iqr_line(
        ax,
        stats,
        color="tab:purple",
        label="Entropy",
        alpha=0.16,
    )

    if not plotted:
        set_missing(ax)

    if show_legend and plotted:
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")


def style_axis(ax, row_idx, col_idx, n_rows):
    ax.grid(True, alpha=0.35, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlim(left=0)

    if row_idx == n_rows - 1:
        ax.set_xlabel("Steps (M)", fontsize=8)
    else:
        ax.tick_params(labelbottom=False)

    if col_idx != 0:
        ax.tick_params(labelleft=False)


def plot_group_dashboard(group_name, map_names, records_by_map, out_file):
    available_maps = [map_name for map_name in map_names if map_name in records_by_map]

    if not available_maps:
        return []

    row_labels = [
        "Test win",
        "Explorer usage",
        "Intrinsic mean/std",
        "Explorer entropy",
    ]
    plotters = [
        lambda ax, grouped, records, show_legend: plot_test_win(ax, grouped, show_legend),
        lambda ax, grouped, records, show_legend: plot_usage(ax, records, show_legend),
        lambda ax, grouped, records, show_legend: plot_intrinsic(ax, records, show_legend),
        lambda ax, grouped, records, show_legend: plot_entropy(ax, records, show_legend),
    ]

    n_rows = len(row_labels)
    n_cols = len(available_maps)
    fig_width = max(9.2, 3.0 * n_cols)
    fig_height = 8.2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

    for col_idx, map_name in enumerate(available_maps):
        grouped = records_by_map[map_name]
        r2scope_records = grouped[ICES_ALGORITHM]
        axes[0, col_idx].set_title(result_plot.display_map_name(map_name), fontsize=10)

        for row_idx, plotter in enumerate(plotters):
            ax = axes[row_idx, col_idx]
            plotter(ax, grouped, r2scope_records, show_legend=(col_idx == 0))
            style_axis(ax, row_idx, col_idx, n_rows)

            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=8)

    fig.suptitle(
        f"R2SCOPE Diagnostics: {group_name.replace('_', ' ').title()} Maps",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.982))
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return available_maps


def write_manifest(outdir, records_by_map, generated, skipped):
    payload = {
        "output_root": str(outdir),
        "notes": [
            "Dashboards are grouped by common SMAC difficulty buckets.",
            "Each dashboard column is one map; each row is one diagnostic signal.",
            "Lines are medians across available seeds; shaded bands show the 25-75 percentile range across seeds.",
            "Intrinsic row uses ices_intrinsic_mean as the solid line and ices_intrinsic_std as the dashed line.",
        ],
        "difficulty_groups": DIFFICULTY_GROUPS,
        "generated_plots": generated,
        "maps": {},
        "skipped_runs": skipped,
    }

    for map_name, grouped in sorted(records_by_map.items()):
        payload["maps"][map_name] = {
            "map": result_plot.display_map_name(map_name),
            "algorithms": {
                algorithm: {
                    "display_name": result_plot.ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm),
                    "seeds": sorted(int(record["seed"]) for record in records),
                    "runs": len(records),
                }
                for algorithm, records in grouped.items()
            },
        }

    with open(outdir / "compact_ices_diagnostics_manifest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot compact R2SCOPE/ICES diagnostics by map difficulty."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing rode-results run folders.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots" / "ices_diagnostics_compact",
        help="Directory where compact diagnostic plots will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    records_by_map, skipped = load_grouped_records(args.root)
    generated = {}

    for group_name, map_names in DIFFICULTY_GROUPS.items():
        out_file = args.outdir / f"{group_name}_diagnostics_dashboard.png"
        plotted_maps = plot_group_dashboard(group_name, map_names, records_by_map, out_file)

        if plotted_maps:
            generated[f"{group_name}_dashboard"] = str(out_file)
            print(f"Saved {group_name} dashboard: {out_file}")

    write_manifest(args.outdir, records_by_map, generated, skipped)
    print(f"Wrote manifest: {args.outdir / 'compact_ices_diagnostics_manifest.json'}")


if __name__ == "__main__":
    main()
