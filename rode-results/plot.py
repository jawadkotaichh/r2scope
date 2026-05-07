import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PROGRESS_METRICS = (
    "test_battle_won_mean",
    "test_return_mean",
    "return_mean",
)

# Only these two algorithms will be plotted
TARGET_ALGORITHMS = ["rode_ices", "rode"]
REQUIRED_SEEDS = (0, 1, 2, 3)

ALGORITHM_DISPLAY_NAMES = {
    "rode_ices": "R2SCOPE",
    "rode": "RODE",
}

ALGORITHM_COLORS = {
    "rode_ices": "tab:red",
    "rode": "tab:blue",
}

FALLBACK_COLORS = (
    "tab:purple",
    "tab:blue",
    "tab:green",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_map_name(map_name: str) -> str:
    map_name = map_name.strip()
    if map_name.startswith("sc2_"):
        return map_name[4:]
    return map_name


def display_map_name(map_name: str) -> str:
    return clean_map_name(str(map_name)).replace("_", " ")


def safe_folder_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "map"


def normalize_algorithm_name(name: str) -> str:
    name = str(name).lower().strip()
    name = name.replace("-", "_")

    if name in ["rode_ices", "rodeices"]:
        return "rode_ices"

    if name == "rode":
        return "rode"

    return name


@lru_cache(maxsize=None)
def numeric_sacred_dirs(run_dir: Path):
    sacred_dir = run_dir / "sacred"
    if not sacred_dir.is_dir():
        return []

    return sorted(
        [p for p in sacred_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


@lru_cache(maxsize=None)
def run_config(run_dir: Path):
    for sacred_run_dir in reversed(numeric_sacred_dirs(run_dir)):
        config_path = sacred_run_dir / "config.json"

        if not config_path.exists():
            continue

        try:
            return load_json(config_path)
        except json.JSONDecodeError:
            continue

    return {}


@lru_cache(maxsize=None)
def run_map_name(run_dir: Path):
    config = run_config(run_dir)
    return config.get("env_args", {}).get("map_name")


@lru_cache(maxsize=None)
def run_seed(run_dir: Path):
    config = run_config(run_dir)

    if config.get("seed") is not None:
        try:
            return int(config["seed"])
        except (TypeError, ValueError):
            pass

    match = re.search(r"_seed(\d+)_", run_dir.name)
    return int(match.group(1)) if match else 10**12


@lru_cache(maxsize=None)
def run_algorithm(run_dir: Path):
    config = run_config(run_dir)

    if config.get("name"):
        return normalize_algorithm_name(config["name"])

    match = re.match(r"(.+?)_seed\d+_", run_dir.name)

    if match:
        return normalize_algorithm_name(match.group(1))

    return normalize_algorithm_name(run_dir.name)


@lru_cache(maxsize=None)
def run_dir_map_name(run_dir: Path):
    config_map = run_map_name(run_dir)

    if config_map is not None:
        return clean_map_name(str(config_map))

    match = re.search(r"_sc2_(.+)$", run_dir.name)
    return clean_map_name(match.group(1)) if match else None


def algorithm_color(algorithm: str, algorithm_order):
    algorithm = normalize_algorithm_name(algorithm)

    if algorithm in ALGORITHM_COLORS:
        return ALGORITHM_COLORS[algorithm]

    idx = algorithm_order.index(algorithm) if algorithm in algorithm_order else 0
    return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]


def map_matches(run_dir: Path, wanted_map: str) -> bool:
    wanted = clean_map_name(wanted_map)
    return run_dir_map_name(run_dir) == wanted


def discover_run_dirs(root: Path, map_name: str):
    run_dirs = [p for p in root.iterdir() if p.is_dir() and map_matches(p, map_name)]

    return sorted(run_dirs, key=lambda p: (run_algorithm(p), run_seed(p), p.name))


def discover_run_dirs_by_map(root: Path):
    grouped = {}

    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue

        map_name = run_dir_map_name(run_dir)

        if map_name is None:
            continue

        grouped.setdefault(map_name, []).append(run_dir)

    return {
        map_name: sorted(
            run_dirs,
            key=lambda p: (run_algorithm(p), run_seed(p), p.name),
        )
        for map_name, run_dirs in sorted(grouped.items())
    }


def metric_progress(info, progress_metrics=DEFAULT_PROGRESS_METRICS):
    best_t = -1.0
    point_count = 0

    for metric_name in progress_metrics:
        t_key = f"{metric_name}_T"
        values = info.get(t_key)

        if not values:
            continue

        numeric_values = [float(v) for v in values]
        best_t = max(best_t, max(numeric_values))
        point_count = max(point_count, len(numeric_values))

    if best_t >= 0:
        return best_t, point_count

    for key, values in info.items():
        if not key.endswith("_T") or not values:
            continue

        numeric_values = [float(v) for v in values]
        best_t = max(best_t, max(numeric_values))
        point_count = max(point_count, len(numeric_values))

    return best_t, point_count


def metric_pairs(info):
    names = []

    for key, values in info.items():
        if key.endswith("_T"):
            continue

        t_key = f"{key}_T"

        if t_key in info and values and info[t_key]:
            names.append(key)

    return sorted(names)


def sorted_unique_last(x, y):
    points = {}

    for t, value in zip(x, y):
        points[float(t)] = float(value)

    if not points:
        return [], []

    ordered_t = sorted(points)
    return ordered_t, [points[t] for t in ordered_t]


def append_attempt_metric(stitched_x, stitched_y, attempt_x, attempt_y):
    x = np.asarray(attempt_x, dtype=float)
    y = np.asarray(attempt_y, dtype=float)

    if len(x) == 0 or len(y) == 0:
        return stitched_x, stitched_y

    length = min(len(x), len(y))
    x = x[:length]
    y = y[:length]

    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    x, y = sorted_unique_last(x, y)

    if stitched_x:
        resume_t = x[0]
        kept = [(t, value) for t, value in zip(stitched_x, stitched_y) if t < resume_t]

        stitched_x = [t for t, _ in kept]
        stitched_y = [value for _, value in kept]

    stitched_x.extend(x)
    stitched_y.extend(y)

    return sorted_unique_last(stitched_x, stitched_y)


def load_attempt_infos(run_dir: Path):
    attempts = []

    for sacred_run_dir in numeric_sacred_dirs(run_dir):
        info_path = sacred_run_dir / "info.json"

        if not info_path.exists():
            continue

        try:
            info = load_json(info_path)
        except json.JSONDecodeError:
            continue

        max_t, point_count = metric_progress(info)

        if max_t < 0:
            continue

        attempts.append(
            (
                int(sacred_run_dir.name),
                info_path,
                info,
                max_t,
                point_count,
            )
        )

    if not attempts:
        raise FileNotFoundError(
            f"Could not find a usable sacred/*/info.json under {run_dir}"
        )

    return sorted(attempts, key=lambda item: item[0])


def stitch_attempt_infos(run_dir: Path):
    attempts = load_attempt_infos(run_dir)
    stitched = {}

    metric_names = sorted(
        {name for _, _, info, _, _ in attempts for name in metric_pairs(info)}
    )

    for metric_name in metric_names:
        stitched_x = []
        stitched_y = []
        t_key = f"{metric_name}_T"

        for _, _, info, _, _ in attempts:
            if metric_name not in info or t_key not in info:
                continue

            stitched_x, stitched_y = append_attempt_metric(
                stitched_x,
                stitched_y,
                info[t_key],
                info[metric_name],
            )

        if stitched_x:
            stitched[metric_name] = stitched_y
            stitched[t_key] = stitched_x

    return stitched, attempts


def get_metric(info, metric_name):
    x_key = f"{metric_name}_T"

    if metric_name not in info or x_key not in info:
        return None, None

    x = np.asarray(info[x_key], dtype=float)
    y = np.asarray(info[metric_name], dtype=float)

    return x, y


def deduplicate_xy(x, y):
    points = {}

    for xi, yi in zip(x, y):
        if np.isfinite(xi) and np.isfinite(yi):
            points[float(xi)] = float(yi)

    if not points:
        return np.array([]), np.array([])

    xs = np.array(sorted(points.keys()), dtype=float)
    ys = np.array([points[xi] for xi in xs], dtype=float)

    return xs, ys


def interp_to_ref(ref_x, x, y):
    x, y = deduplicate_xy(x, y)

    if len(x) == 0:
        return np.full_like(ref_x, np.nan, dtype=float)

    return np.interp(ref_x, x, y)


def smooth_curve(y, window=3):
    if window <= 1 or len(y) < window:
        return y

    kernel = np.ones(window) / window
    padded = np.pad(
        y,
        (window // 2, window - 1 - window // 2),
        mode="edge",
    )

    return np.convolve(padded, kernel, mode="valid")


def aggregate_metric(run_records, metric_name, n_points=250):
    curves = []

    for record in run_records:
        info = record["info"]
        x, y = get_metric(info, metric_name)

        if x is None:
            continue

        x, y = deduplicate_xy(x, y)

        if len(x) < 2:
            continue

        curves.append((record["run_name"], x, y))

    if not curves:
        return None, None

    # Common timestep range shared by all seeds
    start = max(x[0] for _, x, _ in curves)
    end = min(x[-1] for _, x, _ in curves)

    if end <= start:
        ref_idx = np.argmin([len(x) for _, x, _ in curves])
        ref_x = curves[ref_idx][1]
    else:
        ref_x = np.linspace(start, end, n_points)

    ys = []

    for _, x, y in curves:
        ys.append(interp_to_ref(ref_x, x, y))

    ys = np.vstack(ys)

    return ref_x, ys


def records_by_algorithm(run_records):
    grouped = {}

    for record in run_records:
        algorithm = normalize_algorithm_name(record["algorithm"])

        # Keep only R2SCOPE and RODE
        if algorithm not in TARGET_ALGORITHMS:
            continue

        record["algorithm"] = algorithm
        grouped.setdefault(algorithm, []).append(record)

    ordered_grouped = {}

    for algorithm in TARGET_ALGORITHMS:
        if algorithm in grouped:
            ordered_grouped[algorithm] = sorted(
                grouped[algorithm],
                key=lambda r: (r["seed"], r["run_name"]),
            )

    return ordered_grouped


def missing_required_seeds(grouped_records, required_seeds=REQUIRED_SEEDS):
    missing = {}

    for algorithm in TARGET_ALGORITHMS:
        available_seeds = {int(record["seed"]) for record in grouped_records.get(algorithm, [])}
        missing_seeds = [seed for seed in required_seeds if seed not in available_seeds]

        if missing_seeds:
            missing[algorithm] = missing_seeds

    return missing


def format_missing_requirements(missing_seeds):
    issues = []

    for algorithm in TARGET_ALGORITHMS:
        if algorithm not in missing_seeds:
            continue

        issues.append(
            f"{ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm)} missing seed(s) "
            f"{', '.join(str(seed) for seed in missing_seeds[algorithm])}"
        )

    return "; ".join(issues)


def plot_mean_std(x, ys, title, ylabel, out_file, scale=1.0, ylim=None):
    y_mean = ys.mean(axis=0) * scale
    y_std = ys.std(axis=0) * scale
    x_plot = x / 1e6

    plt.figure(figsize=(4.2, 2.4))

    plt.plot(x_plot, y_mean, linewidth=2, label="Mean")
    plt.fill_between(
        x_plot,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.22,
        label="+/- 1 std",
    )

    plt.xlabel("T (mil)")
    plt.ylabel(ylabel)
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.grid(True, alpha=0.55, linewidth=0.8)
    plt.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_algorithm_comparison(
    grouped_records,
    metric_name,
    title,
    ylabel,
    out_file,
    scale=1.0,
    ylim=None,
    center="median",
    smooth_window=3,
    show_legend=True,
):
    algorithm_order = list(grouped_records)

    plt.figure(figsize=(4.2, 2.4))
    ax = plt.gca()

    plotted = False

    for algorithm, records in grouped_records.items():
        x, ys = aggregate_metric(records, metric_name)

        if x is None:
            continue

        y_scaled = ys * scale

        if center == "median":
            mid = np.nanmedian(y_scaled, axis=0)
            low = np.nanpercentile(y_scaled, 25, axis=0)
            high = np.nanpercentile(y_scaled, 75, axis=0)
        else:
            mid = np.nanmean(y_scaled, axis=0)
            spread = np.nanstd(y_scaled, axis=0)
            low = mid - spread
            high = mid + spread

        mid = smooth_curve(mid, window=smooth_window)
        low = smooth_curve(low, window=smooth_window)
        high = smooth_curve(high, window=smooth_window)

        x_plot = x / 1e6
        color = algorithm_color(algorithm, algorithm_order)

        ax.plot(
            x_plot,
            mid,
            linewidth=2.0,
            color=color,
            label=ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm),
        )

        if len(records) > 1:
            ax.fill_between(
                x_plot,
                low,
                high,
                color=color,
                alpha=0.22,
                linewidth=0,
            )

        plotted = True

    if not plotted:
        plt.close()
        return False

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("T (mil)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlim(left=0)

    ax.grid(True, alpha=0.55, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=9)

    if show_legend:
        ax.legend(fontsize=7, frameon=False)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    return True


def plot_single_curve(x, y, title, ylabel, out_file, scale=1.0, ylim=None, color=None):
    x_plot = x / 1e6
    y_plot = y * scale

    plt.figure(figsize=(4.2, 2.4))

    plt.plot(x_plot, y_plot, linewidth=2, color=color)

    plt.xlabel("T (mil)")
    plt.ylabel(ylabel)
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlim(left=0)
    plt.grid(True, alpha=0.55, linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_median_iqr(x, ys, title, ylabel, out_file, scale=1.0, ylim=None):
    y_median = np.median(ys, axis=0) * scale
    y_p25 = np.percentile(ys, 25, axis=0) * scale
    y_p75 = np.percentile(ys, 75, axis=0) * scale
    x_plot = x / 1e6

    plt.figure(figsize=(4.2, 2.4))

    plt.plot(x_plot, y_median, linewidth=2, label="Median")
    plt.fill_between(
        x_plot,
        y_p25,
        y_p75,
        alpha=0.22,
        label="25-75 percentile",
    )

    plt.xlabel("T (mil)")
    plt.ylabel(ylabel)
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlim(left=0)
    plt.grid(True, alpha=0.55, linewidth=0.8)
    plt.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def write_manifest(outdir: Path, map_name: str, grouped_records, skipped_runs):
    manifest = {
        "map": display_map_name(map_name),
        "map_key": map_name,
        "plotted_algorithms": list(grouped_records.keys()),
        "algorithms": {},
        "skipped_runs": skipped_runs,
    }

    lines = [f"Map: {display_map_name(map_name)}", f"Map key: {map_name}", ""]

    for algorithm, records in grouped_records.items():
        seeds = [record["seed"] for record in records]

        manifest["algorithms"][algorithm] = {
            "display_name": ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm),
            "seeds": seeds,
            "runs": [
                {
                    "run_name": record["run_name"],
                    "seed": record["seed"],
                    "max_t": record["max_t"],
                    "point_count": record["point_count"],
                    "attempts": record["attempt_labels"],
                }
                for record in records
            ],
        }

        lines.append(
            f"{ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm)}: "
            f"seeds {', '.join(str(seed) for seed in seeds)}"
        )

        for record in records:
            lines.append(
                f"  - {record['run_name']} seed={record['seed']} "
                f"max_t={record['max_t']} points={record['point_count']} "
                f"attempts={record['attempt_labels']}"
            )

        lines.append("")

    if skipped_runs:
        lines.append("Skipped runs:")

        for skipped in skipped_runs:
            lines.append(f"  - {skipped['run_name']}: {skipped['reason']}")

    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    with open(outdir / "manifest.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def load_run_records(run_dirs):
    run_records = []
    skipped_runs = []

    for run_dir in run_dirs:
        algorithm = run_algorithm(run_dir)

        # Skip anything that is not R2SCOPE or RODE
        if algorithm not in TARGET_ALGORITHMS:
            print(
                f"Skipping {run_dir.name}: algorithm '{algorithm}' is not R2SCOPE or RODE"
            )
            continue

        try:
            info, attempts = stitch_attempt_infos(run_dir)
        except FileNotFoundError as exc:
            print(f"Skipping {run_dir.name}: {exc}")
            skipped_runs.append({"run_name": run_dir.name, "reason": str(exc)})
            continue

        max_t, point_count = metric_progress(info)
        seed = run_seed(run_dir)

        max_t_label = int(max_t) if max_t >= 0 else "unknown"

        attempt_labels = ", ".join(
            f"{info_path.parent.name}:{int(attempt_max_t)}"
            for _, info_path, _, attempt_max_t, _ in attempts
        )

        run_records.append(
            {
                "run_dir": run_dir,
                "run_name": run_dir.name,
                "algorithm": algorithm,
                "seed": seed,
                "info": info,
                "attempts": attempts,
                "attempt_labels": attempt_labels,
                "max_t": max_t_label,
                "point_count": point_count,
            }
        )

        print(
            f"Loaded: {run_dir.name} "
            f"[{ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm)}, seed={seed}] "
            f"stitched {len(attempts)} attempt(s) "
            f"(max_t={max_t_label}, points={point_count}; attempts={attempt_labels})"
        )

    return run_records, skipped_runs


def plot_map_outputs(map_name: str, outdir_base: Path, grouped_records, skipped_runs):
    run_records = [record for records in grouped_records.values() for record in records]
    outdir = outdir_base / safe_folder_name(map_name)
    outdir.mkdir(parents=True, exist_ok=True)

    individual_dir = outdir / "individual_runs"
    individual_dir.mkdir(parents=True, exist_ok=True)

    write_manifest(outdir, map_name, grouped_records, skipped_runs)
    map_title = display_map_name(map_name)

    # Combined train score plot
    if not plot_algorithm_comparison(
        grouped_records=grouped_records,
        metric_name="return_mean",
        title=map_title,
        ylabel="Averaged Score",
        out_file=outdir / "combined_avg_score.png",
        center="median",
        smooth_window=3,
        show_legend=True,
    ):
        print(f"{map_name}: metric return_mean not found in the selected runs.")

    # Combined test return plot
    if not plot_algorithm_comparison(
        grouped_records=grouped_records,
        metric_name="test_return_mean",
        title=map_title,
        ylabel="Test Averaged Score",
        out_file=outdir / "combined_test_score.png",
        center="median",
        smooth_window=3,
        show_legend=True,
    ):
        print(f"{map_name}: metric test_return_mean not found in the selected runs.")

    # Combined test win-rate plot
    if not plot_algorithm_comparison(
        grouped_records=grouped_records,
        metric_name="test_battle_won_mean",
        title=map_title,
        ylabel="Test Win %",
        out_file=outdir / "combined_test_win.png",
        scale=100.0,
        ylim=(0, 100),
        center="median",
        smooth_window=3,
        show_legend=True,
    ):
        print(f"{map_name}: metric test_battle_won_mean not found in the selected runs.")

    # Individual plots for each selected run
    per_run_specs = (
        ("return_mean", "Averaged Score", "avg_score.png", 1.0, None),
        ("test_return_mean", "Test Averaged Score", "test_score.png", 1.0, None),
        ("test_battle_won_mean", "Test Win %", "test_win.png", 100.0, (0, 100)),
    )

    algorithm_order = list(grouped_records)

    for record in run_records:
        run_name = record["run_name"]
        info = record["info"]

        run_outdir = individual_dir / safe_folder_name(run_name)
        run_outdir.mkdir(parents=True, exist_ok=True)

        for metric_name, ylabel, filename, scale, ylim in per_run_specs:
            x, y = get_metric(info, metric_name)

            if x is None:
                continue

            plot_single_curve(
                x=x,
                y=y,
                title=f"{run_name}: {ylabel}",
                ylabel=ylabel,
                out_file=run_outdir / filename,
                scale=scale,
                ylim=ylim,
                color=algorithm_color(record["algorithm"], algorithm_order),
            )

    return outdir


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "map_name",
        nargs="?",
        default=None,
        help=(
            "Optional SC2 map name to plot, e.g. 6h_vs_8z, MMM2, or sc2_6h_vs_8z. "
            "If omitted, all maps under the root are checked and only complete ones are plotted."
        ),
    )

    parser.add_argument(
        "--root",
        default=None,
        help="Root directory containing the run folders. Defaults to this script's directory.",
    )

    parser.add_argument(
        "--outdir",
        default=None,
        help="Base directory where plots will be saved. A map-name subfolder is always created.",
    )

    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Optional explicit run folder names. If omitted, runs are discovered from the map name.",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).expanduser().resolve() if args.root else script_dir
    outdir_base = Path(args.outdir).expanduser() if args.outdir else root / "plots"

    if args.runs:
        run_dirs = [root / run_name for run_name in args.runs]
        missing_run_dirs = [str(run_dir) for run_dir in run_dirs if not run_dir.is_dir()]

        if missing_run_dirs:
            raise SystemExit(
                "These run folders do not exist:\n" + "\n".join(missing_run_dirs)
            )

        if args.map_name:
            requested_map = clean_map_name(args.map_name)
            discovered_maps = sorted(
                {
                    map_name
                    for map_name in (run_dir_map_name(run_dir) for run_dir in run_dirs)
                    if map_name is not None
                }
            )

            if discovered_maps and discovered_maps != [requested_map]:
                raise SystemExit(
                    f"The provided --runs belong to map(s) {', '.join(discovered_maps)}, "
                    f"not '{requested_map}'."
                )

            map_to_run_dirs = {requested_map: run_dirs}
        else:
            discovered_maps = sorted(
                {
                    map_name
                    for map_name in (run_dir_map_name(run_dir) for run_dir in run_dirs)
                    if map_name is not None
                }
            )

            if len(discovered_maps) != 1:
                raise SystemExit(
                    "When using --runs without map_name, all runs must belong to exactly one map."
                )

            map_to_run_dirs = {discovered_maps[0]: run_dirs}
    elif args.map_name:
        map_name = clean_map_name(args.map_name)
        map_to_run_dirs = {map_name: discover_run_dirs(root, map_name)}
    else:
        map_to_run_dirs = discover_run_dirs_by_map(root)

    if not map_to_run_dirs:
        raise SystemExit(f"No run folders found under {root}")

    plotted_maps = []
    skipped_maps = []

    for map_name, run_dirs in map_to_run_dirs.items():
        if not run_dirs:
            reason = f"No run folders found for map '{map_name}' under {root}"
            print(f"Skipping map {map_name}: {reason}")
            skipped_maps.append({"map": map_name, "reason": reason})
            continue

        print(f"Evaluating map {map_name} ({len(run_dirs)} run folder(s))")
        run_records, skipped_runs = load_run_records(run_dirs)

        if not run_records:
            reason = (
                f"No usable R2SCOPE or RODE sacred info.json files found for map '{map_name}'"
            )
            print(f"Skipping map {map_name}: {reason}")
            skipped_maps.append({"map": map_name, "reason": reason})
            continue

        grouped_records = records_by_algorithm(run_records)

        if not grouped_records:
            reason = (
                f"No R2SCOPE or RODE runs found for map '{map_name}' after filtering."
            )
            print(f"Skipping map {map_name}: {reason}")
            skipped_maps.append({"map": map_name, "reason": reason})
            continue

        missing_seeds = missing_required_seeds(grouped_records)

        if missing_seeds:
            reason = format_missing_requirements(missing_seeds)
            print(f"Skipping map {map_name}: {reason}")
            skipped_maps.append({"map": map_name, "reason": reason})
            continue

        outdir = plot_map_outputs(map_name, outdir_base, grouped_records, skipped_runs)
        plotted_maps.append(map_name)
        print(f"Saved plots to: {outdir.resolve()}")

    if plotted_maps:
        print("Plotted maps: " + ", ".join(plotted_maps))

    if skipped_maps:
        print(
            "Skipped map names: "
            + ", ".join(skipped["map"] for skipped in skipped_maps)
        )
        print("Skipped maps:")

        for skipped in skipped_maps:
            print(f"  - {skipped['map']}: {skipped['reason']}")

    if not plotted_maps:
        raise SystemExit(
            "No maps satisfied the requirement: both R2SCOPE and RODE must have "
            f"usable results for seeds {', '.join(str(seed) for seed in REQUIRED_SEEDS)}."
        )


if __name__ == "__main__":
    main()
