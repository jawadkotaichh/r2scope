import argparse
import json
import re
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

ALGORITHM_COLORS = {
    "rode": "tab:blue",
    "rode_ices": "tab:red",
    "rode-ices": "tab:red",
}

FALLBACK_COLORS = (
    "tab:green",
    "tab:orange",
    "tab:purple",
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


def safe_folder_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "map"


def numeric_sacred_dirs(run_dir: Path):
    sacred_dir = run_dir / "sacred"
    if not sacred_dir.is_dir():
        return []
    return sorted(
        [p for p in sacred_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


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


def run_map_name(run_dir: Path):
    config = run_config(run_dir)
    return config.get("env_args", {}).get("map_name")


def run_seed(run_dir: Path):
    config = run_config(run_dir)
    if config.get("seed") is not None:
        return config["seed"]
    match = re.search(r"_seed(\d+)_", run_dir.name)
    return int(match.group(1)) if match else 10**12


def run_algorithm(run_dir: Path):
    config = run_config(run_dir)
    if config.get("name"):
        return str(config["name"])

    match = re.match(r"(.+?)_seed\d+_", run_dir.name)
    if match:
        return match.group(1)

    return run_dir.name


def algorithm_color(algorithm: str, algorithm_order):
    key = algorithm.lower().replace("-", "_")
    if key in ALGORITHM_COLORS:
        return ALGORITHM_COLORS[key]
    if algorithm in ALGORITHM_COLORS:
        return ALGORITHM_COLORS[algorithm]
    idx = algorithm_order.index(algorithm) if algorithm in algorithm_order else 0
    return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]


def map_matches(run_dir: Path, wanted_map: str) -> bool:
    wanted = clean_map_name(wanted_map)
    config_map = run_map_name(run_dir)
    if config_map is not None:
        return clean_map_name(str(config_map)) == wanted

    return run_dir.name.endswith(f"_{wanted}") or run_dir.name.endswith(f"_sc2_{wanted}")


def discover_run_dirs(root: Path, map_name: str):
    run_dirs = [p for p in root.iterdir() if p.is_dir() and map_matches(p, map_name)]
    return sorted(run_dirs, key=lambda p: (run_algorithm(p), run_seed(p), p.name))


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
        attempts.append((int(sacred_run_dir.name), info_path, info, max_t, point_count))

    if not attempts:
        raise FileNotFoundError(f"Could not find a usable sacred/*/info.json under {run_dir}")

    return sorted(attempts, key=lambda item: item[0])


def stitch_attempt_infos(run_dir: Path):
    attempts = load_attempt_infos(run_dir)
    stitched = {}

    metric_names = sorted({name for _, _, info, _, _ in attempts for name in metric_pairs(info)})
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


def interp_to_ref(ref_x, x, y):
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    unique_x, idx = np.unique(x, return_index=True)
    unique_y = y[idx]

    return np.interp(ref_x, unique_x, unique_y)


def aggregate_metric(run_records, metric_name):
    curves = []

    for record in run_records:
        info = record["info"]
        x, y = get_metric(info, metric_name)
        if x is None:
            continue
        curves.append((record["run_name"], x, y))

    if not curves:
        return None, None

    # Use the shortest x-grid as the interpolation reference.
    ref_idx = np.argmin([len(x) for _, x, _ in curves])
    ref_x = curves[ref_idx][1]

    ys = []
    for _, x, y in curves:
        ys.append(interp_to_ref(ref_x, x, y))

    ys = np.vstack(ys)
    return ref_x, ys


def records_by_algorithm(run_records):
    grouped = {}
    for record in run_records:
        grouped.setdefault(record["algorithm"], []).append(record)
    return {
        algorithm: sorted(records, key=lambda r: (r["seed"], r["run_name"]))
        for algorithm, records in sorted(grouped.items())
    }


def plot_mean_std(x, ys, title, ylabel, out_file, scale=1.0, ylim=None):
    y_mean = ys.mean(axis=0) * scale
    y_std = ys.std(axis=0) * scale
    x_plot = x / 1e6

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_mean, linewidth=2, label="Mean")
    plt.fill_between(x_plot, y_mean - y_std, y_mean + y_std, alpha=0.2, label="+/- 1 std")
    plt.xlabel("T (mil)")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_algorithm_comparison(grouped_records, metric_name, title, ylabel, out_file, scale=1.0, ylim=None,
                              center="mean"):
    algorithm_order = list(grouped_records)

    plt.figure(figsize=(10, 6))
    plotted = False
    for algorithm, records in grouped_records.items():
        x, ys = aggregate_metric(records, metric_name)
        if x is None:
            continue

        y_scaled = ys * scale
        if center == "median":
            mid = np.median(y_scaled, axis=0)
            low = np.percentile(y_scaled, 25, axis=0)
            high = np.percentile(y_scaled, 75, axis=0)
            band_label = "25-75 percentile"
        else:
            mid = y_scaled.mean(axis=0)
            spread = y_scaled.std(axis=0)
            low = mid - spread
            high = mid + spread
            band_label = "+/- 1 std"

        color = algorithm_color(algorithm, algorithm_order)
        seed_labels = ", ".join(str(record["seed"]) for record in records)
        label = f"{algorithm} (seeds: {seed_labels})"
        plt.plot(x / 1e6, mid, linewidth=2, label=label, color=color)
        if len(records) > 1:
            plt.fill_between(x / 1e6, low, high, alpha=0.18, color=color, label=f"{algorithm} {band_label}")
        plotted = True

    if not plotted:
        plt.close()
        return False

    plt.xlabel("T (mil)")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_single_curve(x, y, title, ylabel, out_file, scale=1.0, ylim=None, color=None):
    x_plot = x / 1e6
    y_plot = y * scale

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, linewidth=2, color=color)
    plt.xlabel("T (mil)")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_median_iqr(x, ys, title, ylabel, out_file, scale=1.0, ylim=None):
    y_median = np.median(ys, axis=0) * scale
    y_p25 = np.percentile(ys, 25, axis=0) * scale
    y_p75 = np.percentile(ys, 75, axis=0) * scale
    x_plot = x / 1e6

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_median, linewidth=2, label="Median")
    plt.fill_between(x_plot, y_p25, y_p75, alpha=0.2, label="25-75 percentile")
    plt.xlabel("T (mil)")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def write_manifest(outdir: Path, map_name: str, grouped_records, skipped_runs):
    manifest = {
        "map": map_name,
        "algorithms": {},
        "skipped_runs": skipped_runs,
    }

    lines = [f"Map: {map_name}", ""]
    for algorithm, records in grouped_records.items():
        seeds = [record["seed"] for record in records]
        manifest["algorithms"][algorithm] = {
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
        lines.append(f"{algorithm}: seeds {', '.join(str(seed) for seed in seeds)}")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "map_name",
        help="SC2 map name to plot, e.g. 6h_vs_8z or sc2_6h_vs_8z",
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
    map_name = clean_map_name(args.map_name)
    outdir_base = Path(args.outdir).expanduser() if args.outdir else root / "plots"

    if args.runs:
        run_dirs = [root / run_name for run_name in args.runs]
    else:
        run_dirs = discover_run_dirs(root, map_name)

    if not run_dirs:
        raise SystemExit(f"No run folders found for map '{map_name}' under {root}")

    run_records = []
    skipped_runs = []
    for run_dir in run_dirs:
        try:
            info, attempts = stitch_attempt_infos(run_dir)
        except FileNotFoundError as exc:
            print(f"Skipping {run_dir.name}: {exc}")
            skipped_runs.append({"run_name": run_dir.name, "reason": str(exc)})
            continue
        max_t, point_count = metric_progress(info)
        seed = run_seed(run_dir)
        algorithm = run_algorithm(run_dir)
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
            f"Loaded: {run_dir.name} [{algorithm}, seed={seed}] stitched {len(attempts)} attempt(s) "
            f"(max_t={max_t_label}, points={point_count}; attempts={attempt_labels})"
        )

    if not run_records:
        raise SystemExit(f"No usable sacred info.json files found for map '{map_name}'")

    grouped_records = records_by_algorithm(run_records)
    outdir = outdir_base / safe_folder_name(map_name)
    outdir.mkdir(parents=True, exist_ok=True)
    individual_dir = outdir / "individual_runs"
    individual_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(outdir, map_name, grouped_records, skipped_runs)

    if not plot_algorithm_comparison(
            grouped_records=grouped_records,
            metric_name="return_mean",
            title=f"{map_name}: Averaged Score vs Timesteps",
            ylabel="Averaged Score",
            out_file=outdir / "combined_avg_score.png",
    ):
        print("Metric return_mean not found in the selected runs.")

    if not plot_algorithm_comparison(
            grouped_records=grouped_records,
            metric_name="test_return_mean",
            title=f"{map_name}: Test Averaged Score vs Timesteps",
            ylabel="Test Averaged Score",
            out_file=outdir / "combined_test_score.png",
    ):
        print("Metric test_return_mean not found in the selected runs.")

    if not plot_algorithm_comparison(
            grouped_records=grouped_records,
            metric_name="test_battle_won_mean",
            title=f"{map_name}: Test Win Rate vs Timesteps",
            ylabel="Test Win %",
            out_file=outdir / "combined_test_win.png",
            scale=100.0,
            ylim=(0, 100),
            center="median",
    ):
        print("Metric test_battle_won_mean not found in the selected runs.")

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
                title=f"{run_name}: {ylabel} vs Timesteps",
                ylabel=ylabel,
                out_file=run_outdir / filename,
                scale=scale,
                ylim=ylim,
                color=algorithm_color(record["algorithm"], algorithm_order),
            )

    print(f"Saved plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
