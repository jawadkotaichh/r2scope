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


def map_matches(run_dir: Path, wanted_map: str) -> bool:
    wanted = clean_map_name(wanted_map)
    config_map = run_map_name(run_dir)
    if config_map is not None:
        return clean_map_name(str(config_map)) == wanted

    return run_dir.name.endswith(f"_{wanted}") or run_dir.name.endswith(f"_sc2_{wanted}")


def discover_run_dirs(root: Path, map_name: str):
    run_dirs = [p for p in root.iterdir() if p.is_dir() and map_matches(p, map_name)]
    return sorted(run_dirs, key=lambda p: (run_seed(p), p.name))


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


def find_info_path(run_dir: Path) -> Path:
    candidates = []
    for sacred_run_dir in numeric_sacred_dirs(run_dir):
        info_path = sacred_run_dir / "info.json"
        if not info_path.exists():
            continue
        try:
            info = load_json(info_path)
        except json.JSONDecodeError:
            continue
        max_t, point_count = metric_progress(info)
        candidates.append((max_t, point_count, int(sacred_run_dir.name), info_path))

    if not candidates:
        raise FileNotFoundError(f"Could not find a usable sacred/*/info.json under {run_dir}")

    # Pick the attempt with the most logged training progress. The numeric
    # Sacred id is only a tie-breaker for repeated attempts with equal progress.
    return max(candidates, key=lambda item: item[:3])[3]


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


def aggregate_metric(run_infos, metric_name):
    curves = []

    for run_name, info in run_infos:
        x, y = get_metric(info, metric_name)
        if x is None:
            continue
        curves.append((run_name, x, y))

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
        help="Base directory where plots will be saved. A map-name and seed-count subfolder is always created.",
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

    run_infos = []
    loaded_seeds = []
    for run_dir in run_dirs:
        try:
            info_path = find_info_path(run_dir)
        except FileNotFoundError as exc:
            print(f"Skipping {run_dir.name}: {exc}")
            continue
        info = load_json(info_path)
        max_t, point_count = metric_progress(info)
        run_infos.append((run_dir.name, info))
        loaded_seeds.append(run_seed(run_dir))
        max_t_label = int(max_t) if max_t >= 0 else "unknown"
        print(f"Loaded: {info_path} (max_t={max_t_label}, points={point_count})")

    if not run_infos:
        raise SystemExit(f"No usable sacred info.json files found for map '{map_name}'")

    seed_count = len(set(loaded_seeds))
    outdir = outdir_base / f"{safe_folder_name(map_name)}_{seed_count}seeds"
    outdir.mkdir(parents=True, exist_ok=True)

    x, ys = aggregate_metric(run_infos, "return_mean")
    if x is not None:
        plot_mean_std(
            x=x,
            ys=ys,
            title=f"{map_name}: Averaged Score vs Timesteps",
            ylabel="Averaged Score",
            out_file=outdir / "avg_score_mean_std.png",
        )
    else:
        print("Metric return_mean not found in the selected runs.")

    x, ys = aggregate_metric(run_infos, "test_return_mean")
    if x is not None:
        plot_mean_std(
            x=x,
            ys=ys,
            title=f"{map_name}: Test Averaged Score vs Timesteps",
            ylabel="Test Averaged Score",
            out_file=outdir / "test_score_mean_std.png",
        )
    else:
        print("Metric test_return_mean not found in the selected runs.")

    x, ys = aggregate_metric(run_infos, "test_battle_won_mean")
    if x is not None:
        plot_median_iqr(
            x=x,
            ys=ys,
            title=f"{map_name}: Test Win Rate vs Timesteps",
            ylabel="Test Win %",
            out_file=outdir / "test_win_paper_style.png",
            scale=100.0,
            ylim=(0, 100),
        )
    else:
        print("Metric test_battle_won_mean not found in the selected runs.")

    print(f"Saved plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
