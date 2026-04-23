import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def find_info_path(root: Path, run_name: str) -> Path:
    path = root / run_name / "sacred" / "1" / "info.json"
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    return path


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

    # use shortest x-grid as reference
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
    plt.fill_between(x_plot, y_mean - y_std, y_mean + y_std, alpha=0.2, label="±1 std")
    plt.xlabel("T (mil)")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()


def plot_median_iqr(x, ys, title, ylabel, out_file, scale=1.0, ylim=None):
    y_median = np.median(ys, axis=0) * scale
    y_p25 = np.percentile(ys, 25, axis=0) * scale
    y_p75 = np.percentile(ys, 75, axis=0) * scale
    x_plot = x / 1e6

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_median, linewidth=2, label="Median")
    plt.fill_between(x_plot, y_p25, y_p75, alpha=0.2, label="25–75 percentile")
    plt.xlabel("T (mil)")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_names",
        nargs="+",
        help="Space-separated run folder names. Each must contain sacred/1/info.json",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory containing the run folders",
    )
    parser.add_argument(
        "--outdir",
        default="rode_plots",
        help="Directory where plots will be saved",
    )
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_infos = []
    for run_name in args.run_names:
        info_path = find_info_path(root, run_name)
        info = load_json(info_path)
        run_infos.append((run_name, info))
        print(f"Loaded: {info_path}")

    # 1) Averaged training score vs timesteps, mean ± std across seeds
    x, ys = aggregate_metric(run_infos, "return_mean")
    if x is not None:
        plot_mean_std(
            x=x,
            ys=ys,
            title="Averaged Score vs Timesteps",
            ylabel="Averaged Score",
            out_file=outdir / "avg_score_mean_std.png",
        )
    else:
        print("Metric return_mean not found in the provided runs.")

    # 2) Test averaged score vs timesteps, mean ± std across seeds
    x, ys = aggregate_metric(run_infos, "test_return_mean")
    if x is not None:
        plot_mean_std(
            x=x,
            ys=ys,
            title="Test Averaged Score vs Timesteps",
            ylabel="Test Averaged Score",
            out_file=outdir / "test_score_mean_std.png",
        )
    else:
        print("Metric test_return_mean not found in the provided runs.")

    # 3) Paper-style test win % vs timesteps, median + 25–75 percentile
    x, ys = aggregate_metric(run_infos, "test_battle_won_mean")
    if x is not None:
        plot_median_iqr(
            x=x,
            ys=ys,
            title="Test Win Rate vs Timesteps",
            ylabel="Test Win %",
            out_file=outdir / "test_win_paper_style.png",
            scale=100.0,
            ylim=(0, 100),
        )
    else:
        print("Metric test_battle_won_mean not found in the provided runs.")

    print(f"Saved plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
    