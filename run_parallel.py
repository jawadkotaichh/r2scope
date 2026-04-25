"""Launch multiple RODE/PyMARL experiments in parallel as isolated subprocesses.

Each experiment is a completely independent process — independent SC2 instance,
independent replay buffer, independent policy, independent Sacred run folder.
Sharing an SC2 runner across different training runs is not safe (per-episode
game state leaks between policies), so we deliberately do not attempt it.

What this launcher *does* buy you:
  * concurrency cap so you don't accidentally spawn N_runs SC2 games at once
  * staggered SC2 boots (pysc2 map-load is the slowest startup step)
  * BLAS/OMP thread caps per child so N runs don't each grab all cores
  * deterministic per-run results folder (results/<alg>_seed<seed>_<env>_<map>/)

Usage:
    # inline matrix: all combinations of algs x maps x seeds
    python run_parallel.py --alg rode --map sc2_3m sc2_2s_vs_1sc --seed 0 1 2 \
        --env sc2 --max-parallel 3

    # from a YAML spec (list of runs, one dict per run)
    python run_parallel.py --spec experiments.yaml --max-parallel 4

YAML spec format:
    defaults:
      env: sc2
      extra: ["t_max=2050000"]
    runs:
      - {alg: rode, map: sc2_3m,       seed: 0}
      - {alg: rode, map: sc2_2s_vs_1sc, seed: 0}
      - {alg: qmix, map: sc2_3m,       seed: 1, extra: ["t_max=1000000"]}
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yaml
except ImportError:
    yaml = None

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN_PY = os.path.join(REPO_ROOT, "src", "main.py")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def build_cmd(alg, env, map_name, seed, extra, torch_threads, python_exe):
    cmd = [python_exe, MAIN_PY, "--alg={}".format(alg), "--env={}".format(env)]
    if map_name:
        cmd.append("--map={}".format(map_name))
    cmd.append("with")
    cmd.append("seed={}".format(seed))
    cmd.append("torch_num_threads={}".format(torch_threads))
    for e in extra or []:
        cmd.append(e)
    return cmd


def child_env(cpu_threads, gpu_id=None):
    # Cap BLAS/OpenMP thread pools per child. Default to all cores otherwise,
    # which wrecks throughput when several processes do it simultaneously.
    env = os.environ.copy()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[k] = str(cpu_threads)
    # Pin this child to a single GPU. Without this, every child defaults to
    # device 0 and the other allocated GPUs sit idle.
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def should_append_log(cmd):
    return any(
        arg == "auto_resume=True" or arg.startswith("checkpoint_path=")
        for arg in cmd
    )


def run_one(idx, cmd, cpu_threads, stagger_s, log_path, gpu_id=None, first_wave=0):
    # Stagger SC2 boots: if every subprocess starts at t=0, pysc2 map-load
    # contention and Blizzard license-server calls both slow down markedly.
    # Only the first wave needs staggering — later runs start when an earlier
    # one finishes, so there is no concurrent-boot contention to spread out.
    if idx < first_wave:
        time.sleep(idx * stagger_s)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    gpu_tag = " gpu={}".format(gpu_id) if gpu_id is not None else ""
    print("[launcher] start  [{}]{} -> {}".format(idx, gpu_tag, " ".join(cmd)), flush=True)
    log_mode = "a" if should_append_log(cmd) else "w"
    with open(log_path, log_mode) as lf:
        if log_mode == "a":
            lf.write("\n# ==== resume submission {} ====\n".format(
                time.strftime("%Y-%m-%d %H:%M:%S")))
        lf.write("# cmd: {}\n".format(" ".join(cmd)))
        if gpu_id is not None:
            lf.write("# CUDA_VISIBLE_DEVICES={}\n".format(gpu_id))
        lf.flush()
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT,
                             env=child_env(cpu_threads, gpu_id), cwd=REPO_ROOT)
    print("[launcher] finish [{}] rc={} log={}".format(idx, rc, log_path), flush=True)
    return rc


def resolve_gpus(explicit):
    # Priority: explicit --gpus > Slurm-set CUDA_VISIBLE_DEVICES > none.
    # Slurm sets CUDA_VISIBLE_DEVICES to the physical IDs it allocated to the
    # job (e.g. "0,1" or "3,7"); CUDA honors those IDs at the driver level,
    # so passing one of them to a child's env pins that child to that GPU.
    if explicit:
        return [str(g) for g in explicit]
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return []
    return [g.strip() for g in raw.split(",") if g.strip()]


def load_spec(path):
    if yaml is None:
        raise SystemExit("PyYAML is required for --spec. pip install pyyaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def runs_from_args(args):
    runs = []
    if args.spec:
        spec = load_spec(args.spec)
        defaults = spec.get("defaults", {}) or {}
        for r in spec.get("runs", []) or []:
            merged = {**defaults, **r}
            runs.append(merged)
        return runs

    combos = itertools.product(args.alg, args.map or [None], args.seed)
    for alg, mp, seed in combos:
        runs.append({"alg": alg, "map": mp, "seed": seed,
                     "env": args.env, "extra": args.extra})
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alg", nargs="+", default=["rode"])
    p.add_argument("--map", nargs="*", default=None,
                   help="config/maps/<name>.yaml (omit for env default map)")
    p.add_argument("--env", default="sc2")
    p.add_argument("--seed", nargs="+", type=int, default=[0])
    p.add_argument("--extra", nargs="*", default=[],
                   help="extra 'key=value' overrides appended after 'with'")
    p.add_argument("--spec", default=None,
                   help="YAML file describing runs (overrides --alg/--map/--seed)")
    p.add_argument("--max-parallel", type=int, default=max(1, (os.cpu_count() or 2) // 4),
                   help="concurrent experiments; each SC2 run is CPU-heavy")
    p.add_argument("--cpu-threads", type=int, default=1,
                   help="OMP/MKL/BLAS threads per child process")
    p.add_argument("--stagger", type=float, default=8.0,
                   help="seconds between subprocess starts (SC2 boot is slow)")
    p.add_argument("--python", default=sys.executable,
                   help="python interpreter for children (default: the one running this launcher)")
    p.add_argument("--gpus", nargs="*", default=None,
                   help="GPU ids to round-robin across children (e.g. 0 1). "
                        "Defaults to parsing CUDA_VISIBLE_DEVICES (set by Slurm).")
    args = p.parse_args()

    # subprocess resolves the executable before cwd is applied, so relative
    # paths like "./venv/bin/python" break. Normalize to an absolute path.
    args.python = os.path.abspath(args.python)
    if not os.path.isfile(args.python):
        raise SystemExit("--python not found: {}".format(args.python))

    runs = runs_from_args(args)
    if not runs:
        raise SystemExit("No runs to launch.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    launcher_log_dir = os.path.join(RESULTS_DIR, "_launcher")
    os.makedirs(launcher_log_dir, exist_ok=True)

    gpus = resolve_gpus(args.gpus)
    print("[launcher] {} runs, max_parallel={}, cpu_threads/child={}, stagger={}s, gpus={}".format(
        len(runs), args.max_parallel, args.cpu_threads, args.stagger,
        gpus if gpus else "none (inherit env)"), flush=True)

    jobs = []
    for i, r in enumerate(runs):
        cmd = build_cmd(
            alg=r["alg"], env=r.get("env", args.env),
            map_name=r.get("map"), seed=r["seed"],
            extra=r.get("extra", []), torch_threads=args.cpu_threads,
            python_exe=args.python,
        )
        # One console log per run; the real per-run artifacts live under
        # results/<alg>_seed<seed>_<env>_<map>/ (chosen by src/main.py).
        tag = "{}_seed{}_{}_{}".format(
            r["alg"], r["seed"], r.get("env", args.env), r.get("map") or "default")
        log_path = os.path.join(launcher_log_dir, tag + ".log")
        # Round-robin on submission index so the first wave of max_parallel
        # runs is evenly split across GPUs.
        gpu_id = gpus[i % len(gpus)] if gpus else None
        jobs.append((i, cmd, log_path, gpu_id))

    first_wave = min(args.max_parallel, len(jobs))
    failures = 0
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futs = [pool.submit(run_one, i, cmd, args.cpu_threads, args.stagger,
                            log_path, gpu_id, first_wave)
                for (i, cmd, log_path, gpu_id) in jobs]
        for f in as_completed(futs):
            if f.result() != 0:
                failures += 1

    print("[launcher] done. failures={}/{}".format(failures, len(jobs)), flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
