# R2SCOPE

This repository builds on **RODE: Learning Roles to Decompose Multi-Agent Tasks** for cooperative multi-agent reinforcement learning on the StarCraft Multi-Agent Challenge (SMAC).

RODE discovers roles by decomposing the joint action space according to action effects. This codebase keeps the original RODE training stack, which is based on [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac), and adds an R2SCOPE variant through the `rode_ices` algorithm config.

R2SCOPE adds an ICES-style scaffold and auxiliary explorer on top of RODE. The explorer changes training-time primitive action sampling while preserving RODE's role-scoped action constraints. The main comparison in this repo is therefore:

- `rode`: the RODE baseline.
- `rode_ices`: R2SCOPE.

The SMAC environment has also been modified so enemy units are sorted before attack-action IDs are assigned. In vanilla SMAC, the same attack action ID can refer to different enemy units after environment resets, which makes action representations less stable.

## Repository Layout

- `src/`: training code, algorithms, controllers, learners, modules, and configs.
- `src/config/algs/rode.yaml`: RODE baseline config.
- `src/config/algs/rode_ices.yaml`: R2SCOPE config.
- `src/config/maps/`: map-specific overrides.
- `src/modules/ices/`: ICES scaffold and explorer modules.
- `run_parallel.py`: launches many independent training runs with a concurrency limit.
- `install_sc2.sh`: installs StarCraft II and SMAC maps into `3rdparty/`.
- `rode-results/`: collected experiment outputs and plotting scripts.
- `rode-results/plots/`: generated figures and plot manifests.

## Setup

The easiest supported setup is Docker:

```bash
cd docker
bash build.sh
```

Then install StarCraft II and the SMAC maps:

```bash
bash install_sc2.sh
```

The installer downloads StarCraft II into `3rdparty/StarCraftII` and copies the maps needed by SMAC.

You can also install the Python dependencies directly:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Direct virtualenv installs are more sensitive to PyTorch, SMAC, and StarCraft II version issues than the Docker path.

## Running Experiments

Run one RODE experiment:

```bash
python src/main.py --alg=rode --env=sc2 --map=sc2_2s3z with seed=0 t_max=2050000
```

Run one R2SCOPE experiment:

```bash
python src/main.py --alg=rode_ices --env=sc2 --map=sc2_2s3z with seed=0 t_max=2050000
```

Run several maps/seeds in parallel:

```bash
python run_parallel.py \
  --alg rode rode_ices \
  --map sc2_2s3z sc2_3s5z_vs_3s6z \
  --seed 0 1 2 3 \
  --max-parallel 4 \
  --extra t_max=2050000
```

Each child process gets its own SC2 instance, replay buffer, Sacred folder, and result directory.

## Results

Training runs write to deterministic folders under `results/`:

```text
results/<algorithm>_seed<seed>_<env>_<map>/
```

For example:

```text
results/rode_ices_seed0_sc2_2s3z/
```

Important files inside each run:

- `sacred/<id>/config.json`: full resolved config.
- `sacred/<id>/info.json`: logged metrics used by the plotting scripts.
- `sacred/<id>/cout.txt`: captured console output.
- `models/<timestep>/`: saved checkpoints.
- `tb/`: TensorBoard event files.
- `replay/`: saved SC2 replays, when replay saving is enabled.

The `rode-results/` directory contains collected runs and analysis outputs. The plotting scripts there expect run folders to live directly under `rode-results/`.

## Plotting

Regenerate the main score, return, and win-rate plots:

```bash
python rode-results/plot.py
```

Regenerate time-to-first-win plots:

```bash
python rode-results/time_to_first_win.py
```

Regenerate compact R2SCOPE diagnostics:

```bash
python rode-results/plot_ices_diagnostics.py
```

Generated figures are written under:

```text
rode-results/plots/
```

Useful plot folders:

- `rode-results/plots/<map>/`: main RODE vs R2SCOPE curves for each map.
- `rode-results/plots/time_to_first_win/`: per-map and summary time-to-first-win plots.
- `rode-results/plots/ices_diagnostics_compact/`: compact diagnostics dashboards for R2SCOPE.
- `rode-results/plots/roles/`: role/action-space visualizations, when generated.

The current compact diagnostics dashboards group maps by difficulty and include:

- test win rate
- explorer usage
- intrinsic mean/std
- explorer entropy

## Resuming And Evaluating

If a run already has checkpoints, resume it from the latest local checkpoint with:

```bash
python src/main.py --alg=rode_ices --env=sc2 --map=sc2_2s3z with seed=0 auto_resume=True
```

You can also load a specific checkpoint directory:

```bash
python src/main.py --alg=rode_ices --env=sc2 --map=sc2_2s3z with checkpoint_path=results/rode_ices_seed0_sc2_2s3z/models load_step=2050000 evaluate=True
```

To save replays during evaluation, add:

```bash
save_replay=True runner=episode
```

Linux StarCraft II can save replays, but viewing `.SC2Replay` files usually requires the Windows or macOS StarCraft II client.
