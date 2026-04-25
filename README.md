__Modifications to the SMAC Environments: In the SMAC framework, an attacking action is tied to a specific enemy unit. However, when the environment is reset at each episode, the same action ID may correspond to different enemies. This variability alters the semantics of actions across different episodes, making it difficult to learn consistent and meaningful action representations. Therefore, we sorted enemies in the SMAC environment file.__

# RODE: Learning Roles to Decompose Multi-Agent Tasks

RODE ([ArXiv Link](https://arxiv.org/pdf/2010.01523.pdf)) is a scalable role-based multi-agent learning method which effectively discovers roles based on joint action space decomposition according to action effects. It establishes a new state of the art on the StarCraft multi-agent benchmark.

This implementation is written in PyTorch and is based on [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac).

The results of RODE on the SMAC benchmark can be found [here](https://drive.google.com/file/d/1iZUoZO2x-rNBIxhfxn9txhsc4pVReU38/view?usp=sharing).

## Installation instructions

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Octopus setup

The `octopus/` folder contains Slurm job scripts for running the experiments on AUB Octopus.

Each job script currently:

- loads the Octopus Python and CMake modules
- recreates a per-job virtual environment such as `.venv1`
- installs the Python requirements and nightly PyTorch
- installs StarCraft II and SMAC if they are missing
- launches `run_parallel.py` with the maps and seeds assigned to that job

### Submit a job

From the repository root on Octopus:

```shell
sbatch octopus/rode1.sh
```

The other prepared jobs can be submitted in the same way:

```shell
sbatch octopus/rode2.sh
sbatch octopus/rode3.sh
sbatch octopus/rode4.sh
sbatch octopus/rode5.sh
sbatch octopus/rode6.sh
```

### Logs and status

Slurm writes the batch log to:

```shell
slurm-<jobid>.out
```

Useful commands:

```shell
squeue -u $USER
scontrol show job <jobid>
cat slurm-<jobid>.out
```

The parallel launcher also writes one log per run under:

```shell
results/_launcher/
```

For example:

```shell
cat results/_launcher/rode_seed0_sc2_sc2_27m_vs_30m.log
```

### Virtual environments on Octopus

The job scripts create hidden virtual environments in the repository root, such as `.venv1`, `.venv2`, and so on. Because they are hidden directories, `ls` will not show them unless you use:

```shell
ls -a
```

### Line endings

Shell scripts must use Unix line endings on Octopus. This repository uses `.gitattributes` to keep `*.sh` files on `LF`. If a script was copied manually and Slurm reports DOS line break errors, convert it with:

```shell
sed -i 's/\r$//' install_sc2.sh octopus/*.sh
```

### Notes

- The Octopus jobs currently pin `setuptools<82` because `sacred==0.8.5` still imports `pkg_resources`.
- The requirements include `dm-tree==0.1.8` so Octopus can install a wheel instead of trying to build a newer `dm-tree` with a newer CMake than the cluster provides.
- `sc2_27m_vs_30m` uses a smaller replay buffer override because the default RODE buffer is too large for that map on Octopus RAM.

## Run an experiment 

```shell
python3 src/main.py --config=rode --env-config=sc2 with env_args.map_name=corridor n_role_clusters=3 role_interval=5 t_max=5050000
```

To change the annealing time of epsilon, set `epsilon_anneal_time_exp`.

All results will be stored in the `Results` folder.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.
