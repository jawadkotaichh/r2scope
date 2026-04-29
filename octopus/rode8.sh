#!/bin/bash

#SBATCH --job-name=rode8
#SBATCH --account=rhe34

#SBATCH --partition=msfea-ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32000
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --time=3-00:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=rhe34@mail.aub.edu

source ~/.bashrc
set -euo pipefail
module purge
module load python/ai-4 || module load python/3
module load cmake/3.15.4 || module load cmake
module list
python3 --version
cmake --version
python3 - <<'PY'
import sys
if sys.version_info < (3, 9):
    raise SystemExit("Python 3.9+ is required; loaded {}".format(sys.version.split()[0]))
PY

cd "${SLURM_SUBMIT_DIR}"
export SC2PATH="${PWD}/3rdparty/StarCraftII"

python3 -m venv --clear .venv8
source .venv8/bin/activate
python --version
python3 -m pip install --upgrade pip "setuptools<82" wheel
python3 -m pip install "dm-tree==0.1.8" -r requirements.txt
python3 -m pip install --upgrade torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda devices:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing to run GPU experiment on CPU.")
PY
[ -d 3rdparty/StarCraftII/Versions ] || bash install_sc2.sh
extra_args=()
if [ "${AUTO_RESUME:-0}" = "1" ]; then
  extra_args+=(--extra auto_resume=True)
fi

python3 run_parallel.py --alg rode --seed 2 --max-parallel 4 --map sc2_27m_vs_30m sc2_3s_vs_5z sc2_2c_vs_64zg sc2_3s5z_vs_3s6z ${extra_args[@]+"${extra_args[@]}"}
exec sleep infinity
