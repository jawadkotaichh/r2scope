#!/bin/bash

#SBATCH --job-name=rode7
#SBATCH --account=rhe34

#SBATCH --partition=msfea-ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=gpu:v100d32q:2
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

python3 -m venv --clear .venv7
source .venv7/bin/activate
python --version
python3 -m pip install --upgrade pip "setuptools<82" wheel
python3 -m pip install "dm-tree==0.1.8" -r requirements.txt
python3 -m pip install --pre --upgrade torch --extra-index-url https://download.pytorch.org/whl/nightly/cu121
[ -d 3rdparty/StarCraftII/Versions ] || bash install_sc2.sh
extra_args=()
if [ "${AUTO_RESUME:-0}" = "1" ]; then
  extra_args+=(--extra auto_resume=True)
fi
python3 run_parallel.py --spec octopus/rode7.yaml --max-parallel 3 "${extra_args[@]}"
