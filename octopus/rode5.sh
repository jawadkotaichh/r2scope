#!/bin/bash

#SBATCH --job-name=rode5
#SBATCH --account=rhe34

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=gpu:v100d32q:2
#SBATCH --time=0-06:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=rhe34@mail.aub.edu

cd "$(dirname "$0")/.."

python -m venv .venv5
source .venv5/bin/activate
pip install -r requirements.txt
pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cu121
[ -d 3rdparty/StarCraftII/Versions ] || bash install_sc2.sh
python run_parallel.py --alg rode --seed 0 1 --max-parallel 4 --map sc2_3s5z sc2_bane_vs_bane