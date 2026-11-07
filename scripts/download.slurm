#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --mem=22000M
#SBATCH -A cis250037p
#SBATCH --cpus-per-task=5
#SBATCH -t 04:00:00
#SBATCH -J si_dl
#SBATCH -o logs/download_%j.log
cd /ocean/projects/cis250037p/ywang134/mmml-analysis1
PY=/ocean/projects/cis250037p/ywang134/envs/aw310/bin/python
# archives 0-2 = the three used in the team's earlier reports; 3-19 extend coverage
for g in "0 1 2 3 4" "5 6 7 8 9" "10 11 12 13 14" "15 16 17 18 19"; do
  $PY scripts/download.py data 0 $g &
done
wait
