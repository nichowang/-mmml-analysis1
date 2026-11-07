#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --mem=22000M
#SBATCH -A cis250037p
#SBATCH --cpus-per-task=5
#SBATCH -t 03:00:00
#SBATCH -J si_dl2
#SBATCH -o logs/download_extra_%j.log
# archives 10-59: training-only data for the learned reference encoder (Idea 2 midterm)
cd /ocean/projects/cis250037p/ywang134/mmml-analysis1
PY=/ocean/projects/cis250037p/ywang134/envs/aw310/bin/python
for g in "30 31 32 33 34" "35 36 37 38 39" "40 41 42 43 44" "45 46 47 48 49" "50 51 52 53 54" "55 56 57 58 59"; do
  $PY scripts/download.py data 0 $g &
done
wait
echo DONE
