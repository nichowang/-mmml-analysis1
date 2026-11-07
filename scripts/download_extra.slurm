#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --mem=22000M
#SBATCH -A cis250037p
#SBATCH --cpus-per-task=5
#SBATCH -t 03:00:00
#SBATCH -J si_dl2
#SBATCH -o logs/download_extra_%j.log
# archives 10-29: training-only data for the learned reference encoder (Idea 2 midterm)
cd /ocean/projects/cis250037p/ywang134/mmml-analysis1
PY=/ocean/projects/cis250037p/ywang134/envs/aw310/bin/python
for g in "10 11 12 13 14" "15 16 17 18 19" "20 21 22 23 24" "25 26 27 28 29"; do
  $PY scripts/download.py data 0 $g &
done
wait
echo DONE
