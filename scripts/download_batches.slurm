#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --mem=22000M
#SBATCH -A cis250037p
#SBATCH --cpus-per-task=5
#SBATCH -t 01:30:00
#SBATCH -J si_dlb
#SBATCH -o logs/download_batches_%j.log
# batches 0001-0005, archives 0-9: new participants, training-only data for the learned encoder
cd /ocean/projects/cis250037p/ywang134/mmml-analysis1
PY=/ocean/projects/cis250037p/ywang134/envs/aw310/bin/python
for b in 1 2 3 4 5; do
  $PY scripts/download.py data $b 0 1 2 3 4 5 6 7 8 9 &
done
wait
echo DONE
