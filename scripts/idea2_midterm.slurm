#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --mem=22000M
#SBATCH -A cis250037p
#SBATCH --cpus-per-task=5
#SBATCH -t 04:00:00
#SBATCH -J idea2mid
#SBATCH -o logs/idea2_midterm_%j.log
# usage: sbatch scripts/idea2_midterm.slurm ARCHIVE_GLOB OUT_DIR STEPS
cd /ocean/projects/cis250037p/ywang134/mmml-analysis1
PY=/ocean/projects/cis250037p/ywang134/envs/aw310/bin/python
export OMP_NUM_THREADS=5
$PY scripts/idea2_features.py data cache "$1" 5
$PY scripts/idea2_midterm.py cache data "$2" "$3"
echo "DONE exit=$?"
