#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --mem=22000M
#SBATCH -A cis250037p
#SBATCH --cpus-per-task=5
#SBATCH -t 00:45:00
#SBATCH -J idea2
#SBATCH -o logs/analysis2_%j.log
cd /ocean/projects/cis250037p/ywang134/mmml-analysis1
PY=/ocean/projects/cis250037p/ywang134/envs/aw310/bin/python
export OMP_NUM_THREADS=5
$PY scripts/analysis2.py data results/idea2_a0-9 "0000_000[0-9]"
echo "DONE exit=$?"
