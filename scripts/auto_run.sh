#!/bin/bash
# Wait for archives 0-2, then run Analysis 1 + figure automatically.
cd /ocean/projects/cis250037p/ywang134/mmml-analysis1
PY=/ocean/projects/cis250037p/ywang134/envs/aw310/bin/python
until [ -e data/0000_0000/.done ] && [ -e data/0000_0001/.done ] && [ -e data/0000_0002/.done ]; do sleep 30; done
mkdir -p results/primary_a0-2 figures
ln -sfn ../data/0000_0000 results/primary_a0-2/in0 2>/dev/null
$PY scripts/analysis1.py data results/primary_a0-2 "0000_000[0-2]" && \
$PY scripts/make_figure.py results/primary_a0-2/analysis1_results.json figures/fig1_primary
echo "AUTO_RUN_FINISHED exit=$?"
