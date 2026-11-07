#!/bin/bash
# After the primary download (archives 0-2) finishes, fetch archives 3-9 and rerun on 0-9.
cd /ocean/projects/cis250037p/ywang134/mmml-analysis1
PY=/ocean/projects/cis250037p/ywang134/envs/aw310/bin/python
until [ -e data/0000_0002/.done ]; do sleep 30; done
$PY scripts/download.py data 0 3 4 5 6 7 8 9
mkdir -p results/extended_a0-9
$PY scripts/analysis1.py data results/extended_a0-9 "0000_000[0-9]" && \
$PY scripts/make_figure.py results/extended_a0-9/analysis1_results.json figures/fig1_extended
echo "AUTO_EXT_FINISHED exit=$?"
