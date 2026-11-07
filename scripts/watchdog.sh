#!/bin/bash
# Watchdog: every 2 min, if archives 0-9 are not all done and no download is running,
# restart the download for the missing ones; if a finished stage has no results, rerun it.
cd /ocean/projects/cis250037p/ywang134/mmml-analysis1
PY=/ocean/projects/cis250037p/ywang134/envs/aw310/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
run_stage() {  # name glob figname
  log "running analysis $1"
  mkdir -p results/$1
  $PY scripts/analysis1.py data results/$1 "$2" >> logs/analysis_$1.log 2>&1 && \
  $PY scripts/make_figure.py results/$1/analysis1_results.json figures/$3 >> logs/analysis_$1.log 2>&1
  log "analysis $1 exit=$?"
}
fails=0
while true; do
  missing=""
  for a in 0 1 2 3 4 5 6 7 8 9; do
    [ -e data/0000_$(printf %04d $a)/.done ] || missing="$missing $a"
  done
  if [ -n "$missing" ] && ! pgrep -u $USER -f "scripts/download.py" >/dev/null; then
    log "no downloader alive, missing:$missing -> restarting"
    $PY scripts/download.py data 0 $missing >> logs/download_watchdog.log 2>&1 &
  fi
  p_ok=$(ls data/0000_000[0-2]/.done 2>/dev/null | wc -l)
  if [ "$p_ok" = 3 ] && [ ! -s results/primary_a0-2/analysis1_results.json ] \
     && ! pgrep -u $USER -f "analysis1.py data results/primary" >/dev/null && [ $fails -lt 3 ]; then
    sleep 120  # give auto_run.sh the first chance
    if [ ! -s results/primary_a0-2/analysis1_results.json ] && ! pgrep -u $USER -f "analysis1.py data results/primary" >/dev/null; then
      run_stage primary_a0-2 "0000_000[0-2]" fig1_primary; [ -s results/primary_a0-2/analysis1_results.json ] || fails=$((fails+1))
    fi
  fi
  if [ -z "$missing" ] && [ ! -s results/extended_a0-9/analysis1_results.json ] \
     && ! pgrep -u $USER -f "analysis1.py data results/extended" >/dev/null && [ $fails -lt 3 ]; then
    sleep 120
    if [ ! -s results/extended_a0-9/analysis1_results.json ] && ! pgrep -u $USER -f "analysis1.py data results/extended" >/dev/null; then
      run_stage extended_a0-9 "0000_000[0-9]" fig1_extended; [ -s results/extended_a0-9/analysis1_results.json ] || fails=$((fails+1))
    fi
  fi
  if [ -s results/extended_a0-9/analysis1_results.json ]; then log "all done"; exit 0; fi
  [ $fails -ge 3 ] && { log "analysis failed 3x, needs manual fix"; exit 1; }
  sleep 120
done
