#!/usr/bin/env bash
# Reproduce everything in results/ (about 2-3 minutes).
#   bash run_all.sh            or      PYTHON=~/anaconda3/bin/python bash run_all.sh
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY verify.py
$PY run.py examples/l_beam.yaml --quiet
$PY run.py examples/biclamped_beam.yaml --quiet
$PY scripts/compare_with_paper.py
for f in bracket_with_hole roof_polygon hook_from_image; do
  $PY run.py examples/$f.yaml --quiet
done
echo "All done - see results/"
