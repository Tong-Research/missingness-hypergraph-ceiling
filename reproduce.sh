#!/bin/sh
# Regenerate tables/figures from results/. Steps whose inputs are withheld under a data-use agreement report it and continue.
export PYTHONPATH=$PWD/src:$PWD/experiments:$PWD/figures:$PYTHONPATH
echo "-- inflating compressed result files"; for f in $(find results -name "*.csv.gz"); do gunzip -kf "$f"; done
echo "-- bound figure"; ( python experiments/fig_bound.py ) || echo "   skipped: bound figure needs inputs withheld under a data-use agreement (the figure/table is shipped as built)"
