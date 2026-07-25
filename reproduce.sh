#!/usr/bin/env bash
# Reproduce the three self-contained studies (ASlib needs a one-time data clone; see README).
set -e
here="$(cd "$(dirname "$0")" && pwd)"
echo "==== [1/3] 34-configuration diagnostic sweep ===="
( cd "$here/diagnostics" && python3 diagnostic_scale.py )
echo "==== [2/3] calibration / non-inferiority / tail risk ===="
( cd "$here/diagnostics" && python3 -W ignore calib_analysis.py )
echo "==== [3/3] controlled simulation ===="
( cd "$here/simulation" && python3 -W ignore controlled_sim.py )
echo "==== done. For ASlib external validation see README (one-time data clone). ===="
