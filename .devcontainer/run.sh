#!/usr/bin/env bash
set -e
nohup tensorboard --logdir=/mnt/notebooks/logs --host=0.0.0.0 --port=6006 >/dev/null 2>&1 &
find /mnt/notebooks -name "*.ipynb" -exec jupyter trust {} \; || true
exec xvfb-run -s "-screen 0 1280x720x24" \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser \
  --notebook-dir=/mnt/notebooks
