#!/usr/bin/env bash
set -e
nohup tensorboard --logdir=/workspaces/gdrl/notebooks/logs --host=0.0.0.0 --port=6006 >/dev/null 2>&1 &