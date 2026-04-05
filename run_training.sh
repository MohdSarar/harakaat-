#!/bin/bash
cd /workspace/harakaat-
PYTHONPATH=. python scripts/train.py --config configs/gpu_rtx4090.yaml --resume checkpoints/best.pt > logs/train.log 2>&1
