#!/bin/bash
cd /workspace/harakaat-

PYTHONPATH=. python scripts/train.py --config configs/gpu_rtx4090.yaml --resume checkpoints/best.pt > logs/train.log 2>&1
EXIT_CODE=$?

echo "Training finished with exit code $EXIT_CODE" >> logs/train.log

# Auto-stop the pod after training (requires RUNPOD_API_KEY and POD_ID env vars)
if [ -n "$RUNPOD_API_KEY" ] && [ -n "$POD_ID" ]; then
    echo "Stopping pod $POD_ID..." >> logs/train.log
    curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id } }\"}" \
        >> logs/train.log 2>&1
fi
