#!/bin/bash
cd /workspace/harakaat-

PYTHONPATH=. python scripts/train.py --config configs/gpu_rtx4090.yaml --resume checkpoints/best.pt > logs/train.log 2>&1
EXIT_CODE=$?

echo "Training finished with exit code $EXIT_CODE at $(date)" >> logs/train.log

# Save checkpoint to HuggingFace Hub if HF_TOKEN is set
if [ -n "$HF_TOKEN" ] && [ -n "$HF_REPO" ]; then
    echo "Uploading checkpoint to HuggingFace..." >> logs/train.log
    pip install -q huggingface_hub
    python -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_file(
    path_or_fileobj='checkpoints/best.pt',
    path_in_repo='best.pt',
    repo_id=os.environ['HF_REPO'],
    repo_type='model',
)
api.upload_file(
    path_or_fileobj='logs/train.log',
    path_in_repo='train.log',
    repo_id=os.environ['HF_REPO'],
    repo_type='model',
)
print('Upload done.')
" >> logs/train.log 2>&1
fi

# Terminate pod completely (zero cost) — requires RUNPOD_API_KEY and POD_ID
if [ -n "$RUNPOD_API_KEY" ] && [ -n "$POD_ID" ]; then
    echo "Terminating pod $POD_ID..." >> logs/train.log
    curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"mutation { podTerminate(input: {podId: \\\"$POD_ID\\\"}) }\"}" \
        >> logs/train.log 2>&1
fi
