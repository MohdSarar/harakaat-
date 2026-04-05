#!/bin/bash
cd /workspace/harakaat-

mkdir -p logs checkpoints

# Install dependencies
echo "Installing dependencies at $(date)..." | tee logs/train.log
pip install -e "." -q >> logs/train.log 2>&1
echo "Dependencies installed." | tee -a logs/train.log

# Download checkpoint from HuggingFace if local one missing
if [ ! -f checkpoints/best.pt ] && [ -n "$HF_TOKEN" ] && [ -n "$HF_REPO" ]; then
    echo "Downloading checkpoint from HuggingFace..." | tee -a logs/train.log
    python -c "
from huggingface_hub import hf_hub_download
import os, shutil
path = hf_hub_download(repo_id=os.environ['HF_REPO'], filename='best.pt', token=os.environ['HF_TOKEN'])
shutil.copy(path, 'checkpoints/best.pt')
print('Checkpoint downloaded.')
" 2>&1 | tee -a logs/train.log
fi

PYTHONPATH=. python scripts/train.py --config configs/arabert.yaml --resume checkpoints/best.pt >> logs/train.log 2>&1
EXIT_CODE=$?

echo "Training finished with exit code $EXIT_CODE at $(date)" | tee -a logs/train.log

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

# Only upload and terminate if training succeeded
if [ "$EXIT_CODE" -eq 0 ] && [ -n "$RUNPOD_API_KEY" ]; then
    # Get real pod ID from hostname if POD_ID not set or wrong
    REAL_POD_ID="${POD_ID:-$(hostname)}"
    echo "Terminating pod $REAL_POD_ID..." | tee -a logs/train.log
    curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"mutation { podTerminate(input: {podId: \\\"$REAL_POD_ID\\\"}) }\"}" \
        >> logs/train.log 2>&1
else
    echo "Training failed (exit $EXIT_CODE) — pod NOT terminated. Check logs/train.log" | tee -a logs/train.log
fi
