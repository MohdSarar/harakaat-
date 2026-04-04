#!/usr/bin/env python3
"""
Serve the diacritization API.

Usage:
    python scripts/serve.py --port 8000 --checkpoint checkpoints/best.pt
"""

import argparse
import uvicorn
from src.api.app import create_app


def main():
    parser = argparse.ArgumentParser(description="Serve diacritization API")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    app = create_app(config_path=args.config, checkpoint_path=args.checkpoint)
    
    print(f"Starting Arabic Diacritization API on {args.host}:{args.port}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"\nEndpoints:")
    print(f"  POST /full_diacritize")
    print(f"  POST /partial_diacritize")
    print(f"  POST /suggest")
    print(f"  GET  /health")
    print(f"\nDocs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
