#!/usr/bin/env python3
"""
Train the diacritization model.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --resume checkpoints/checkpoint_epoch_10.pt
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.utils.config import Config
from src.utils.vocab import CharVocab
from src.data_layer.dataset import DiacritizationDataset, collate_fn
from src.model.diacritizer import DiacritizationModel
from src.model.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train diacritization model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    config_dict = config.to_dict()

    # Seed (deterministic across CPU and CUDA)
    seed = config_dict.get("project", {}).get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    device_str = config_dict.get("project", {}).get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.1f} GB")

    # Load vocab
    dc = config_dict.get("data", {})
    vocab_path = Path(dc.get("lexicons_dir", "data/lexicons")) / "char_vocab.json"
    if not vocab_path.exists():
        print(f"ERROR: Vocabulary not found at {vocab_path}")
        print("Run: python scripts/prepare_data.py --config configs/default.yaml")
        return
    vocab = CharVocab.load(vocab_path)
    print(f"Vocabulary: {len(vocab)} chars")

    # Datasets
    splits_dir = Path(dc.get("splits_dir", "data/splits"))
    max_len = dc.get("max_sentence_length", 512)

    excluded_genres = dc.get("excluded_genres", [])
    if excluded_genres:
        print(f"Excluding genres from training: {excluded_genres}")

    mc = config_dict.get("model", {})
    enc = mc.get("encoder", {})
    emb = mc.get("char_embedding", {})
    weh = mc.get("word_ending_head", {})
    encoder_type = enc.get("type", "bilstm")

    # AraBERT tokenizer (only for arabert encoder)
    tokenizer = None
    if encoder_type == "arabert":
        from transformers import AutoTokenizer
        arabert_name = enc.get("model_name", "aubmindlab/bert-base-arabertv02")
        print(f"Loading AraBERT tokenizer: {arabert_name}")
        tokenizer = AutoTokenizer.from_pretrained(arabert_name)

    dataset_kwargs = dict(
        max_length=max_len,
        excluded_genres=excluded_genres,
        tokenizer=tokenizer,
        max_bert_length=enc.get("max_bert_length", 512),
    )
    train_dataset = DiacritizationDataset(splits_dir / "train" / "data.jsonl", vocab, **dataset_kwargs)
    valid_dataset = DiacritizationDataset(splits_dir / "valid" / "data.jsonl", vocab, **dataset_kwargs)
    mode_label = "AraBERT" if tokenizer else "BiLSTM"
    print(f"Train: {len(train_dataset)} | Valid: {len(valid_dataset)} ({mode_label} mode)")

    tc = config_dict.get("training", {})
    num_workers = tc.get("num_workers", 4)
    use_cuda = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=tc.get("batch_size", 64),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=tc.get("batch_size", 64) * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    model = DiacritizationModel(
        vocab_size=len(vocab),
        embed_dim=emb.get("dim", 128),
        embed_dropout=emb.get("dropout", 0.1),
        encoder_type=encoder_type,
        hidden_dim=enc.get("hidden_dim", 256),
        num_layers=enc.get("num_layers", 3),
        encoder_dropout=enc.get("dropout", 0.1),
        num_heads=enc.get("num_heads", 8),
        ff_dim=enc.get("ff_dim", 1024),
        use_crf=mc.get("use_crf", True),
        use_word_ending_head=weh.get("enable", True),
        we_hidden_dim=weh.get("hidden_dim", 128),
        we_context_window=weh.get("context_window", 5),
        we_loss_weight=weh.get("loss_weight", 0.3),
        arabert_model_name=enc.get("model_name", "aubmindlab/bert-base-arabertv02"),
        arabert_freeze_layers=enc.get("freeze_layers", 0),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    # Train (resume handled inside Trainer if --resume provided)
    trainer = Trainer(
        model, train_loader, valid_loader, config_dict, device,
        resume_path=Path(args.resume) if args.resume else None,
    )
    result = trainer.train()

    print(f"\nTraining complete. Best val loss: {result['best_val_loss']:.4f}")
    print(f"Checkpoints saved to: {tc.get('checkpoint_dir', 'checkpoints')}/")
    print(f"\nNext step: python scripts/evaluate.py --config {args.config} --checkpoint checkpoints/best.pt")


if __name__ == "__main__":
    main()
