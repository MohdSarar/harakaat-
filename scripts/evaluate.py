#!/usr/bin/env python3
"""
Evaluate a trained diacritization model.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import Config
from src.utils.vocab import CharVocab
from src.utils import strip_diacritics, DIAC_LABELS
from src.data_layer.dataset import DiacritizationDataset, collate_fn
from src.model.diacritizer import DiacritizationModel
from src.decoding.hybrid_decoder import HybridDecoder
from src.linguistic.lexicon import FrequencyLexicon
from src.evaluation.metrics import run_full_evaluation, EvaluationReport


def main():
    parser = argparse.ArgumentParser(description="Evaluate diacritization model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="evaluation_report.json")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--no-reranking", action="store_true",
                        help="Disable WordEndingHead reranking (use for checkpoints trained before the head fix)")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    config_dict = config.to_dict()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load vocab
    dc = config_dict.get("data", {})
    vocab = CharVocab.load(Path(dc.get("lexicons_dir", "data/lexicons")) / "char_vocab.json")

    # Load test data
    test_path = Path(dc.get("splits_dir", "data/splits")) / args.split / "data.jsonl"
    test_dataset = DiacritizationDataset(test_path, vocab)
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=4
    )
    print(f"Test samples: {len(test_dataset)}")

    # Load checkpoint first to infer vocab size actually used during training
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_vocab_size = checkpoint["model_state_dict"]["char_embed.embedding.weight"].shape[0]
    if ckpt_vocab_size != len(vocab):
        print(f"Warning: checkpoint vocab size ({ckpt_vocab_size}) differs from current vocab ({len(vocab)}). Using checkpoint size.")

    mc = config_dict.get("model", {})
    enc = mc.get("encoder", {})
    emb = mc.get("char_embedding", {})
    weh = mc.get("word_ending_head", {})

    model = DiacritizationModel(
        vocab_size=ckpt_vocab_size,
        embed_dim=emb.get("dim", 128),
        encoder_type=enc.get("type", "bilstm"),
        hidden_dim=enc.get("hidden_dim", 256),
        num_layers=enc.get("num_layers", 3),
        use_crf=mc.get("use_crf", True),
        use_word_ending_head=weh.get("enable", True),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded from {args.checkpoint}")

    # Load lexicon + decoder
    lex_path = Path(dc.get("lexicons_dir", "data/lexicons")) / "frequency_lexicon.json"
    lexicon = FrequencyLexicon.load(lex_path) if lex_path.exists() else None
    dconf = config_dict.get("decoding", {})
    use_reranking = dconf.get("use_reranking", True) and not args.no_reranking
    decoder = HybridDecoder(
        lexicon=lexicon,
        use_morphological_constraints=dconf.get("use_morphological_constraints", True),
        use_lexicon=dconf.get("use_lexicon", True),
        use_reranking=use_reranking,
        word_ending_override_threshold=dconf.get("word_ending_override_threshold", 0.40),
    )
    if args.no_reranking:
        print("WordEndingHead reranking disabled (--no-reranking)")

    # Load raw test records for metadata (genre, variety)
    raw_records = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))

    # Run inference
    predictions = []
    references = []
    idx = 0

    unk_idx = vocab.unk_idx

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            # Clamp indices that exceed training vocab size to UNK
            if "input_ids" in batch:
                ids = batch["input_ids"]
                out_of_range = ids >= ckpt_vocab_size
                if out_of_range.any():
                    ids = ids.clone()
                    ids[out_of_range] = min(unk_idx, ckpt_vocab_size - 1)
                    batch["input_ids"] = ids
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                word_end_mask=batch.get("word_end_mask"),
                lengths=batch.get("lengths"),
            )
            
            for i in range(batch["input_ids"].size(0)):
                if idx >= len(raw_records):
                    break
                
                rec = raw_records[idx]
                text_undiac = rec["text_undiac"]
                seq_len = batch["attention_mask"][i].sum().item()
                pred_labels = output["predictions"][i][:int(seq_len)]
                
                emissions = output["emissions"][i][:int(seq_len)]
                we_logits = output.get("word_ending_logits")
                if we_logits is not None:
                    we_logits = we_logits[i][:int(seq_len)]
                
                result = decoder.decode(text_undiac, pred_labels, emissions, we_logits)
                
                predictions.append({
                    "text_diac": result.text_diac,
                    "text_undiac": text_undiac,
                    "genre": rec.get("genre", "unknown"),
                    "variety": rec.get("variety", "unknown"),
                })
                references.append({
                    "text_diac": rec["text_diac"],
                    "text_undiac": text_undiac,
                    "genre": rec.get("genre", "unknown"),
                    "variety": rec.get("variety", "unknown"),
                })
                idx += 1

    # Run evaluation
    vocab_words = set()
    train_path = Path(dc.get("splits_dir", "data/splits")) / "train" / "data.jsonl"
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                vocab_words.update(rec["text_undiac"].split())

    report = run_full_evaluation(predictions, references, vocab_words)
    
    # Print and save
    print(report)
    report.save(args.output)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
