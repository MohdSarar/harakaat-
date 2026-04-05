"""
Training loop for the diacritization model.

GPU-optimized:
- torch.amp (new API, PyTorch >= 2.1)
- CRF protected in float32 under autocast
- CUDA benchmark mode
- non_blocking transfers
- zero_grad(set_to_none=True)
- GPU memory tracking
- Scaler state in checkpoints
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.diacritizer import DiacritizationModel
from src.evaluation.metrics import compute_der, compute_wer


class Trainer:

    def __init__(
        self,
        model: DiacritizationModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: dict,
        device: torch.device,
        resume_path: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config
        self.device = device

        tc = config.get("training", {})
        self.epochs = tc.get("epochs", 50)
        self.lr = tc.get("learning_rate", 1e-3)
        self.weight_decay = tc.get("weight_decay", 1e-4)
        self.gradient_clip = tc.get("gradient_clip", 5.0)
        self.patience = tc.get("early_stopping_patience", 7)
        self.use_fp16 = tc.get("fp16", True) and device.type == "cuda"
        self.checkpoint_dir = Path(tc.get("checkpoint_dir", "checkpoints"))
        self.log_dir = Path(tc.get("log_dir", "logs"))

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ---- GPU diagnostics ----
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(device)
            gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            print(f"CUDA: {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}")
            print(f"Mixed precision (fp16): {'ON' if self.use_fp16 else 'OFF'}")
            torch.backends.cudnn.benchmark = True

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Scheduler
        sched_type = tc.get("scheduler", "cosine")
        warmup = tc.get("warmup_steps", 0)
        total_steps = self.epochs * len(train_loader)

        if sched_type == "cosine":
            cosine_steps = max(total_steps - warmup, 1)
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cosine_steps
            )
            if warmup > 0:
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_sched, cosine_sched],
                    milestones=[warmup],
                )
                print(f"Scheduler: linear warmup ({warmup} steps) → cosine decay ({cosine_steps} steps)")
            else:
                self.scheduler = cosine_sched
                print(f"Scheduler: cosine decay ({cosine_steps} steps)")
        elif sched_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )

        # ---- Mixed precision scaler (new API) ----
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_fp16)

        # Tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: list[dict] = []
        self.start_epoch = 1

        # ---- Resume from checkpoint ----
        if resume_path is not None:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best_val_loss = ckpt.get("val_loss", float("inf"))
            # Fast-forward scheduler to match completed epochs
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                steps_done = (self.start_epoch - 1) * len(train_loader)
                for _ in range(steps_done):
                    self.scheduler.step()
            print(f"Resumed from epoch {self.start_epoch - 1} (best val_loss={self.best_val_loss:.4f})")

    def train(self) -> dict:
        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()

            train_loss = self._train_epoch(epoch)
            val_loss, val_der, val_wer = self._validate(epoch)

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            mem_info = ""
            if self.device.type == "cuda":
                mem_alloc = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                mem_reserved = torch.cuda.max_memory_reserved(self.device) / (1024**3)
                mem_info = f" | GPU: {mem_alloc:.1f}/{mem_reserved:.1f}GB"
                torch.cuda.reset_peak_memory_stats(self.device)

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_der": round(val_der, 6),
                "val_wer": round(val_wer, 6),
                "lr": lr,
                "time_s": round(elapsed, 1),
            }
            self.history.append(record)

            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"DER: {val_der:.4f} | WER: {val_wer:.4f} | "
                f"LR: {lr:.6f} | {elapsed:.1f}s{mem_info}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss, val_der, val_wer, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

        with open(self.log_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return {"history": self.history, "best_val_loss": self.best_val_loss}

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)
        for batch in pbar:
            batch = {
                k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_fp16):
                output = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    word_end_mask=batch.get("word_end_mask"),
                    lengths=batch.get("lengths"),
                )
                loss = output["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, epoch: int) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        n_batches = 0

        for batch in tqdm(self.valid_loader, desc=f"Valid {epoch}", leave=False):
            batch = {
                k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            with torch.amp.autocast("cuda", enabled=self.use_fp16):
                output = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    word_end_mask=batch.get("word_end_mask"),
                    lengths=batch.get("lengths"),
                )

            total_loss += output["loss"].item()
            n_batches += 1

            preds = output["predictions"]
            labels = batch["labels"]
            mask = batch["attention_mask"]

            for i in range(len(preds)):
                seq_len = mask[i].sum().item()
                pred_seq = (
                    preds[i][:int(seq_len)]
                    if isinstance(preds[i], list)
                    else preds[i][:int(seq_len)].tolist()
                )
                label_seq = labels[i][:int(seq_len)].cpu().tolist()
                all_preds.append(pred_seq)
                all_labels.append(label_seq)

        avg_loss = total_loss / max(n_batches, 1)
        der = compute_der(all_preds, all_labels)
        wer = compute_wer(all_preds, all_labels)
        return avg_loss, der, wer

    def _save_checkpoint(
        self, epoch: int, val_loss: float,
        val_der: float = 0.0, val_wer: float = 0.0,
        is_best: bool = False,
    ):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
            "val_der": val_der,
            "val_wer": val_wer,
            "config": self.config,
        }

        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(state, path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            print(f"  → Best (loss={val_loss:.4f} DER={val_der:.4f} WER={val_wer:.4f})")

    @staticmethod
    def load_checkpoint(
        path: str | Path,
        model: DiacritizationModel,
        device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> dict:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        return checkpoint
