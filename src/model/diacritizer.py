"""
Main diacritization model — supports BiLSTM+CRF and Transformer+CRF architectures,
plus a specialized word-ending head (Layer 5).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional

from src.utils import NUM_DIAC_CLASSES
from src.model.crf import CRF


class CharEmbedding(nn.Module):
    """Character embedding with optional positional encoding."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        # Sinusoidal positional encoding
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        emb = emb + self.pe[:, :x.size(1)]
        return self.dropout(emb)


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.output_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            output, _ = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.lstm(x)
        return output


class TransformerEncoder(nn.Module):
    """Lightweight Transformer encoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3,
                 num_heads: int = 8, ff_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.proj(x)
        # mask: (batch, seq_len) bool where True = valid → invert for Transformer
        src_key_padding_mask = ~mask if mask is not None else None
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class WordEndingHead(nn.Module):
    """
    Specialized prediction head for word-final diacritics (Layer 5).
    
    Focuses on the last character(s) of each word, where grammatical
    case endings (إعراب) are most challenging.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = NUM_DIAC_CLASSES,
                 context_window: int = 5):
        super().__init__()
        self.context_window = context_window
        self.fc = nn.Sequential(
            nn.Linear(input_dim * context_window, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, hidden_states: torch.Tensor, word_end_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            word_end_mask: (batch, seq_len) bool — True at word-final positions
            
        Returns:
            logits: (batch, seq_len, num_classes) — only valid at word-end positions
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        ctx = self.context_window
        
        # Pad left with zeros so position i can look back ctx positions
        pad = hidden_states.new_zeros(batch_size, ctx, hidden_dim)
        padded = torch.cat([pad, hidden_states], dim=1)  # (batch, ctx + seq_len, hidden)
        
        # Use unfold to extract all windows in one vectorized op (GPU-fast)
        # (batch, ctx + seq_len, hidden) → transpose → unfold → reshape
        # unfold operates on the sequence dimension
        windows = padded.unfold(1, ctx, 1)  # (batch, seq_len, hidden, ctx)
        windows = windows.permute(0, 1, 3, 2)  # (batch, seq_len, ctx, hidden)
        windows = windows.reshape(batch_size, seq_len, ctx * hidden_dim)
        
        logits = self.fc(windows)  # (batch, seq_len, num_classes)
        return logits


class DiacritizationModel(nn.Module):
    """
    Complete diacritization model.
    
    Architecture:
        Input (char indices)
        → CharEmbedding
        → Encoder (BiLSTM or Transformer)
        → Main classification head → CRF
        → Word-ending specialized head (optional)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        embed_dropout: float = 0.1,
        encoder_type: str = "bilstm",  # "bilstm" or "transformer"
        hidden_dim: int = 256,
        num_layers: int = 3,
        encoder_dropout: float = 0.3,
        num_heads: int = 8,
        ff_dim: int = 1024,
        use_crf: bool = True,
        num_classes: int = NUM_DIAC_CLASSES,
        # Word ending head
        use_word_ending_head: bool = True,
        we_hidden_dim: int = 128,
        we_context_window: int = 5,
        we_loss_weight: float = 0.3,
    ):
        super().__init__()
        
        self.use_crf = use_crf
        self.use_word_ending_head = use_word_ending_head
        self.we_loss_weight = we_loss_weight
        
        # Character embedding
        self.char_embed = CharEmbedding(vocab_size, embed_dim, embed_dropout)
        
        # Encoder
        if encoder_type == "bilstm":
            self.encoder = BiLSTMEncoder(embed_dim, hidden_dim, num_layers, encoder_dropout)
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(embed_dim, hidden_dim, num_layers, num_heads, ff_dim, encoder_dropout)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        enc_output_dim = self.encoder.output_dim
        
        # Main classification head
        self.classifier = nn.Linear(enc_output_dim, num_classes)
        
        # CRF
        if use_crf:
            self.crf = CRF(num_classes)
        
        # Word-ending head
        if use_word_ending_head:
            self.word_ending_head = WordEndingHead(
                enc_output_dim, we_hidden_dim, num_classes, we_context_window
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        word_end_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Returns dict with:
            - loss (if labels provided)
            - emissions (logits)
            - predictions (decoded tags)
            - word_ending_logits (if head enabled)
        """
        # Embed
        embedded = self.char_embed(input_ids)
        
        # Encode
        if isinstance(self.encoder, BiLSTMEncoder):
            hidden = self.encoder(embedded, lengths)
        else:
            hidden = self.encoder(embedded, attention_mask)
        
        # Main emissions
        emissions = self.classifier(hidden)  # (batch, seq, num_classes)
        
        result = {"emissions": emissions}
        
        # Compute loss
        if labels is not None:
            if self.use_crf:
                # CRF loss
                # Replace -1 padding in labels with 0 for CRF
                crf_labels = labels.clone()
                crf_labels[crf_labels == -1] = 0
                main_loss = self.crf(emissions, crf_labels, attention_mask)
            else:
                main_loss = nn.functional.cross_entropy(
                    emissions.view(-1, emissions.size(-1)),
                    labels.view(-1),
                    ignore_index=-1,
                )
            
            total_loss = main_loss
            
            # Word-ending head loss
            if self.use_word_ending_head and word_end_mask is not None:
                we_logits = self.word_ending_head(hidden, word_end_mask)
                result["word_ending_logits"] = we_logits
                
                # Only compute loss at word-end positions
                we_mask = word_end_mask & attention_mask
                if we_mask.any():
                    we_loss = nn.functional.cross_entropy(
                        we_logits[we_mask],
                        labels[we_mask],
                        ignore_index=-1,
                    )
                    total_loss = total_loss + self.we_loss_weight * we_loss
                    result["word_ending_loss"] = we_loss
            
            result["loss"] = total_loss
        
        # Decode predictions
        if self.use_crf:
            result["predictions"] = self.crf.decode(emissions, attention_mask)
        else:
            result["predictions"] = emissions.argmax(dim=-1).tolist()
        
        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> dict:
        """Inference-only forward pass."""
        self.eval()
        with torch.no_grad():
            return self.forward(input_ids, attention_mask, lengths=lengths)
