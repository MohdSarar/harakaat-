"""
Conditional Random Field (CRF) layer for structured prediction.
Ensures valid diacritic label transitions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from src.utils import NUM_DIAC_CLASSES


class CRF(nn.Module):
    """
    Linear-chain CRF for diacritic sequence labeling.
    
    Implements:
    - Forward algorithm (partition function)
    - Viterbi decoding
    - Negative log-likelihood loss
    """

    def __init__(self, num_tags: int = NUM_DIAC_CLASSES):
        super().__init__()
        self.num_tags = num_tags
        
        # Transition matrix: transitions[i][j] = score of j → i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        self._init_constraints()

    def _init_constraints(self):
        """Initialize transition constraints (optional hard constraints)."""
        # No hard constraints by default — learned from data
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood.
        
        NOTE: CRF computations (logsumexp, forward algorithm) are numerically
        unstable in fp16. We force float32 here even under autocast.
        """
        # Force float32 for CRF numerical stability under AMP
        emissions = emissions.float()
        
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_algorithm(emissions, mask)
        nll = (forward_score - gold_score)
        return nll.mean()

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[list[int]]:
        """Viterbi decoding. Forces float32 for numerical stability."""
        emissions = emissions.float()
        batch_size, seq_len, _ = emissions.shape
        
        # Initialize
        score = self.start_transitions + emissions[:, 0]  # (batch, num_tags)
        history = []
        
        for t in range(1, seq_len):
            # (batch, num_tags, 1) + (num_tags, num_tags) + (batch, 1, num_tags)
            broadcast_score = score.unsqueeze(2)  # (batch, num_tags, 1)
            broadcast_emission = emissions[:, t].unsqueeze(1)  # (batch, 1, num_tags)
            
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)  # (batch, num_tags)
            
            # Apply mask
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
            history.append(indices)
        
        # End transition
        score += self.end_transitions
        
        # Trace back
        best_tags_list = []
        _, best_last_tag = score.max(dim=1)  # (batch,)
        
        for i in range(batch_size):
            best_tags = [best_last_tag[i].item()]
            seq_length = mask[i].sum().int().item()
            
            for hist in reversed(history[:seq_length - 1]):
                best_last_tag_i = hist[i][best_tags[-1]]
                best_tags.append(best_last_tag_i.item())
            
            best_tags.reverse()
            best_tags_list.append(best_tags)
        
        return best_tags_list

    def _forward_algorithm(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log partition function via forward algorithm."""
        batch_size, seq_len, num_tags = emissions.shape
        
        # (batch, num_tags)
        score = self.start_transitions + emissions[:, 0]
        
        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)  # (batch, num_tags, 1)
            broadcast_emission = emissions[:, t].unsqueeze(1)  # (batch, 1, num_tags)
            
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score = torch.logsumexp(next_score, dim=1)  # (batch, num_tags)
            
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
        
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)  # (batch,)

    def _score_sentence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score a gold tag sequence."""
        batch_size, seq_len, _ = emissions.shape
        
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        
        for t in range(1, seq_len):
            m = mask[:, t].float()
            emit_score = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            trans_score = self.transitions[tags[:, t], tags[:, t - 1]]
            score += (emit_score + trans_score) * m
        
        # End transition
        last_tag_indices = mask.sum(dim=1).long() - 1
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        
        return score
