"""
FastAPI application for Arabic diacritization.

Endpoints:
    POST /full_diacritize   — Complete diacritization
    POST /partial_diacritize — Preserve existing diacritics
    POST /suggest            — High-confidence suggestions only
    GET  /health             — Health check
"""

from __future__ import annotations

import torch
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils import strip_diacritics, has_diacritics, HARAKAT_PATTERN
from src.utils.vocab import CharVocab
from src.utils.config import Config
from src.model.diacritizer import DiacritizationModel
from src.decoding.hybrid_decoder import HybridDecoder, DecodingResult
from src.linguistic.lexicon import FrequencyLexicon


# ---- Pydantic schemas ----

class DiacritizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Arabic text to diacritize")


class DiacritizeResponse(BaseModel):
    text_diac: str
    confidence: float
    corrections_applied: int
    flagged_for_review: bool


class SuggestResponse(BaseModel):
    original: str
    suggestions: list[dict]  # [{"word": str, "diacritized": str, "confidence": float}]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# ---- App state ----

class AppState:
    """Holds loaded model and decoder."""
    model: Optional[DiacritizationModel] = None
    vocab: Optional[CharVocab] = None
    decoder: Optional[HybridDecoder] = None
    device: torch.device = torch.device("cpu")
    max_length: int = 512


state = AppState()


def create_app(
    config_path: str = "configs/default.yaml",
    checkpoint_path: Optional[str] = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Arabic Diacritization API",
        description="Production-grade Arabic text diacritization service",
        version="0.1.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def load_model():
        """Load model on startup."""
        config = Config.from_yaml(config_path)
        
        # Device
        device_str = config.get("device", "auto")
        if device_str == "auto":
            state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            state.device = torch.device(device_str)
        
        state.max_length = config.data.get("max_sentence_length", 512)
        
        # Load vocab
        vocab_path = Path("data/lexicons/char_vocab.json")
        if vocab_path.exists():
            state.vocab = CharVocab.load(vocab_path)
        else:
            print("WARNING: Vocab not found. Model predictions unavailable.")
            return
        
        # Load model
        if checkpoint_path and Path(checkpoint_path).exists():
            mc = config.model.to_dict()
            state.model = DiacritizationModel(
                vocab_size=len(state.vocab),
                embed_dim=mc.get("char_embedding", {}).get("dim", 128),
                encoder_type=mc.get("encoder", {}).get("type", "bilstm"),
                hidden_dim=mc.get("encoder", {}).get("hidden_dim", 256),
                num_layers=mc.get("encoder", {}).get("num_layers", 3),
                use_crf=mc.get("use_crf", True),
                use_word_ending_head=mc.get("word_ending_head", {}).get("enable", True),
            )
            checkpoint = torch.load(checkpoint_path, map_location=state.device)
            state.model.load_state_dict(checkpoint["model_state_dict"])
            state.model.to(state.device)
            state.model.eval()
            print(f"Model loaded from {checkpoint_path} on {state.device}")
        else:
            print("WARNING: No checkpoint found. Model predictions unavailable.")
        
        # Load lexicon
        lexicon_path = Path(config.decoding.get("lexicon_path", "data/lexicons/frequency_lexicon.json"))
        lexicon = None
        if lexicon_path.exists():
            lexicon = FrequencyLexicon.load(lexicon_path)
            print(f"Lexicon loaded: {len(lexicon)} entries")
        
        # Build decoder
        dc = config.decoding.to_dict()
        state.decoder = HybridDecoder(
            lexicon=lexicon,
            use_morphological_constraints=dc.get("use_morphological_constraints", True),
            use_lexicon=dc.get("use_lexicon", True),
            use_reranking=dc.get("use_reranking", True),
            confidence_threshold=dc.get("confidence_threshold", 0.85),
        )

    def _run_inference(text: str) -> DecodingResult:
        """Run model inference + hybrid decoding on a text."""
        if state.model is None or state.vocab is None or state.decoder is None:
            raise HTTPException(503, "Model not loaded")
        
        text_undiac = strip_diacritics(text)
        
        # Encode
        input_ids = torch.tensor(
            [state.vocab.encode(text_undiac[:state.max_length])],
            dtype=torch.long,
            device=state.device,
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        lengths = torch.tensor([input_ids.size(1)], device=state.device)
        
        # Predict
        output = state.model.predict(input_ids, attention_mask, lengths)
        
        pred_labels = output["predictions"][0]
        emissions = output["emissions"][0]  # (seq_len, num_classes)
        we_logits = output.get("word_ending_logits")
        if we_logits is not None:
            we_logits = we_logits[0]
        
        # Hybrid decode
        result = state.decoder.decode(
            text_undiac, pred_labels, emissions, we_logits
        )
        return result

    @app.post("/full_diacritize", response_model=DiacritizeResponse)
    async def full_diacritize(request: DiacritizeRequest):
        """Full diacritization — ignores any existing diacritics."""
        result = _run_inference(request.text)
        return DiacritizeResponse(
            text_diac=result.text_diac,
            confidence=round(result.confidence, 4),
            corrections_applied=result.corrections_applied,
            flagged_for_review=result.flagged_for_review,
        )

    @app.post("/partial_diacritize", response_model=DiacritizeResponse)
    async def partial_diacritize(request: DiacritizeRequest):
        """
        Partial diacritization — preserves existing diacritics in the input
        and only adds missing ones.
        """
        text = request.text
        text_undiac = strip_diacritics(text)
        
        # Run full diacritization
        result = _run_inference(text)
        
        # Merge: keep existing diacritics where present
        if has_diacritics(text):
            from src.utils import extract_diacritics
            original_diacs = extract_diacritics(text)
            predicted_diacs = extract_diacritics(result.text_diac)
            
            merged = []
            chars = list(text_undiac)
            for i, ch in enumerate(chars):
                merged.append(ch)
                if i < len(original_diacs) and original_diacs[i]:
                    merged.append(original_diacs[i])  # keep original
                elif i < len(predicted_diacs) and predicted_diacs[i]:
                    merged.append(predicted_diacs[i])  # add predicted
            
            result.text_diac = "".join(merged)
        
        return DiacritizeResponse(
            text_diac=result.text_diac,
            confidence=round(result.confidence, 4),
            corrections_applied=result.corrections_applied,
            flagged_for_review=result.flagged_for_review,
        )

    @app.post("/suggest", response_model=SuggestResponse)
    async def suggest(request: DiacritizeRequest):
        """
        Suggestion mode — only returns diacritizations for words
        where the model is highly confident.
        """
        result = _run_inference(request.text)
        text_undiac = strip_diacritics(request.text)
        words_undiac = text_undiac.split()
        words_diac = result.text_diac.split()
        
        suggestions = []
        for i, (wu, wd) in enumerate(zip(words_undiac, words_diac)):
            conf = result.per_word_confidence[i] if i < len(result.per_word_confidence) else 0.0
            if conf >= 0.85:  # high confidence only
                suggestions.append({
                    "word": wu,
                    "diacritized": wd,
                    "confidence": round(conf, 4),
                    "position": i,
                })
        
        return SuggestResponse(
            original=request.text,
            suggestions=suggestions,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            model_loaded=state.model is not None,
            device=str(state.device),
        )

    return app
