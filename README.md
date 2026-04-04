# Arabic Diacritization Pipeline — Complete Architecture

A production-grade, modular Arabic diacritization system inspired by:
- **SinaLab / Birzeit University** — linguistic layer, NLP tools
- **HBKU / Doha** — sequence-to-sequence diacritization, multi-genre corpus strategies

## Architecture Overview

```
Corpus diacritisé brut
→ Normalisation Unicode
→ Génération version non-diacritisée
→ Alignement phrase par phrase
→ Enrichissement linguistique (tokenisation, morphologie, POS, lemmas)
→ Split train / valid / test
→ Entraînement modèle principal char-level
→ Entraînement tête spécialisée word-ending
→ Décodage hybride (modèle + règles + lexique + vote)
→ Évaluation DER / WER
→ Correction humaine (HITL)
→ Réentraînement
→ Déploiement API
```

## Project Structure

```
arabic-diacritization-pipeline/
├── configs/                  # YAML configurations
├── data/                     # Data directory (raw, normalized, aligned, splits)
├── docs/                     # Documentation
├── notebooks/                # Jupyter notebooks for exploration
├── scripts/                  # CLI scripts (download, train, evaluate, serve)
├── src/
│   ├── data_layer/           # Layer 1: Data loading, corpus management
│   ├── normalization/        # Layer 2: Unicode normalization, text cleaning
│   ├── linguistic/           # Layer 3: Morphology, POS, lemmatization (SinaTools)
│   ├── model/                # Layer 4-5: Char-level model + word-ending head
│   ├── decoding/             # Layer 6: Hybrid decoding (neural + rules + lexicon)
│   ├── evaluation/           # Layer 8: DER/WER metrics, benchmarking
│   ├── api/                  # Layer 9: FastAPI deployment
│   ├── hitl/                 # Layer 7: Human-in-the-loop correction
│   └── utils/                # Shared utilities
├── tests/                    # Unit & integration tests
├── pyproject.toml
├── Makefile
└── Dockerfile
```

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Download data
python scripts/download_data.py --dataset tashkeela

# 3. Prepare data
python scripts/prepare_data.py --config configs/default.yaml

# 4. Train
python scripts/train.py --config configs/default.yaml

# 5. Evaluate
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt

# 6. Serve API
python scripts/serve.py --port 8000
```

## Manual Steps Required

See `docs/MANUAL_STEPS.md` for what you need to download and configure manually.

## License

MIT
