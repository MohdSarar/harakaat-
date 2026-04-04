# Manual Steps — What You Need to Do

## 1. Data Downloads (OBLIGATOIRE)

Le pipeline est fonctionnel mais les **données ne sont pas incluses** — tu dois les télécharger toi-même.

### Tashkeela (corpus principal MSA)
- **URL**: https://sourceforge.net/projects/tashkeela/
- **Article**: https://pmc.ncbi.nlm.nih.gov/articles/PMC5310197/
- **Action**: Télécharger l'archive, extraire les fichiers `.txt` dans :
  ```
  data/raw/msa/tashkeela/
  ```

### Sadeed_Tashkeela (HuggingFace)
- **URL**: https://huggingface.co/datasets/Misraj/Sadeed_Tashkeela
- **Action**: Le script `download_data.py` le télécharge automatiquement via la lib `datasets`
  ```bash
  python scripts/download_data.py --dataset sadeed
  ```

### Quranic Arabic Corpus (morphologie + texte diacritisé)
- **URL**: https://corpus.quran.com/download/
- **Format**: Fichier TSV morphologique en Buckwalter transliteration
  - En-tête : `LOCATION	FORM	TAG	FEATURES`
  - Chaque ligne : `(sura:aya:word:segment)	form_buckwalter	TAG	FEATURES`
- **Action**: Télécharger le fichier morphologie `quranic-corpus-morphology-0.4.txt`, placer dans :
  ```
  data/raw/classical/quran/quranic-corpus-morphology-0.4.txt
  ```
- **Note**: Le loader convertit automatiquement le Buckwalter en arabe Unicode et reconstruit les versets complets.
- **Alternative**: Si tu as un fichier Tanzil en texte brut (un verset par ligne, déjà en arabe), change le format dans la config :
  ```yaml
  format: "quran_tanzil"  # au lieu de "quran_morphology"
  ```

### Deep Diacritization (benchmark)
- **URL**: https://github.com/BKHMSI/deep-diacritization
- **Action**: Cloner pour utiliser comme benchmark de référence
  ```bash
  git clone https://github.com/BKHMSI/deep-diacritization.git benchmarks/deep-diacritization
  ```

---

## 2. Installation des dépendances

```bash
# Installation de base
pip install -e ".[dev]"

# Avec SinaTools (optionnel mais recommandé)
pip install sinatools
# Docs: https://sina.birzeit.edu/sinatools/

# PyTorch GPU (si CUDA disponible)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Pipeline d'exécution complet

```bash
# Étape 1: Télécharger les données
python scripts/download_data.py --dataset all

# Étape 2: Préparer (normaliser, aligner, split, vocab, lexique)
python scripts/prepare_data.py --config configs/default.yaml

# Étape 3: Entraîner
python scripts/train.py --config configs/default.yaml

# Étape 4: Évaluer
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt

# Étape 5: Servir l'API
python scripts/serve.py --port 8000 --checkpoint checkpoints/best.pt
```

---

## 4. SinaTools (couche linguistique avancée)

Pour activer la couche linguistique complète (POS, morphologie, lemmatisation) :

1. Installer : `pip install sinatools`
2. Dans `configs/default.yaml`, mettre :
   ```yaml
   linguistic:
     use_sinatools: true
   ```
3. Documentation : https://sina.birzeit.edu/sinatools/

Sans SinaTools, le pipeline utilise des heuristiques rule-based (fonctionnel mais moins précis).

---

## 5. Structure des données attendue

Après téléchargement, ta structure devrait ressembler à :

```
data/
├── raw/
│   ├── msa/
│   │   ├── tashkeela/          ← fichiers .txt diacritisés
│   │   └── sadeed/             ← auto-téléchargé via HF
│   ├── classical/
│   │   └── quran/
│   │       └── quranic-corpus-morphology-0.4.txt  ← TSV Buckwalter
│   └── dialect/                ← (à remplir si tu as des données dialectales)
```

---

## 6. Entraînement GPU recommandé

Le modèle BiLSTM+CRF par défaut est raisonnable sur CPU pour le debug, mais pour un entraînement sérieux :

- **GPU**: NVIDIA avec ≥ 8 Go VRAM
- **Temps estimé**: ~2-4h sur RTX 3090 avec Tashkeela complet
- **Mixed precision**: activé par défaut (`fp16: true`)

Pour utiliser le Transformer au lieu du BiLSTM :
```yaml
model:
  encoder:
    type: "transformer"
    num_heads: 8
    ff_dim: 1024
```

---

## 7. HITL (Human-in-the-Loop)

Le système HITL est fonctionnel mais il n'y a pas encore d'interface web. 
Pour l'utiliser :

```python
from src.hitl.review import HITLManager

hitl = HITLManager("data/metadata/hitl.db")

# Voir les items en attente
pending = hitl.get_pending(limit=20)

# Approuver
hitl.approve(item_id=1)

# Corriger
hitl.correct(item_id=2, corrected_text="النَّصُّ الْمُصَحَّحُ")

# Exporter pour réentraînement
hitl.export_corrections("data/aligned/hitl_corrections.jsonl")
```

TODO futur : construire une interface Streamlit ou web pour la correction HITL.

---

## 8. Ajout de données dialectales

Pour ajouter du dialecte (palestinien, égyptien, etc.) :

1. Placer les fichiers dans `data/raw/dialect/`
2. Ajouter une source dans `configs/default.yaml` :
   ```yaml
   data:
     sources:
       - name: "palestinian"
         type: "local"
         path: "data/raw/dialect/palestinian"
         variety: "dialect"
         format: "txt"
   ```
3. Re-run `prepare_data.py`

---

## 9. Limites connues

| Limitation | Détail |
|---|---|
| **Pas de données incluses** | Tu dois télécharger Tashkeela, Quran, etc. manuellement |
| **SinaTools optionnel** | Sans SinaTools, la couche linguistique est basique |
| **Pas d'interface HITL** | La queue HITL est en SQLite, pas de GUI |
| **Seq2Seq non implémenté** | Seuls BiLSTM+CRF et Transformer+CRF sont implémentés |
| **Pas de modèle pré-entraîné** | Tu dois entraîner depuis zéro |
| **Dialecte limité** | Le pipeline est orienté MSA/classique ; le dialecte demande des données spécifiques |
| **Word-ending head** | Fonctionne mais peut nécessiter un tuning du `loss_weight` |
| **Évaluation case endings** | La séparation case/non-case dépend d'une heuristique de frontières de mots |
