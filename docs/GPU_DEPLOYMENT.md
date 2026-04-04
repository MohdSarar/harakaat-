# Guide de Déploiement GPU — RTX 4090 (RunPod / Vast.ai / Lambda)

## Résumé des modifications GPU

### Ce qui a été modifié (accélération UNIQUEMENT)
| Composant | Modification | Raison |
|---|---|---|
| `torch.amp` API | Migration `torch.cuda.amp` → `torch.amp` | Nouvelle API PyTorch ≥ 2.1 |
| CRF `float32` | Forward + Viterbi forcés en `float32` sous autocast | Stabilité numérique (logsumexp explose en fp16) |
| WordEndingHead | Python loop → `torch.unfold` vectorisé | 10-50x plus rapide sur GPU |
| DataLoader | `non_blocking=True`, `persistent_workers`, `prefetch_factor` | Alimenter le GPU sans attente |
| `zero_grad` | `set_to_none=True` | Micro-optimisation mémoire |
| `cudnn.benchmark` | Activé | Optimise les kernels CUDA pour tailles fixes |
| Checkpoints | Sauvegarde du `scaler_state_dict` | Reprise propre du fp16 |
| CUDA seed | `torch.cuda.manual_seed_all(seed)` | Reproductibilité GPU |

### Ce qui est STRICTEMENT identique
| Composant | Statut |
|---|---|
| `use_crf: true` | ✓ Préservé |
| `word_ending_head: true` | ✓ Préservé |
| Architecture BiLSTM 3 couches | ✓ Préservée |
| Dimensions (embed=128, hidden=256) | ✓ Préservées |
| Pipeline de données | ✓ Identique |
| Normalisation | ✓ Identique |
| Décodage hybride | ✓ Identique |
| Métriques d'évaluation | ✓ Identiques |

---

## Estimation temps d'entraînement

| Config | CPU (i7/Ryzen) | RTX 3090 | RTX 4090 |
|---|---|---|---|
| batch_size=64, BiLSTM+CRF | ~10-15h/epoch | ~45-60min/epoch | ~20-30min/epoch |
| batch_size=128, BiLSTM+CRF | impossible | ~30-45min/epoch | ~15-25min/epoch |
| 50 epochs complets | ~25-30 jours | ~25-50h | ~12-25h |
| Avec early stopping (~15-20 epochs) | ~7-12 jours | ~10-20h | ~5-10h |

---

## Déploiement sur RunPod

### 1. Créer l'instance

```
Plateforme : runpod.io
Template   : RunPod PyTorch 2.2+ (CUDA 12.1)
GPU        : RTX 4090 (24 GB)
Disk       : 50 GB minimum
```

**Prix indicatif** : ~$0.44/h pour RTX 4090 en Community Cloud

### 2. Se connecter

```bash
# Via SSH (fourni par RunPod)
ssh root@<IP_RUNPOD> -p <PORT> -i <KEY>

# Ou via le terminal web dans l'interface RunPod
```

### 3. Upload du projet

**Option A — Git (recommandé)**
```bash
cd /workspace
git clone https://github.com/<ton-repo>/arabic-diacritization-pipeline.git
cd arabic-diacritization-pipeline
```

**Option B — Upload zip**
```bash
cd /workspace
# Depuis ta machine locale :
scp -P <PORT> arabic-diacritization-pipeline.zip root@<IP>:/workspace/
# Sur RunPod :
unzip arabic-diacritization-pipeline.zip
cd arabic-diacritization-pipeline
```

### 4. Installation des dépendances

```bash
# Vérifier CUDA
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Installer le projet
pip install -e ".[dev]"

# Vérification complète
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
# Test fp16
x = torch.randn(100, 100, device='cuda', dtype=torch.float16)
y = torch.matmul(x, x.T)
print(f'fp16 test: OK ({y.shape})')
print('All checks passed ✓')
"
```

### 5. Préparer les données

```bash
# Upload tes données (si pas déjà incluses)
# Depuis local :
scp -P <PORT> -r data/raw root@<IP>:/workspace/arabic-diacritization-pipeline/data/

# Préparer (normalisation, splits, vocab, lexique)
python scripts/prepare_data.py --config configs/gpu_rtx4090.yaml
```

### 6. Lancer l'entraînement

```bash
# Entraînement GPU avec la config optimisée
python scripts/train.py --config configs/gpu_rtx4090.yaml

# OU via module (si installé)
python -m scripts.train --config configs/gpu_rtx4090.yaml
```

**Ce que tu dois voir au démarrage :**
```
Device: cuda
GPU: NVIDIA GeForce RTX 4090
VRAM: 24.0 GB
GPU: NVIDIA GeForce RTX 4090 (24.0 GB)
CUDA: 12.1 | cuDNN: 8902
Mixed precision (fp16): ON
Vocabulary: XXX chars
Train: XXXXX | Valid: XXXXX
Model params: X,XXX,XXX total, X,XXX,XXX trainable
```

### 7. Monitoring

```bash
# Dans un autre terminal
watch -n 2 nvidia-smi

# Ou suivre les logs
tail -f logs/training_history.json
```

### 8. Après l'entraînement

```bash
# Évaluer
python scripts/evaluate.py --config configs/gpu_rtx4090.yaml --checkpoint checkpoints/best.pt

# Télécharger le checkpoint
scp -P <PORT> root@<IP>:/workspace/arabic-diacritization-pipeline/checkpoints/best.pt ./

# Télécharger le rapport
scp -P <PORT> root@<IP>:/workspace/arabic-diacritization-pipeline/evaluation_report.json ./

# IMPORTANT : éteindre l'instance RunPod pour arrêter la facturation !
```

---

## Déploiement sur Vast.ai (alternative moins chère)

```bash
# 1. Choisir une instance RTX 4090 sur vast.ai (~$0.30-0.40/h)
# 2. Template : PyTorch 2.2 + CUDA 12.1
# 3. Même procédure que RunPod (SSH + upload + install + train)
```

---

## Vérification qualité CPU vs GPU

Après entraînement GPU, compare avec la config CPU (default.yaml) sur **le même test set** :

```bash
# Évaluer checkpoint GPU
python scripts/evaluate.py \
  --config configs/gpu_rtx4090.yaml \
  --checkpoint checkpoints/best.pt \
  --output eval_gpu.json

# Résultats attendus : DER et WER doivent être IDENTIQUES ou MEILLEURS
# La seule différence acceptable est liée au non-déterminisme GPU mineur
# (ordre des opérations flottantes), mais DER doit être dans ±0.001
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Réduire batch_size dans configs/gpu_rtx4090.yaml
# batch_size: 128 → 96 ou 64
```

### fp16 NaN/Inf
```bash
# Le CRF est déjà protégé en float32.
# Si problème persistant, désactiver fp16 :
# fp16: false dans la config
```

### Lenteur inattendue
```bash
# Vérifier que le GPU est bien utilisé :
nvidia-smi
# GPU-Util doit être > 80%

# Si GPU-Util bas, augmenter num_workers :
# num_workers: 8 → 12
```

### Reprise après interruption
```bash
python scripts/train.py \
  --config configs/gpu_rtx4090.yaml \
  --resume checkpoints/checkpoint_epoch_XX.pt
```

---

## Coût estimé total

| Scénario | Durée GPU | Coût (RTX 4090 RunPod) |
|---|---|---|
| Entraînement complet (50 epochs) | ~15-25h | ~$7-11 |
| Avec early stopping (~15 epochs) | ~5-8h | ~$2-4 |
| Entraînement + évaluation + debug | ~10-15h | ~$5-7 |

**Budget recommandé : ~$10-15 au total** pour un entraînement complet avec marge de debug.
