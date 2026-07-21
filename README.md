# Faithful CNN-vs-ViT Explainability on Static ASL Fingerspelling

Research code for a conference paper comparing Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) through the lens of **faithful, architecture-agnostic explainability (XAI)** on the static ASL Alphabet dataset — with a novel **Hand-Region Alignment** contribution based on MediaPipe hand landmarks.

> **Research question.** Do ViTs produce more faithful and more linguistically aligned explanations than CNNs at *matched accuracy* on fine-grained static hand-shape recognition?
>
> **Hypothesis.** Because ViTs are more shape-biased and CNNs more texture-biased, ViT explanations should place more energy on the hand region (higher ROI alignment) and score higher on faithfulness for fingerspelling, where hand shape is the signal and background is a confound.

## Contributions
1. The first **faithful, architecture-agnostic** CNN-vs-ViT explanation comparison on static sign language recognition, using perturbation-based methods scored with quantitative faithfulness metrics.
2. A **Hand-Region Alignment protocol** (MediaPipe-derived ground-truth hand masks) measuring whether explanations focus on the linguistically relevant hand vs. background — directly exploiting the dataset's known background-leakage flaw.
3. A **leakage-controlled evaluation**: group-aware, near-duplicate-free splits plus a real-world out-of-distribution (OOD) test set.

## Data
- **Train:** Kaggle "ASL Alphabet" (grassknoted / Akash Nagaraj, 2018) — ~87,000 200×200 RGB images, 29 classes (A–Z, SPACE, DELETE, NOTHING).
- **OOD test:** "ASL Alphabet Test" (Dan Rasband, 2018) — 870 real-world images (30 × 29). Used to measure generalization beyond the controlled training set.

Place downloads under `data/raw/` (train) and `data/raw/ood/` (Rasband). These directories are gitignored.

> ⚠️ The training set has near-duplicate frames and background leakage that inflate in-distribution accuracy. This repo **never** uses a plain random split — see `CLAUDE.md` constraint #1.

## Models (matched capacity, matched recipe)
| Family | Model | ~Params |
|---|---|---|
| CNN | ConvNeXt-Tiny (primary), ResNet-50 (baseline) | ~28M / ~25M |
| ViT | DeiT-Small, Swin-Tiny (primary), ViT-B/16 (secondary) | ~22M / ~28M / ~86M |

All backbones come from `timm`, ImageNet-pretrained, trained under one identical recipe.

## XAI methods
- **Architecture-agnostic (primary, fair):** RISE, occlusion, Integrated Gradients, KernelSHAP.
- **Family-native (secondary):** Grad-CAM / Grad-CAM++ (CNN); Chefer relevance propagation + attention rollout (ViT).

## Metrics
Faithfulness (deletion/insertion AUC, comprehensiveness/sufficiency, infidelity, sensitivity), localization (pointing game, energy pointing game, IoU), and the novel **Hand-Region Alignment Score** — scored with [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus) and [Captum](https://captum.ai/).

## Quickstart
```bash
pip install -r requirements.txt

# 1. build leakage-free splits
python scripts/prepare_data.py --config configs/data.yaml

# 2. train (repeat per model & seed)
python scripts/train.py --config configs/model_convnext_tiny.yaml --seed 0

# 3. evaluate (in-distribution + OOD)
python scripts/evaluate.py --run experiments/<run_id>

# 4. explanations + metrics
python scripts/explain.py   --run experiments/<run_id> --config configs/xai.yaml
python scripts/score_xai.py --run experiments/<run_id> --config configs/xai.yaml

# 5. paper deliverables
python scripts/make_tables.py
python scripts/make_figures.py
```

## Repository layout
See `CLAUDE.md` for the annotated directory map and the non-negotiable scientific constraints. See `PLAN.md` for the phased implementation checklist.

## Status
Scaffold + planning stage. No model code yet — implementation proceeds phase by phase per `PLAN.md`.

## Notes
Code in this repo is written with the help of Claude Code / the Claude VS Code extension; `CLAUDE.md` is the shared project context the assistant reads.
