# MAE on Texture-Only Images (DTD)

**Masked Autoencoders under extreme context scarcity on the Describable Textures Dataset.**

## Goal

This project investigates how MAE behaves when global structure is minimal and images are dominated by texture statistics. We use the [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/) — 47 texture categories (e.g., striped, woven, bumpy) with no strong object-level semantics.

## Core Research Question

*How does masked reconstruction behave under increasing context scarcity when trained on structure-minimal texture images?*

- Does MAE rely on global structure?
- How robust is reconstruction under extreme masking?
- What is the geometry of learned texture representations?

## Experiments

| Config   | Mask ratio | Purpose                    |
|----------|------------|----------------------------|
| mask75   | 75%        | Standard baseline          |
| mask90   | 90%        | High masking               |
| mask95   | 95%        | Extreme masking            |

## Setup

```bash
# Option A: pip
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate mae-dtd
```

## Data

Download DTD and place it under `data/dtd/`:

```bash
# Example: download and extract DTD
mkdir -p data/dtd
# Place DTD images in data/dtd/ (e.g. images/, labels/ or standard DTD layout)
```

Expected layout (DTD standard):

```
data/dtd/
  images/
    banded/
    blotchy/
    ...
  labels/
    train1.txt, val1.txt, test1.txt
```

## Reproducibility

### 1. Pretrain MAE (per mask ratio)

```bash
# 75% masking (baseline)
python -m src.main pretrain --config configs/mask75.yaml

# 90% masking
python -m src.main pretrain --config configs/mask90.yaml

# 95% masking
python -m src.main pretrain --config configs/mask95.yaml
```

Checkpoints and logs go to `experiments/checkpoints/` and `experiments/logs/` (with run names derived from config).

### 2. Evaluate reconstruction

```bash
python -m src.main reconstruct --config configs/mask75.yaml --checkpoint experiments/checkpoints/mask75_run1/best.pt
```

### 3. Representation spectrum (eigenvalues, effective rank)

```bash
python -m src.main spectrum --config configs/mask75.yaml --checkpoint experiments/checkpoints/mask75_run1/best.pt
```

### 4. Linear probe

```bash
python -m src.main linear_probe --config configs/mask75.yaml --checkpoint experiments/checkpoints/mask75_run1/best.pt
```

### 5. Analysis (notebooks)

- `notebooks/visualize_reconstructions.ipynb` — visual comparison across mask ratios
- `notebooks/embedding_spectrum.ipynb` — eigenvalue spectrum, effective rank
- `notebooks/linear_probe_analysis.ipynb` — accuracy comparison (random vs 75/90/95)

### Colab (plug & play)

- **`notebooks/colab_mae_dtd.ipynb`** — run in Google Colab: setup → install → download DTD → train → evaluate → visualize. Upload the project (zip or clone from GitHub), set `MASK_RATIO` and `EPOCHS`, then run all cells.

## Project structure

```
mae-texture-dtd/
├── README.md
├── requirements.txt
├── environment.yml
├── configs/           # base, mask75, mask90, mask95
├── data/dtd/
├── src/
│   ├── datasets/      # DTD loader
│   ├── models/        # MAE, ViT encoder
│   ├── training/      # train_mae, linear_probe, trainer
│   ├── evaluation/    # reconstruction, metrics, spectrum
│   ├── utils/         # masking, logging, seed
│   └── main.py
├── experiments/       # logs, checkpoints, results
└── notebooks/         # analysis only
```

## Citation

If you use this code, please cite the original MAE paper and DTD:

- He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
- Cimpoi et al., "Describing Textures in the Wild", CVPR 2014
