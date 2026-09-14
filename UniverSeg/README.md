# UniverSeg Research Experiments

This repository contains experiments extending and evaluating [UniverSeg](https://github.com/JJGO/UniverSeg), a universal medical image segmentation model. It includes dataset loaders, training utilities, evaluation scripts, notebooks, metrics, and prediction visualizations.

## Repository contents

- `src/datasets/`: dataset loaders for ACDC, CAMUS, CHAOS, DRIVE, OASIS, and PanDental
- `src/training/`: training and loss functions
- `src/tasks/`: task definitions used by the experiments
- `experiments/`: training, inference, and evaluation entry points
- `notebooks/`: dataset inspection notebooks
- `results/metrics/`: small CSV summaries and metric plots
- `UniverSeg-main/`: the upstream UniverSeg implementation used as a reference

## Data and model weights

Raw medical datasets and trained model checkpoints are intentionally not stored in this repository. They can be large, subject to their own licenses, and may contain sensitive medical information. Obtain each dataset from its official source, confirm that you are allowed to use it, and place it under `data/raw/` using the directory names expected by the loaders.

Trained weights are also excluded from Git. Store them under `models/ours/checkpoints/` locally, or publish a selected release through GitHub Releases or an approved model repository instead of committing `.pth` files.

## Setup

Use Python 3.10 or a compatible version, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run commands from the repository root. For example:

```powershell
python experiments/04_acdc_evaluation.py
```

Check each experiment script before running it because dataset paths, checkpoint paths, GPU settings, and evaluation splits may need to be adjusted for your machine.

## Reproducibility

Keep raw data, local checkpoints, and generated predictions outside Git. Commit source code, dependency changes, notebooks, metric CSV files, and concise configuration needed to reproduce the reported results. Record dataset versions, preprocessing choices, random seeds, hardware, and checkpoint identifiers in the experiment documentation or result metadata.

## Citation and licenses

This work builds on UniverSeg. Please cite the original paper and comply with the licenses of the upstream code, model weights, and every dataset used. Dataset access and redistribution terms are separate from this repository's code.

```bibtex
@inproceedings{butoi2023universeg,
  title={UniverSeg: Universal Medical Image Segmentation},
  author={Butoi, Victor Ion and Gonzalez Ortiz, Jose Javier and Ma, Tianyu and Sabuncu, Mert R. and Guttag, John and Dalca, Adrian V.},
  booktitle={International Conference on Computer Vision},
  year={2023}
}
```
