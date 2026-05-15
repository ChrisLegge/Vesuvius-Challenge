# Reproducibility

This repository is designed to be reviewable without committing heavyweight competition files.

## Environment

Install the lightweight local environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest
```

## Data

Download the official dataset from Kaggle:

https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection

Kaggle notebooks expect paths similar to:

```text
/kaggle/input/vesuvius-challenge-surface-detection/
/kaggle/input/vesuvius-trained-weights/
```

## Determinism Notes

The metadata tracks:

- model seed
- fold
- patch size
- best validation proxy
- thresholds
- temperature
- training phase changes

Full GPU determinism is not guaranteed because CUDA kernels, Kaggle hardware, and notebook environments can vary.

## Local Verification

The local tests validate repository-level behavior:

- inference weights sum to 1
- metadata files have expected fields
- hysteresis postprocessing behaves as documented
- large data/checkpoint artifacts are not present
