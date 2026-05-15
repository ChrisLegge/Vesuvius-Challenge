# Data and Weights Policy

This repository intentionally does not commit competition data or trained checkpoint files.

## Data

The Vesuvius Challenge Surface Detection dataset should be downloaded from Kaggle:

https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection

Expected local/Kaggle layout depends on the notebook being run, but the project assumes Kaggle-style input paths such as:

```text
/kaggle/input/vesuvius-challenge-surface-detection/
/kaggle/input/vesuvius-trained-weights/
```

## Weights

Large model artifacts are excluded by `.gitignore`:

- `*.pt`
- `*.pth`
- `*.ckpt`
- `*.safetensors`
- `*.onnx`

Only lightweight metadata and manifests are tracked under `outputs/metadata/`.

## Why

This keeps the repository:

- cloneable
- GitHub-safe
- competition-data compliant
- suitable for LinkedIn and portfolio review
- focused on research, code, and experiment design
