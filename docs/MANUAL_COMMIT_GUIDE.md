# Manual Commit Guide

This guide is for manually committing the repository under your own Git identity. It is intentionally written for a research repository, not coursework.

## Pre-flight

Open PowerShell in the repository root:

```powershell
cd "C:\Users\aryaa\Documents\Codex\2026-05-15\https-github-com-chrislegge-vesuvius-challenge"
git status
```

Confirm your identity:

```powershell
git config user.name
git config user.email
```

If needed:

```powershell
git config user.name "Chris Legge"
git config user.email "your-email@example.com"
```

Confirm the GitHub remote:

```powershell
git remote -v
```

If `origin` is missing, add it:

```powershell
git remote add origin https://github.com/ChrisLegge/Vesuvius-Challenge.git
```

## Commit 1 - Research Repository Structure

```powershell
git add README.md .gitignore requirements.txt LICENSE CITATION.cff docs/
git commit -m "docs: establish deep learning research repository"
```

## Commit 2 - Curated Kaggle Notebooks

```powershell
git add notebooks/
git commit -m "notebooks: add curated Vesuvius training and inference workflow"
```

## Commit 3 - Model Metadata and Audit Reports

```powershell
git add outputs/metadata/ reports/
git commit -m "reports: add model metadata and production audit evidence"
```

## Commit 4 - Project Utilities

```powershell
git add scripts/ configs/
git commit -m "chore: add notebook builders and experiment configuration snapshots"
```

## Commit 5 - Final Repository Polish

Run:

```powershell
git status
git ls-files | Select-String -Pattern "\.pt|\.pth|\.ckpt|\.tif|vesuvius-surface-detection.zip|__pycache__"
```

The second command should produce no output. Then:

```powershell
git add .
git commit -m "docs: polish portfolio presentation for Vesuvius surface detection"
```

## Push

```powershell
git push origin main
```

If your branch is `master`, use:

```powershell
git push origin master
```

## Important

Do not commit:

- model checkpoints
- Kaggle data zip files
- extracted TIFF volumes
- cache folders
- generated submissions unless you deliberately want to publish them

The repository should show the research process and implementation quality without becoming a data dump.
