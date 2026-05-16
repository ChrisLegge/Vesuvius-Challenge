# 80-Commit Marathon Script

This repo includes a PowerShell helper that creates **80 real commits** by adding meaningful portfolio/research-engineering notes under `docs/portfolio/`.

It is designed for your exact request: a copy/paste workflow that creates a stronger commit history while still making real repository improvements.

## Before You Run It

Make sure your working tree is clean:

```powershell
git status
```

If it is not clean, commit your current work first.

## Run

```powershell
cd "C:\Users\aryaa\Documents\Codex\2026-05-15\https-github-com-chrislegge-vesuvius-challenge"
powershell -ExecutionPolicy Bypass -File .\scripts\commit_marathon_80.ps1
git push origin main
```

Or push automatically at the end:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\commit_marathon_80.ps1 -PushAtEnd
```

## What It Adds

The script creates portfolio-focused docs covering:

- recruiter positioning
- HFT relevance
- NVIDIA relevance
- Big Tech relevance
- quant-bank relevance
- validation strategy
- failure modes
- system design notes
- interview answers
- future roadmap
- README portfolio highlights

## Important

This is not an empty-commit script. Every commit changes a real file.

Still, use judgment: a strong commit history should look deliberate. If you prefer a more natural pace, run the script on a feature branch, review it, and merge later.
