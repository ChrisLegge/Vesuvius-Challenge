# Four-Day Repository Upgrade Plan

Goal: build a visible, high-quality commit history while improving the repository in ways that matter to research engineering, big tech, HFT, NVIDIA, and quantitative finance reviewers.

Target: **50-70 meaningful commits** over four days.  
Stretch target: **80-100 commits** only if each commit is genuinely reviewable and coherent.

Avoid empty commit padding. Recruiters will not count commits one by one, but a clean sequence of small, purposeful commits signals strong engineering habits.

## Commit Style

Use concise conventional commit messages:

```text
docs: add architecture overview
test: validate inference config weights
feat: add hysteresis postprocessing helper
chore: tighten repository artifact ignore rules
refactor: split metadata validation helpers
```

Each commit should answer one question:

- What changed?
- Why does it matter?
- Can a reviewer understand it in under one minute?

## Day 1 - Research Repo Foundation

Theme: make the repository immediately understandable.

Target: 12-18 commits.

| Commit | Suggested Message | Change |
|---:|---|---|
| 1 | `docs: add project brief for recruiters` | Add or refine `docs/PROJECT_BRIEF.md` |
| 2 | `docs: document repository architecture` | Add `docs/ARCHITECTURE.md` |
| 3 | `docs: add experiment log table` | Add `docs/EXPERIMENTS.md` |
| 4 | `docs: add topology-aware technical deep dive` | Add `docs/TECHNICAL_DEEP_DIVE.md` |
| 5 | `docs: add performance engineering notes` | Add `docs/PERFORMANCE_ENGINEERING.md` |
| 6 | `docs: add reproducibility checklist` | Add `docs/REPRODUCIBILITY.md` |
| 7 | `docs: improve model card with limitations` | Expand `docs/MODEL_CARD.md` |
| 8 | `docs: clarify data and weights policy` | Improve `docs/DATA_AND_WEIGHTS.md` |
| 9 | `docs: add notebook index` | Improve `notebooks/README.md` |
| 10 | `docs: add metadata index` | Improve `outputs/metadata/README.md` |
| 11 | `docs: add report index` | Improve `reports/README.md` |
| 12 | `docs: add script index` | Improve `scripts/README.md` |
| 13 | `docs: add README badges` | Add badges to README |
| 14 | `docs: add standout technical artifacts section` | Link the best docs from README |
| 15 | `chore: add citation metadata` | Add or polish `CITATION.cff` |
| 16 | `chore: add MIT license` | Add or confirm `LICENSE` |
| 17 | `chore: tighten gitignore for ML artifacts` | Update `.gitignore` |
| 18 | `docs: add manual commit guide` | Polish `docs/MANUAL_COMMIT_GUIDE.md` |

## Day 2 - Software Engineering Layer

Theme: convert notebook ideas into testable code.

Target: 12-20 commits.

| Commit | Suggested Message | Change |
|---:|---|---|
| 1 | `chore: add package skeleton` | Add `src/vesuvius/__init__.py` |
| 2 | `feat: add inference config dataclass` | Add `InferenceConfig` |
| 3 | `feat: load inference config summary` | Add `load_inference_config()` |
| 4 | `test: validate inference config weights` | Test weights sum to one |
| 5 | `feat: define required metadata fields` | Add metadata constants |
| 6 | `feat: validate model metadata files` | Add `validate_model_metadata()` |
| 7 | `test: validate tracked model metadata` | Add metadata test |
| 8 | `feat: add hysteresis postprocessing helper` | Add `hysteresis_components()` |
| 9 | `test: keep weak regions connected to strong seeds` | Add hysteresis positive test |
| 10 | `test: remove weak isolated regions` | Add hysteresis negative test |
| 11 | `test: ensure large artifacts are excluded` | Add repository hygiene test |
| 12 | `chore: add pyproject package config` | Add `pyproject.toml` |
| 13 | `docs: document local test workflow` | Update README/Reproducibility |
| 14 | `refactor: clarify postprocess type aliases` | Clean `postprocess.py` |
| 15 | `refactor: tighten config validation errors` | Improve error messages |
| 16 | `refactor: tighten metadata validation errors` | Improve metadata messages |
| 17 | `test: cover invalid threshold ordering` | Add postprocess validation test |
| 18 | `test: cover malformed metadata score` | Add metadata failure test |
| 19 | `test: cover mismatched ensemble weights` | Add config failure test |
| 20 | `docs: add testing section to README` | Make tests visible |

## Day 3 - Research Quality and ML Systems Polish

Theme: show senior-level judgment around experiments, metrics, and systems constraints.

Target: 12-20 commits.

| Commit | Suggested Message | Change |
|---:|---|---|
| 1 | `docs: add metric-aware validation notes` | Add SurfaceDice/VOI/Topo explanation |
| 2 | `docs: add threshold stability discussion` | Explain threshold curves |
| 3 | `docs: add ensemble design rationale` | Expand specialist roles |
| 4 | `docs: add failure mode taxonomy` | Bridges, splits, cavities, dust |
| 5 | `docs: add ablation decision rules` | Keep/drop criteria |
| 6 | `docs: add production inference checklist` | Runtime, weights, TTA, thresholds |
| 7 | `docs: add risk register` | Technical risks and mitigations |
| 8 | `docs: add model metadata summary table` | Pull from JSON metadata |
| 9 | `feat: add metadata summary script` | Optional small script to print metadata |
| 10 | `test: cover metadata summary script` | If script is pure enough |
| 11 | `docs: add Kaggle environment notes` | GPU/runtime assumptions |
| 12 | `docs: add reproducibility caveats` | CUDA nondeterminism, data paths |
| 13 | `docs: add portfolio talking points` | How to discuss project in interviews |
| 14 | `docs: add future work roadmap` | Next improvements |
| 15 | `docs: add limitations section` | Honest constraints |
| 16 | `docs: add comparison to naive baseline` | Why this approach is stronger |
| 17 | `docs: add systems tradeoff table` | Accuracy vs latency/VRAM |
| 18 | `docs: add reviewer quickstart path` | What to read first |
| 19 | `docs: polish README opening` | Sharpen first 10 lines |
| 20 | `docs: add table of contents to key docs` | Improve navigation |

## Day 4 - Final Polish, Verification, and Presentation

Theme: make the repo feel finished.

Target: 12-20 commits.

| Commit | Suggested Message | Change |
|---:|---|---|
| 1 | `test: run local verification suite` | Commit updated test notes if any |
| 2 | `docs: add final repository audit` | Add `reports/REPOSITORY_AUDIT.md` |
| 3 | `docs: add no-large-files audit result` | Record artifact hygiene |
| 4 | `docs: add final file inventory` | Summarize tracked directories |
| 5 | `docs: add recruiter readme path` | Add "Start Here" section |
| 6 | `docs: add interview explanation guide` | How to explain project in 60 seconds |
| 7 | `docs: add HFT relevance notes` | Reliability, tests, performance constraints |
| 8 | `docs: add NVIDIA relevance notes` | GPU, 3D segmentation, inference optimization |
| 9 | `docs: add big tech relevance notes` | Reproducibility, maintainability, systems |
| 10 | `docs: add quant bank relevance notes` | Risk, validation, auditability |
| 11 | `docs: polish architecture diagram labels` | Improve Mermaid clarity |
| 12 | `docs: polish experiment table` | Make results crisp |
| 13 | `docs: polish model card` | Make limitations and intended use clearer |
| 14 | `chore: sort requirements` | Keep requirements tidy |
| 15 | `chore: final formatting pass` | Markdown consistency |
| 16 | `test: verify package imports` | Confirm `python -c "import vesuvius"` |
| 17 | `test: verify pytest suite passes` | Confirm local tests |
| 18 | `docs: update manual commit guide` | Add final push steps |
| 19 | `docs: add final project status` | Update README status |
| 20 | `chore: tag portfolio-ready state` | Optional tag: `v1.0-portfolio` |

## Suggested Daily Command Flow

At the start of each day:

```powershell
cd "C:\Users\aryaa\Documents\Codex\2026-05-15\https-github-com-chrislegge-vesuvius-challenge"
git status
git pull origin main
```

For each small commit:

```powershell
git add <files-you-edited>
git commit -m "type: short useful message"
```

After a batch:

```powershell
python -m pytest
git status
git push origin main
```

## Current Additions To Commit Now

These are the files already prepared and ready for your next commit:

```powershell
git add README.md requirements.txt pyproject.toml src tests docs
git commit -m "feat: add research engineering docs and tested utilities"
git push origin main
```

If `pytest` is missing:

```powershell
pip install pytest
python -m pytest
```

## Quality Bar

A strong 60-commit history beats a weak 100-commit history.

Good commits:

- add one concept
- improve one document
- add one test
- validate one assumption
- clean one module

Weak commits:

- fix typo after typo after typo
- repeatedly rename files
- add generated junk
- commit huge data
- make changes with no clear reason

The target is not to look busy. The target is to look like someone who can decompose ambiguous technical work into clean, reviewable steps.
