# Commit Command Cookbook

Use this as a copy/paste guide. Run commands from PowerShell.

This plan creates a strong commit history without committing data, checkpoints, or generated junk.

## Setup

```powershell
cd "C:\Users\aryaa\Documents\Codex\2026-05-15\https-github-com-chrislegge-vesuvius-challenge"
git status
pip install pytest
python -m pytest
```

## Commit 1 - Current Prepared Upgrade

```powershell
git add README.md requirements.txt pyproject.toml src tests docs
git commit -m "feat: add research engineering docs and tested utilities"
git push origin main
```

## Day 1 - Documentation Foundation

For these commits, make a small edit to the named file before each `git add`. A one-paragraph improvement, table cleanup, or section addition is enough.

```powershell
git add docs/PROJECT_BRIEF.md
git commit -m "docs: sharpen recruiter project brief"
```

```powershell
git add docs/ARCHITECTURE.md
git commit -m "docs: clarify end-to-end architecture flow"
```

```powershell
git add docs/EXPERIMENTS.md
git commit -m "docs: expand experiment decision log"
```

```powershell
git add docs/TECHNICAL_DEEP_DIVE.md
git commit -m "docs: explain topology-aware segmentation risks"
```

```powershell
git add docs/PERFORMANCE_ENGINEERING.md
git commit -m "docs: document GPU memory and runtime tradeoffs"
```

```powershell
git add docs/REPRODUCIBILITY.md
git commit -m "docs: add reproducibility verification path"
```

```powershell
git add docs/MODEL_CARD.md
git commit -m "docs: improve model card limitations"
```

```powershell
git add docs/DATA_AND_WEIGHTS.md
git commit -m "docs: clarify data and checkpoint policy"
```

```powershell
git add notebooks/README.md
git commit -m "docs: improve notebook workflow index"
```

```powershell
git add outputs/metadata/README.md
git commit -m "docs: document tracked model metadata"
```

```powershell
git add reports/README.md
git commit -m "docs: document audit report purpose"
```

```powershell
git add scripts/README.md
git commit -m "docs: document utility script roles"
```

```powershell
git add README.md
git commit -m "docs: add recruiter start-here section"
```

```powershell
git add README.md
git commit -m "docs: highlight tested research utilities"
```

```powershell
git add CITATION.cff
git commit -m "chore: polish citation metadata"
```

```powershell
git add .gitignore
git commit -m "chore: tighten ML artifact ignore rules"
```

```powershell
git add docs/MANUAL_COMMIT_GUIDE.md
git commit -m "docs: update manual commit workflow"
```

```powershell
git push origin main
```

## Day 2 - Tests and Package Quality

Make the described small code/test edit before each commit.

```powershell
git add src/vesuvius/__init__.py
git commit -m "chore: expose package utility API"
```

```powershell
git add src/vesuvius/config.py
git commit -m "feat: add typed inference config object"
```

```powershell
git add src/vesuvius/config.py
git commit -m "feat: validate ensemble configuration constraints"
```

```powershell
git add tests/test_config.py
git commit -m "test: validate inference ensemble weights"
```

```powershell
git add src/vesuvius/metadata.py
git commit -m "feat: define model metadata requirements"
```

```powershell
git add src/vesuvius/metadata.py
git commit -m "feat: validate tracked model metadata"
```

```powershell
git add tests/test_metadata.py
git commit -m "test: cover model metadata validation"
```

```powershell
git add src/vesuvius/postprocess.py
git commit -m "feat: add hysteresis component filtering"
```

```powershell
git add tests/test_postprocess.py
git commit -m "test: keep weak regions connected to strong seeds"
```

```powershell
git add tests/test_postprocess.py
git commit -m "test: remove isolated weak regions"
```

```powershell
git add tests/test_repository_hygiene.py
git commit -m "test: prevent large artifacts in repository"
```

```powershell
git add pyproject.toml
git commit -m "chore: add installable package configuration"
```

```powershell
git add README.md docs/REPRODUCIBILITY.md
git commit -m "docs: document local pytest workflow"
```

```powershell
git add src/vesuvius/postprocess.py
git commit -m "refactor: clarify postprocess type aliases"
```

```powershell
git add src/vesuvius/config.py
git commit -m "refactor: improve config validation messages"
```

```powershell
git add src/vesuvius/metadata.py
git commit -m "refactor: improve metadata validation messages"
```

```powershell
git add tests/test_postprocess.py
git commit -m "test: cover invalid hysteresis thresholds"
```

```powershell
git add tests/test_metadata.py
git commit -m "test: cover malformed metadata scores"
```

```powershell
git add tests/test_config.py
git commit -m "test: cover invalid ensemble configuration"
```

```powershell
python -m pytest
git add tests src
git commit -m "test: verify utility test suite"
git push origin main
```

## Day 3 - Research Engineering Depth

Make one meaningful documentation addition before each commit.

```powershell
git add docs/TECHNICAL_DEEP_DIVE.md
git commit -m "docs: add metric-aware validation notes"
```

```powershell
git add docs/TECHNICAL_DEEP_DIVE.md
git commit -m "docs: add threshold stability discussion"
```

```powershell
git add docs/EXPERIMENTS.md
git commit -m "docs: explain specialist ensemble rationale"
```

```powershell
git add docs/TECHNICAL_DEEP_DIVE.md
git commit -m "docs: add segmentation failure taxonomy"
```

```powershell
git add docs/EXPERIMENTS.md
git commit -m "docs: add ablation decision rules"
```

```powershell
git add reports/README.md
git commit -m "docs: add production inference checklist"
```

```powershell
git add docs/PROJECT_BRIEF.md
git commit -m "docs: add project risk register"
```

```powershell
git add docs/MODEL_CARD.md outputs/metadata/README.md
git commit -m "docs: add model metadata summary"
```

```powershell
git add scripts
git commit -m "chore: organize experiment utility scripts"
```

```powershell
git add docs/REPRODUCIBILITY.md
git commit -m "docs: document Kaggle environment assumptions"
```

```powershell
git add docs/REPRODUCIBILITY.md
git commit -m "docs: add reproducibility caveats"
```

```powershell
git add docs/PROJECT_BRIEF.md
git commit -m "docs: add interview talking points"
```

```powershell
git add docs/PROJECT_BRIEF.md
git commit -m "docs: add future work roadmap"
```

```powershell
git add docs/MODEL_CARD.md
git commit -m "docs: add honest model limitations"
```

```powershell
git add docs/TECHNICAL_DEEP_DIVE.md
git commit -m "docs: compare against naive segmentation baseline"
```

```powershell
git add docs/PERFORMANCE_ENGINEERING.md
git commit -m "docs: add accuracy latency tradeoff table"
```

```powershell
git add README.md
git commit -m "docs: add reviewer quickstart path"
```

```powershell
git add docs/ARCHITECTURE.md
git commit -m "docs: polish architecture diagram labels"
```

```powershell
git add README.md docs
git commit -m "docs: improve research repo navigation"
git push origin main
```

## Day 4 - Final Portfolio Polish

```powershell
python -m pytest
git add tests
git commit -m "test: confirm local verification suite"
```

```powershell
git add reports
git commit -m "docs: add final repository audit notes"
```

```powershell
git add reports
git commit -m "docs: record no-large-files audit"
```

```powershell
git add reports README.md
git commit -m "docs: add final file inventory summary"
```

```powershell
git add README.md
git commit -m "docs: add recruiter reading path"
```

```powershell
git add docs/PROJECT_BRIEF.md
git commit -m "docs: add sixty second project explanation"
```

```powershell
git add docs/PROJECT_BRIEF.md
git commit -m "docs: connect project to HFT engineering skills"
```

```powershell
git add docs/PROJECT_BRIEF.md
git commit -m "docs: connect project to GPU research engineering"
```

```powershell
git add docs/PROJECT_BRIEF.md
git commit -m "docs: connect project to big tech ML systems"
```

```powershell
git add docs/PROJECT_BRIEF.md
git commit -m "docs: connect project to quantitative bank roles"
```

```powershell
git add docs/ARCHITECTURE.md
git commit -m "docs: refine architecture narrative"
```

```powershell
git add docs/EXPERIMENTS.md
git commit -m "docs: refine experiment outcomes"
```

```powershell
git add docs/MODEL_CARD.md
git commit -m "docs: refine model card intended use"
```

```powershell
git add requirements.txt
git commit -m "chore: sort runtime requirements"
```

```powershell
git add README.md docs
git commit -m "docs: final markdown formatting pass"
```

```powershell
python -c "import vesuvius; print(vesuvius.__all__)"
git add src
git commit -m "test: verify package import surface"
```

```powershell
python -m pytest
git add tests
git commit -m "test: verify final pytest suite"
```

```powershell
git add docs/MANUAL_COMMIT_GUIDE.md docs/COMMIT_COMMAND_COOKBOOK.md
git commit -m "docs: add final manual commit instructions"
```

```powershell
git add README.md
git commit -m "docs: update final portfolio project status"
```

```powershell
git tag v1.0-portfolio
git push origin main
git push origin v1.0-portfolio
```

## Stretch Commits 70-100

Only use these if you actually make the small described change first.

```powershell
git add docs/TECHNICAL_DEEP_DIVE.md
git commit -m "docs: add connected component discussion"
```

```powershell
git add docs/TECHNICAL_DEEP_DIVE.md
git commit -m "docs: explain bridge errors in surface masks"
```

```powershell
git add docs/TECHNICAL_DEEP_DIVE.md
git commit -m "docs: explain split errors in surface masks"
```

```powershell
git add docs/PERFORMANCE_ENGINEERING.md
git commit -m "docs: add sliding window inference notes"
```

```powershell
git add docs/PERFORMANCE_ENGINEERING.md
git commit -m "docs: add test-time augmentation cost notes"
```

```powershell
git add docs/PERFORMANCE_ENGINEERING.md
git commit -m "docs: add overlap versus runtime analysis"
```

```powershell
git add docs/REPRODUCIBILITY.md
git commit -m "docs: add seed tracking notes"
```

```powershell
git add docs/REPRODUCIBILITY.md
git commit -m "docs: add data path checklist"
```

```powershell
git add docs/REPRODUCIBILITY.md
git commit -m "docs: add notebook execution checklist"
```

```powershell
git add docs/EXPERIMENTS.md
git commit -m "docs: add experiment acceptance criteria"
```

```powershell
git add docs/EXPERIMENTS.md
git commit -m "docs: add experiment rejection criteria"
```

```powershell
git add docs/EXPERIMENTS.md
git commit -m "docs: add proxy metric interpretation"
```

```powershell
git add docs/MODEL_CARD.md
git commit -m "docs: add expected model failure modes"
```

```powershell
git add docs/MODEL_CARD.md
git commit -m "docs: add ethical use note"
```

```powershell
git add docs/DATA_AND_WEIGHTS.md
git commit -m "docs: add checkpoint recreation notes"
```

```powershell
git add docs/DATA_AND_WEIGHTS.md
git commit -m "docs: add Kaggle data compliance note"
```

```powershell
git add README.md
git commit -m "docs: polish project headline"
```

```powershell
git add README.md
git commit -m "docs: add concise technical highlights"
```

```powershell
git add README.md
git commit -m "docs: add repository quality highlights"
```

```powershell
git add src/vesuvius/config.py tests/test_config.py
git commit -m "test: cover invalid runtime budget"
```

```powershell
git add src/vesuvius/config.py tests/test_config.py
git commit -m "test: cover invalid degradation level"
```

```powershell
git add src/vesuvius/metadata.py tests/test_metadata.py
git commit -m "test: cover invalid threshold metadata"
```

```powershell
git add src/vesuvius/postprocess.py tests/test_postprocess.py
git commit -m "test: cover empty probability grid"
```

```powershell
git add tests/test_repository_hygiene.py
git commit -m "test: exclude generated cache directories"
```

```powershell
git add docs/PROJECT_BRIEF.md
git commit -m "docs: add role-specific interview bullets"
```

```powershell
git add docs/ARCHITECTURE.md
git commit -m "docs: add training versus inference boundary"
```

```powershell
git add docs/ARCHITECTURE.md
git commit -m "docs: add artifact flow explanation"
```

```powershell
git add reports/README.md
git commit -m "docs: add audit evidence summary"
```

```powershell
git add notebooks/README.md
git commit -m "docs: add notebook execution order"
```

```powershell
git add scripts/README.md
git commit -m "docs: clarify script maintenance status"
```

```powershell
python -m pytest
git push origin main
```

