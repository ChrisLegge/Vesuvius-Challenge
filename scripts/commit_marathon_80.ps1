param(
    [switch]$PushAtEnd
)

$ErrorActionPreference = "Stop"

function Add-StepCommit {
    param(
        [string]$Path,
        [string]$Title,
        [string]$Body,
        [string]$Message
    )

    $directory = Split-Path -Parent $Path
    if ($directory -and -not (Test-Path $directory)) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
    }

    if (-not (Test-Path $Path)) {
        "# $($Path | Split-Path -LeafBase | ForEach-Object { $_ -replace '_', ' ' })`n" | Set-Content -Encoding utf8 $Path
    }

    @"

## $Title

$Body
"@ | Add-Content -Encoding utf8 $Path

    git add $Path
    git commit -m $Message
}

$status = git status --porcelain
if ($status) {
    Write-Host "Working tree is not clean. Commit or stash current changes first." -ForegroundColor Red
    git status --short
    exit 1
}

$steps = @(
    @{
        Path = "docs/portfolio/RECRUITER_BRIEF.md"
        Title = "Project Positioning"
        Body = "This project is positioned as a research engineering case study: ambiguous 3D data, GPU constraints, model-selection tradeoffs, and production-style repository hygiene."
        Message = "docs: add recruiter positioning note"
    },
    @{
        Path = "docs/portfolio/RECRUITER_BRIEF.md"
        Title = "One-Line Summary"
        Body = "Built a topology-aware 3D segmentation workflow for Kaggle's Vesuvius surface detection task, combining specialist models, robust inference, and reproducibility controls."
        Message = "docs: add one-line project summary"
    },
    @{
        Path = "docs/portfolio/RECRUITER_BRIEF.md"
        Title = "Why It Stands Out"
        Body = "The repository shows the complete system around the model: experiment design, model cards, metadata validation, performance tradeoffs, tests, and data-governance rules."
        Message = "docs: add portfolio differentiation note"
    },
    @{
        Path = "docs/portfolio/RECRUITER_BRIEF.md"
        Title = "Research Engineering Signal"
        Body = "The strongest signal is not a single notebook; it is the ability to turn a Kaggle workflow into a reviewable, testable, documented research repository."
        Message = "docs: add research engineering signal"
    },
    @{
        Path = "docs/portfolio/RECRUITER_BRIEF.md"
        Title = "Interview Hook"
        Body = "A useful interview opener is: I treated the challenge as a topology-risk problem, not just an overlap-score problem."
        Message = "docs: add interview hook"
    },
    @{
        Path = "docs/portfolio/RECRUITER_BRIEF.md"
        Title = "Best Files To Review"
        Body = "Start with README.md, docs/PROJECT_BRIEF.md, docs/ARCHITECTURE.md, docs/TECHNICAL_DEEP_DIVE.md, and docs/PERFORMANCE_ENGINEERING.md."
        Message = "docs: add best files to review"
    },
    @{
        Path = "docs/portfolio/RECRUITER_BRIEF.md"
        Title = "Role Fit"
        Body = "The project maps to research engineer, ML engineer, computer vision engineer, GPU inference, and quantitative technology roles."
        Message = "docs: add role fit summary"
    },
    @{
        Path = "docs/portfolio/RECRUITER_BRIEF.md"
        Title = "Repository Hygiene Signal"
        Body = "Large artifacts are intentionally excluded while metadata, configs, and tests remain tracked. This makes the repository safe to clone and easy to review."
        Message = "docs: add repository hygiene signal"
    },
    @{
        Path = "docs/portfolio/HFT_RELEVANCE.md"
        Title = "Why This Helps For HFT"
        Body = "HFT engineering values correctness under constraints. This project demonstrates validation, reproducibility, latency tradeoffs, and careful artifact control."
        Message = "docs: add HFT relevance overview"
    },
    @{
        Path = "docs/portfolio/HFT_RELEVANCE.md"
        Title = "Latency Mindset"
        Body = "The inference degradation ladder is a latency-control pattern: the system reduces compute instead of failing when resource budgets tighten."
        Message = "docs: connect inference ladder to latency"
    },
    @{
        Path = "docs/portfolio/HFT_RELEVANCE.md"
        Title = "Risk Controls"
        Body = "Topology failures are treated like tail-risk events. That framing is similar to engineering systems where rare failures dominate operational risk."
        Message = "docs: connect topology failures to risk"
    },
    @{
        Path = "docs/portfolio/HFT_RELEVANCE.md"
        Title = "Testing Signal"
        Body = "The local tests check config validity, metadata structure, postprocessing behavior, and artifact hygiene without needing a GPU."
        Message = "docs: add HFT testing signal"
    },
    @{
        Path = "docs/portfolio/HFT_RELEVANCE.md"
        Title = "Deterministic Review"
        Body = "The repository separates deterministic local checks from nondeterministic GPU notebook execution. This makes review faster and safer."
        Message = "docs: add deterministic review note"
    },
    @{
        Path = "docs/portfolio/HFT_RELEVANCE.md"
        Title = "Engineering Tradeoffs"
        Body = "Patch size, overlap, TTA, and ensemble size are presented as explicit tradeoffs rather than hidden notebook constants."
        Message = "docs: add HFT tradeoff note"
    },
    @{
        Path = "docs/portfolio/HFT_RELEVANCE.md"
        Title = "Auditability"
        Body = "Tracked metadata and production audit notes create an audit trail for model choices and inference settings."
        Message = "docs: add auditability note"
    },
    @{
        Path = "docs/portfolio/HFT_RELEVANCE.md"
        Title = "Failure Mode Thinking"
        Body = "The project documents failure modes before proposing fixes, which is a strong software engineering habit in high-pressure environments."
        Message = "docs: add HFT failure mode note"
    },
    @{
        Path = "docs/portfolio/NVIDIA_RELEVANCE.md"
        Title = "Why This Helps For NVIDIA"
        Body = "The project centers on GPU-heavy 3D segmentation, memory-aware inference, and practical deployment tradeoffs in notebook environments."
        Message = "docs: add NVIDIA relevance overview"
    },
    @{
        Path = "docs/portfolio/NVIDIA_RELEVANCE.md"
        Title = "3D Workload"
        Body = "3D CT segmentation stresses memory bandwidth, VRAM capacity, and sliding-window scheduling more than ordinary 2D image classification."
        Message = "docs: add 3D workload note"
    },
    @{
        Path = "docs/portfolio/NVIDIA_RELEVANCE.md"
        Title = "Inference Optimization"
        Body = "The system exposes inference knobs such as window size, overlap, model count, and TTA so performance can be tuned against quality."
        Message = "docs: add inference optimization note"
    },
    @{
        Path = "docs/portfolio/NVIDIA_RELEVANCE.md"
        Title = "GPU Memory Awareness"
        Body = "Patch-size decisions are documented as memory-quality tradeoffs, which is essential for production GPU workflows."
        Message = "docs: add GPU memory awareness note"
    },
    @{
        Path = "docs/portfolio/NVIDIA_RELEVANCE.md"
        Title = "Model Diversity"
        Body = "Specialist models are used for complementary failure modes, showing ensemble design beyond simple seed averaging."
        Message = "docs: add NVIDIA ensemble diversity note"
    },
    @{
        Path = "docs/portfolio/NVIDIA_RELEVANCE.md"
        Title = "Robust Deployment"
        Body = "The degradation ladder is a deployment reliability pattern: preserve output quality as much as possible when compute budget changes."
        Message = "docs: add robust deployment note"
    },
    @{
        Path = "docs/portfolio/NVIDIA_RELEVANCE.md"
        Title = "Metric-Aware Modeling"
        Body = "The technical notes connect model behavior to metric components, which is important when optimizing complex vision systems."
        Message = "docs: add metric-aware modeling note"
    },
    @{
        Path = "docs/portfolio/NVIDIA_RELEVANCE.md"
        Title = "Portfolio Signal"
        Body = "The repository demonstrates the ability to explain GPU-relevant design decisions clearly, not only to run a model."
        Message = "docs: add NVIDIA portfolio signal"
    },
    @{
        Path = "docs/portfolio/BIG_TECH_RELEVANCE.md"
        Title = "Why This Helps For Big Tech"
        Body = "The project shows maintainability, testing, documentation, reproducibility, and systems thinking around an ML workflow."
        Message = "docs: add big tech relevance overview"
    },
    @{
        Path = "docs/portfolio/BIG_TECH_RELEVANCE.md"
        Title = "Maintainability"
        Body = "Notebook-heavy work is wrapped with a clear repository structure, README navigation, model cards, and local utilities."
        Message = "docs: add maintainability note"
    },
    @{
        Path = "docs/portfolio/BIG_TECH_RELEVANCE.md"
        Title = "Reproducibility"
        Body = "The repository documents data paths, environment assumptions, metadata, and known nondeterminism."
        Message = "docs: add big tech reproducibility note"
    },
    @{
        Path = "docs/portfolio/BIG_TECH_RELEVANCE.md"
        Title = "Reviewability"
        Body = "Small source utilities and tests give reviewers something concrete to inspect without downloading competition data."
        Message = "docs: add reviewability note"
    },
    @{
        Path = "docs/portfolio/BIG_TECH_RELEVANCE.md"
        Title = "Production Thinking"
        Body = "The production audit report and inference checklist show that deployment settings were checked systematically."
        Message = "docs: add production thinking note"
    },
    @{
        Path = "docs/portfolio/BIG_TECH_RELEVANCE.md"
        Title = "Communication"
        Body = "The docs turn complex modeling choices into readable technical narratives, which is a major part of senior engineering impact."
        Message = "docs: add communication signal"
    },
    @{
        Path = "docs/portfolio/BIG_TECH_RELEVANCE.md"
        Title = "Testing Without Heavy Data"
        Body = "The test suite validates invariants that do not require Kaggle data, keeping local checks fast."
        Message = "docs: add lightweight testing note"
    },
    @{
        Path = "docs/portfolio/BIG_TECH_RELEVANCE.md"
        Title = "Code Organization"
        Body = "The package layout separates reusable utilities from notebooks, which makes the project easier to extend."
        Message = "docs: add code organization note"
    },
    @{
        Path = "docs/portfolio/QUANT_BANK_RELEVANCE.md"
        Title = "Why This Helps For Quant Banks"
        Body = "Quant technology teams value validation, audit trails, reproducibility, risk-aware design, and clear communication of uncertainty."
        Message = "docs: add quant bank relevance overview"
    },
    @{
        Path = "docs/portfolio/QUANT_BANK_RELEVANCE.md"
        Title = "Risk Framing"
        Body = "The project frames rare topology failures as important tail events, not just average-score noise."
        Message = "docs: add quant risk framing note"
    },
    @{
        Path = "docs/portfolio/QUANT_BANK_RELEVANCE.md"
        Title = "Model Governance"
        Body = "Model metadata, limitations, and data policies are tracked to support governance-style review."
        Message = "docs: add model governance note"
    },
    @{
        Path = "docs/portfolio/QUANT_BANK_RELEVANCE.md"
        Title = "Validation Discipline"
        Body = "The experiment log records hypotheses, changes, outcomes, and decisions, which mirrors disciplined research workflows."
        Message = "docs: add validation discipline note"
    },
    @{
        Path = "docs/portfolio/QUANT_BANK_RELEVANCE.md"
        Title = "Operational Safety"
        Body = "The repository avoids committing heavyweight or sensitive artifacts while preserving the information needed to understand the system."
        Message = "docs: add operational safety note"
    },
    @{
        Path = "docs/portfolio/QUANT_BANK_RELEVANCE.md"
        Title = "Readable Evidence"
        Body = "Audit reports and metadata summaries make the model choices easier to inspect than raw notebook outputs alone."
        Message = "docs: add readable evidence note"
    },
    @{
        Path = "docs/portfolio/QUANT_BANK_RELEVANCE.md"
        Title = "Failure Controls"
        Body = "The system records how bridge, split, and runtime failures are detected or mitigated."
        Message = "docs: add failure controls note"
    },
    @{
        Path = "docs/portfolio/QUANT_BANK_RELEVANCE.md"
        Title = "Quant Interview Link"
        Body = "A strong interview framing is: I built a risk-aware ML system where tail failures mattered more than average-case appearance."
        Message = "docs: add quant interview framing"
    },
    @{
        Path = "docs/portfolio/VALIDATION_STRATEGY.md"
        Title = "Validation Philosophy"
        Body = "Validation focuses on model behavior, metadata consistency, threshold stability, and repository hygiene rather than a single headline score."
        Message = "docs: add validation philosophy"
    },
    @{
        Path = "docs/portfolio/VALIDATION_STRATEGY.md"
        Title = "Config Invariants"
        Body = "The ensemble weights must match the active model count and sum to one. This prevents silent production misconfiguration."
        Message = "docs: add config invariant note"
    },
    @{
        Path = "docs/portfolio/VALIDATION_STRATEGY.md"
        Title = "Metadata Invariants"
        Body = "Tracked metadata must include model role, seed, fold, patch size, validation score, thresholds, and temperature."
        Message = "docs: add metadata invariant note"
    },
    @{
        Path = "docs/portfolio/VALIDATION_STRATEGY.md"
        Title = "Postprocessing Invariants"
        Body = "Hysteresis thresholding must preserve weak regions attached to confident seeds and remove weak isolated regions."
        Message = "docs: add postprocess invariant note"
    },
    @{
        Path = "docs/portfolio/VALIDATION_STRATEGY.md"
        Title = "Artifact Invariants"
        Body = "The repository should not contain checkpoints, extracted data, submission zips, NumPy dumps, or CT TIFF files."
        Message = "docs: add artifact invariant note"
    },
    @{
        Path = "docs/portfolio/VALIDATION_STRATEGY.md"
        Title = "Local Test Scope"
        Body = "Local tests are intentionally lightweight so they can run without a GPU, Kaggle account, or competition dataset."
        Message = "docs: add local test scope note"
    },
    @{
        Path = "docs/portfolio/VALIDATION_STRATEGY.md"
        Title = "Notebook Validation"
        Body = "Notebook validation is documented through audit reports and metadata because full GPU reruns are expensive."
        Message = "docs: add notebook validation note"
    },
    @{
        Path = "docs/portfolio/VALIDATION_STRATEGY.md"
        Title = "Review Flow"
        Body = "A reviewer can inspect README, architecture docs, model card, metadata, tests, and audit notes in that order."
        Message = "docs: add validation review flow"
    },
    @{
        Path = "docs/portfolio/FAILURE_MODES.md"
        Title = "Bridge Failures"
        Body = "A bridge failure occurs when nearby surfaces are accidentally connected, often causing a large topology penalty."
        Message = "docs: add bridge failure mode"
    },
    @{
        Path = "docs/portfolio/FAILURE_MODES.md"
        Title = "Split Failures"
        Body = "A split failure occurs when a continuous surface is fragmented into disconnected components."
        Message = "docs: add split failure mode"
    },
    @{
        Path = "docs/portfolio/FAILURE_MODES.md"
        Title = "Boundary Drift"
        Body = "Boundary drift reduces surface-distance agreement even when the predicted region visually resembles the target."
        Message = "docs: add boundary drift failure mode"
    },
    @{
        Path = "docs/portfolio/FAILURE_MODES.md"
        Title = "Threshold Instability"
        Body = "A threshold-unstable model changes topology under small threshold shifts, making it risky for production inference."
        Message = "docs: add threshold instability mode"
    },
    @{
        Path = "docs/portfolio/FAILURE_MODES.md"
        Title = "Runtime Timeout"
        Body = "Runtime timeout is an operational failure mode. A model that cannot finish inference is not useful regardless of its offline score."
        Message = "docs: add runtime timeout mode"
    },
    @{
        Path = "docs/portfolio/FAILURE_MODES.md"
        Title = "Artifact Leakage"
        Body = "Artifact leakage occurs when large or private data files are committed, making the repository harder to clone and review."
        Message = "docs: add artifact leakage mode"
    },
    @{
        Path = "docs/portfolio/FAILURE_MODES.md"
        Title = "Proxy Mismatch"
        Body = "Proxy mismatch happens when local validation signals diverge from the competition metric or production objective."
        Message = "docs: add proxy mismatch mode"
    },
    @{
        Path = "docs/portfolio/FAILURE_MODES.md"
        Title = "Ensemble Correlation"
        Body = "Highly correlated ensemble members can add runtime without reducing failure risk. Specialist diversity is more useful."
        Message = "docs: add ensemble correlation mode"
    },
    @{
        Path = "docs/portfolio/SYSTEM_DESIGN_NOTES.md"
        Title = "System Boundary"
        Body = "The repository separates local review utilities from Kaggle notebook execution. That boundary keeps local validation fast."
        Message = "docs: add system boundary note"
    },
    @{
        Path = "docs/portfolio/SYSTEM_DESIGN_NOTES.md"
        Title = "Artifact Boundary"
        Body = "Source code, metadata, and docs are tracked. Heavy data, checkpoints, and generated submissions are excluded."
        Message = "docs: add artifact boundary note"
    },
    @{
        Path = "docs/portfolio/SYSTEM_DESIGN_NOTES.md"
        Title = "Training Boundary"
        Body = "Training notebooks define model behavior; metadata files summarize important training outputs for review."
        Message = "docs: add training boundary note"
    },
    @{
        Path = "docs/portfolio/SYSTEM_DESIGN_NOTES.md"
        Title = "Inference Boundary"
        Body = "Inference configuration captures ensemble, threshold, TTA, and runtime choices separately from the trained weights."
        Message = "docs: add inference boundary note"
    },
    @{
        Path = "docs/portfolio/SYSTEM_DESIGN_NOTES.md"
        Title = "Testing Boundary"
        Body = "Tests validate stable logic and repository invariants, not expensive GPU training behavior."
        Message = "docs: add testing boundary note"
    },
    @{
        Path = "docs/portfolio/SYSTEM_DESIGN_NOTES.md"
        Title = "Documentation Boundary"
        Body = "The docs explain why decisions were made, while notebooks show how the experiments were implemented."
        Message = "docs: add documentation boundary note"
    },
    @{
        Path = "docs/portfolio/SYSTEM_DESIGN_NOTES.md"
        Title = "Review Boundary"
        Body = "The first-review path is README, project brief, architecture, technical deep dive, performance notes, and tests."
        Message = "docs: add review boundary note"
    },
    @{
        Path = "docs/portfolio/SYSTEM_DESIGN_NOTES.md"
        Title = "Maintenance Boundary"
        Body = "Utility scripts are retained for transparency but documented separately from primary notebooks."
        Message = "docs: add maintenance boundary note"
    },
    @{
        Path = "docs/portfolio/INTERVIEW_ANSWERS.md"
        Title = "Tell Me About The Project"
        Body = "I built a topology-aware 3D segmentation workflow for Vesuvius surface detection, then turned it into a reviewable research engineering repository."
        Message = "docs: add interview project overview"
    },
    @{
        Path = "docs/portfolio/INTERVIEW_ANSWERS.md"
        Title = "Hardest Technical Problem"
        Body = "The hardest part was handling structural errors where small local mistakes caused large topology changes."
        Message = "docs: add hardest problem answer"
    },
    @{
        Path = "docs/portfolio/INTERVIEW_ANSWERS.md"
        Title = "Why Not Just A Bigger Model"
        Body = "A bigger model does not automatically solve bridge and split failures. The project needed validation, postprocessing, and specialist diversity."
        Message = "docs: add bigger model answer"
    },
    @{
        Path = "docs/portfolio/INTERVIEW_ANSWERS.md"
        Title = "What I Would Improve"
        Body = "I would add stronger metric proxies, automate metadata dashboards, and run controlled ablations across more folds."
        Message = "docs: add improvement answer"
    },
    @{
        Path = "docs/portfolio/INTERVIEW_ANSWERS.md"
        Title = "What Shows Engineering Maturity"
        Body = "The repository excludes large artifacts, validates configs, tests pure utilities, documents limitations, and explains tradeoffs."
        Message = "docs: add engineering maturity answer"
    },
    @{
        Path = "docs/portfolio/INTERVIEW_ANSWERS.md"
        Title = "What Shows Research Maturity"
        Body = "The experiment log connects hypotheses to outcomes and makes model roles explicit."
        Message = "docs: add research maturity answer"
    },
    @{
        Path = "docs/portfolio/INTERVIEW_ANSWERS.md"
        Title = "What Shows Systems Thinking"
        Body = "The inference path includes a runtime degradation ladder and documents memory-quality tradeoffs."
        Message = "docs: add systems thinking answer"
    },
    @{
        Path = "docs/portfolio/INTERVIEW_ANSWERS.md"
        Title = "Concise Close"
        Body = "The project is a strong example of turning exploratory ML work into a structured, testable, and explainable engineering artifact."
        Message = "docs: add concise interview close"
    },
    @{
        Path = "docs/portfolio/ROADMAP.md"
        Title = "Metric Dashboard"
        Body = "Future work: create a lightweight dashboard that reads model metadata and displays validation proxy trends."
        Message = "docs: add metric dashboard roadmap"
    },
    @{
        Path = "docs/portfolio/ROADMAP.md"
        Title = "Ablation Registry"
        Body = "Future work: maintain a structured ablation registry with hypothesis, config diff, result, and decision."
        Message = "docs: add ablation registry roadmap"
    },
    @{
        Path = "docs/portfolio/ROADMAP.md"
        Title = "Metric Proxy Tests"
        Body = "Future work: add small synthetic masks for testing topology and surface-distance proxy behavior."
        Message = "docs: add metric proxy roadmap"
    },
    @{
        Path = "docs/portfolio/ROADMAP.md"
        Title = "Config Schema"
        Body = "Future work: formalize config validation with a schema so notebook and local configs cannot silently diverge."
        Message = "docs: add config schema roadmap"
    },
    @{
        Path = "docs/portfolio/ROADMAP.md"
        Title = "Notebook Smoke Tests"
        Body = "Future work: add notebook smoke tests that parse code cells and validate expected configuration constants."
        Message = "docs: add notebook smoke test roadmap"
    },
    @{
        Path = "docs/portfolio/ROADMAP.md"
        Title = "Model Card Automation"
        Body = "Future work: generate model-card tables directly from metadata JSON to reduce manual drift."
        Message = "docs: add model card automation roadmap"
    },
    @{
        Path = "docs/portfolio/ROADMAP.md"
        Title = "Performance Benchmarks"
        Body = "Future work: record inference time under each degradation level and summarize quality-runtime tradeoffs."
        Message = "docs: add performance benchmark roadmap"
    },
    @{
        Path = "docs/portfolio/ROADMAP.md"
        Title = "Packaging Improvements"
        Body = "Future work: move more reusable notebook logic into the src package while keeping Kaggle execution simple."
        Message = "docs: add packaging roadmap"
    },
    @{
        Path = "README.md"
        Title = "Recruiter Reading Path"
        Body = "For a quick review, read Project Brief, Architecture, Technical Deep Dive, Performance Engineering, and Reproducibility in that order."
        Message = "docs: add recruiter reading path"
    },
    @{
        Path = "README.md"
        Title = "Engineering Highlights"
        Body = "Highlights: topology-aware segmentation, specialist ensembles, runtime degradation, metadata validation, artifact hygiene, and local tests."
        Message = "docs: add engineering highlights"
    },
    @{
        Path = "README.md"
        Title = "Role-Relevant Skills"
        Body = "Skills demonstrated include PyTorch workflows, 3D segmentation, ML systems design, validation discipline, reproducibility, and technical communication."
        Message = "docs: add role relevant skills"
    },
    @{
        Path = "README.md"
        Title = "Final Review Note"
        Body = "This repository is intended to be read as a research engineering artifact: compact enough to review, but complete enough to explain design decisions."
        Message = "docs: add final review note"
    }
)

$index = 1
foreach ($step in $steps) {
    Write-Host "[$index/$($steps.Count)] $($step.Message)" -ForegroundColor Cyan
    Add-StepCommit -Path $step.Path -Title $step.Title -Body $step.Body -Message $step.Message
    $index++
}

Write-Host "Created $($steps.Count) commits." -ForegroundColor Green

if ($PushAtEnd) {
    git push origin main
}
