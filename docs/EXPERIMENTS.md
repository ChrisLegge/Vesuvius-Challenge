# Experiment Log

This log summarizes the research decisions that made it into the portfolio version of the repository.

| Run | Role | Hypothesis | Change | Result | Decision |
|---|---|---|---|---|---|
| Model A | Generalist | A stable 3D U-Net style model can anchor the ensemble | 192 x 192 x 192 patches, seed 42, fold 0 | best proxy score 0.5816 | Keep as primary anchor |
| Model B | Anti-merge | A specialist can reduce bridge-style topology failures | 160 x 160 x 160 patches, seed 1337, fold 1 | best proxy score 0.5110 | Keep for diversity and merge control |
| Model C | Surface specialist | Surface-focused training can improve boundary behavior | 192 x 192 x 192 patches, seed 2024, fold 2 | best proxy score 0.4902 | Keep as surface specialist |
| Production inference | Runtime robustness | Inference should degrade gracefully under Kaggle limits | TTA, overlap control, model count reduction | six degradation levels verified | Keep |

## What This Shows

The project was not treated as a single notebook experiment. It was treated as a system:

- define a failure mode
- train or configure a component against it
- validate with metadata and audit checks
- keep only components that serve a clear role

That workflow is important for production ML, quant research tooling, and high-performance engineering teams because it makes tradeoffs explicit.
