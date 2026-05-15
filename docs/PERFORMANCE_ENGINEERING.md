# Performance Engineering

The project has meaningful systems constraints:

- 3D volumes are memory-heavy
- patch size controls both context and GPU cost
- overlap improves boundary quality but increases runtime
- TTA improves robustness but multiplies inference cost
- model ensembles improve stability but increase latency

## Key Tradeoffs

| Lever | Benefit | Cost |
|---|---|---|
| Larger 3D patches | More spatial context | Higher VRAM use |
| Higher overlap | Smoother predictions | More inference windows |
| More ensemble members | Better robustness | More GPU time |
| TTA | Better invariance | Multiplicative runtime cost |
| Postprocessing | Fewer structural artifacts | Extra CPU/GPU work |

## Production Notebook Strategy

The inference notebook uses a time-budget ladder. Instead of a single brittle configuration, it can reduce compute progressively:

1. basic TTA, three models
2. no TTA, three models
3. lower overlap, three models
4. lower overlap, two models
5. smaller window, best model only
6. minimum fallback configuration

That design is directly relevant to production ML systems, where the best model is only useful if it can meet latency, memory, and reliability constraints.
