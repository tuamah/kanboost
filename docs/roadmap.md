# Roadmap

The full, versioned roadmap (shipped features per release, deferred
items with reasons, and honest limitations) lives in
[`ROADMAP.md`](https://github.com/tuamah/kanboost/blob/main/ROADMAP.md)
at the repo root — kept there rather than duplicated here so there's
one source of truth that doesn't go stale between a release and a docs
rebuild.

## Honest limitations (see `ROADMAP.md` for the full list)

- **Speed**: each weak learner is a full KAN forward/backward pass in
  pure PyTorch — slower per-iteration than a histogram-based tree
  split. `batch_size` helps on large datasets but doesn't close this gap.
- **Multiclass is one-vs-rest**, not a single joint softmax objective —
  `n_classes` independent binary chains, `n_classes` times the training cost.
- **Categorical encoding** is a simple smoothed target-mean encoder, not
  CatBoost's ordered boosting scheme.
- **Monotonic constraints require `gam=True` and `kan_hidden=1`** — the
  guarantee only holds for a pure additive ensemble.

## Deferred, with reasons

- **`torch.compile` / ONNX export / FastKAN backend** — pykan's `KAN`
  modules don't trace or compile cleanly out of the box.
- **Multi-GPU** — the bottleneck is the number of sequential boosting
  rounds, not per-learner compute.
- **CLI** — the sklearn-style Python API already covers realistic usage
  patterns.

## Contributing

Issues and PRs welcome, especially:

- speed optimizations for the per-iteration KAN fit
- better categorical encoding
- benchmark results on additional public datasets
