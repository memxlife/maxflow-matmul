# Inference co-design: proposal, models and recorded experiments

Start with the [general research proposal](general_research_proposal.md), which asks whether learning how to search can improve verified hardware–software co-design under a fixed budget. It is a proposed research program, not a completed adaptive solver.

- [Original matrix-tiling design](design.md): a shared-memory model using ordinary maximum flow.
- [Synthetic model results](results.md): checked predictions with explicit assumptions.
- [Recorded experiment results](experiment_results.md): simulator and RTX 5090 measurements, including unsuccessful trials and validation limits.
- [Historical homework proposal](proposal.md): an earlier, narrower research plan, preserved separately.

## Run the original synthetic model

Python 3.12 or another compatible Python 3 version is sufficient; no third-party packages are required.

```sh
python search.py --config example.json --output-dir .
python -m unittest discover -s . -v
```

[JSON rankings](rankings.json) and [CSV rankings](rankings.csv) contain every accepted and rejected shape. The example rates are synthetic; these rankings are not measured GPU performance. Later experiments use different execution models, as their reports explain.
