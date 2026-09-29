# Shared-memory matrix tiling with ordinary maximum flow

[Read the design contract](design.md) for the setup, traffic derivation, units, maximum-flow proof, and limitations. [Read the checked results](results.md) for the synthetic demonstration.

One thread computes one output element. For each legal tile shape, four serial edges represent HBM, shared-memory service, abstract register service, and scalar arithmetic. Every edge is measured in completed output tiles per second. An outer enumeration selects shapes by output elements per second.

Run with Python 3.12 or another compatible Python 3 version; no third-party packages are required:

```sh
python search.py --config example.json --output-dir .
python -m unittest discover -s . -v
```

The generated [JSON rankings](rankings.json) include every accepted and rejected candidate and all winning ties. [CSV rankings](rankings.csv) provide the same per-candidate quantities for inspection.

The rates in example.json are deliberately synthetic. This repository contains a model and checked demonstration, not a GPU implementation or calibrated benchmark.
