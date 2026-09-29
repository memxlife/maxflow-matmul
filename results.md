# Checked synthetic results

The prescribed baseline selects a 32-by-32 output tile. Four tested depths, k=1, 8, 32, and 128, tie at **318,471,337.58 output elements per second**. This is an ideal model bound, not measured GPU throughput.

## What was tested

The fixed example uses 256-by-256 matrices, four-byte elements, 32-thread warps, and the synthetic rates and capacities in [example.json](example.json). The search considers 75 shapes: 37 meet the constraints and 38 are rejected. All inputs and the accounting convention are defined in [the design contract](design.md).

For each winning shape, the full-K demands are 69,632 HBM bytes, 1,146,880 shared-service bytes, and 6,430,720 abstract register-service bytes. The register edge limits flow to 311,007.17 complete tiles/s. Multiplying by 1024 output elements per tile gives the reported score. The four depths use 256, 2048, 8192, and 32768 shared-memory bytes respectively. Depth 256 would require 65536 bytes and exceeds the 49152-byte limit.

Smaller tiles can complete more tiles per second while producing fewer output elements per second. The search therefore ranks by output elements per second. It does not favor an arbitrarily smaller unit of work.

## Verification

Eight automated tests passed with the shared Python environment on September 29, 2026. They check a branching maximum-flow network, every legal chain against its analytical minimum cut, independent event counting for shared broadcast and register traffic, tiny scalar matrix multiplication across multiple depths, resource and shape rejection, malformed input rejection, score normalization and depth ties, and the worked example. These are deterministic checks, not repeated timing measurements or statistical estimates.

The independent event count explicitly enumerates warp addresses on a small problem. It confirms that one common A address is counted once per warp, while the receiving threads still receive individual register writes. The scalar multiplication check confirms coverage of the full reduction after splitting it into chunks; it does not validate an executable GPU kernel.

Full generated evidence is in [rankings.json](rankings.json), [rankings.csv](rankings.csv), and [test_search.py](test_search.py). No training or calibration data were used.

## Interpretation and evolution

The first and only baseline, synthetic-baseline-v1, is preserved unchanged. Its prediction that every legal depth ties at fixed output shape is confirmed. No search-efficiency improvement or hardware speedup was measured, so this study supplies no new measured advice for the course optimization workflow.

The shared-memory edge pools bank capacity and can hide idle banks and contention. Register traffic follows an explicit abstract convention. Synchronization, finite block concurrency, and instruction dependencies are omitted. These omissions explain why the result is a bound rather than a schedule. A future hardware study should compare two legal depths at fixed output shape before introducing a timing term to break their tie.

The Markdown mathematics passed source validation and the document received a standalone reader review. Opening the native viewer was requested; rendered equation appearance has not been independently inspected.
