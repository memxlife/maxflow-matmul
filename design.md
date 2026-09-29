# Choosing shared-memory matrix tiles with ordinary maximum flow

## 1. Question and purpose

What tile shape produces the most matrix-multiplication outputs per second, under explicit storage limits and ideal service rates? Shared memory lets threads reuse inputs instead of fetching every operand from external memory. Larger tiles improve that reuse, but need more threads and storage. We want a small, executable model that makes this tradeoff inspectable.

This document is the implementation contract. It uses four sections because this is one prescribed model, not a comparison of research methods. Ordinary maximum flow is the requested solver. The demonstration uses synthetic parameters, not measurements of a real GPU or the course hardware. It does not run the course scorer.

## 2. Model: count the work needed for one complete output tile

### Matrix and thread organization

Let A have M rows and K columns, B have K rows and N columns, and C have M rows and N columns. Output row i, column j is obtained by multiplying matching input entries and adding over their shared dimension:

$$
C_{ij}=\sum_{t=0}^{K-1} A_{it}B_{tj}.
$$

One thread block computes an m-by-n tile of C. Each thread computes exactly one output, keeping one scalar accumulator and its two current operands. There is no register tiling or warp tiling. A warp is a group of W threads that execute instructions together. Consecutive threads follow consecutive output columns.

The block processes K in chunks of depth k. For each chunk, it stages an m-by-k tile of A and a k-by-n tile of B in block-local shared memory. There is one buffer: loading finishes before consumption, and all consumption finishes before the buffer is overwritten. Accumulators start at zero and remain live through all chunks. C is stored once; its old value is never read.

The baseline enumerates positive integer shapes with m dividing M, n dividing N, and k dividing K. It also requires n to be a multiple of W, so warps never cross output rows and there are no partial warps. All matrices use row-major storage. Each input, output, and accumulator occupies s bytes. These restrictions deliberately exclude tails and mixed precision.

One unit of flow means **one m-by-n output tile completed through all K**, not one depth-k chunk. This distinction prevents the solver from rewarding incomplete outputs.

### Storage determines legality

Let T be the maximum threads per block, S the shared-memory bytes available per block, and R the register bytes available per block. Reserve r scalar-sized registers per thread, with r at least three. The extra registers cover temporary values, addresses, and loop state; this is an abstract budget, not a compiler allocation measurement.

A tile must fit the thread limit:

$$
mn\leq T.
$$

Its single shared-memory buffer must fit:

$$
sk(m+n)\leq S.
$$

Its register reservation must fit:

$$
srmn\leq R.
$$

These are capacity constraints, not bandwidth edges. They establish that one block fits the model. They do not establish how many blocks can reside concurrently or whether the machine can sustain its peak rates.

### External-memory traffic

Call external memory HBM. Assume no input reuse through caches between blocks. Across K/k chunks, one block reads mK elements of A and Kn elements of B, then writes mn outputs. Let D_H be its HBM byte demand:

$$
D_H=s[K(m+n)+mn].
$$

This counts useful element bytes under ideal coalescing, with no transaction rounding or redundant transfers. Reuse within shared memory is already included: each thread does not fetch all its operands separately from HBM.

### Shared-memory service and broadcast

Staging the inputs writes K(m+n) elements to shared memory. At each reduction position t, a warp requests one common A address and W consecutive B addresses. The baseline assumes same-address broadcast: the A request needs one shared-memory word service, while B needs W. Requests from different warps are not merged.

There are mn/W warps. Across the full reduction, shared memory therefore serves Kmn/W A reads and Kmn B reads. Let D_S count the distinct-address bytes serviced, including staging writes:

$$
D_S=sK\left[m+n+\frac{mn}{W}+mn\right].
$$

A bank is an independently serviced partition of shared memory. The baseline assumes W single-word banks, consecutive B addresses distributed across them, and same-address A broadcast. The formula does not charge W physical reads for one broadcast.

However, it is an aggregate service bound, not an exact bank schedule. An A broadcast can leave most banks idle; requests from different warps may contend for a bank. Summing all bank capacities ignores both effects. Consequently, the shared-memory rate must mean ideal aggregate **distinct-address read/write bytes per second**. It must not mean effective bytes delivered to all receiving threads. Exact conflict timing would require a separate per-bank, per-instruction model.

### Explicit abstract register accounting

There is no universal physical register-byte count for a multiply-add. This baseline defines one reproducible accounting convention. For each multiply-add, two operands arrive from shared memory and write two registers; arithmetic reads these operands and the old accumulator, then writes the new accumulator. That is six scalar register accesses. Each output also needs one initialization write and one final read for the output store.

The baseline stages every HBM input through a temporary register before writing shared memory, adding one register write and one read per input element. Let D_R count these abstract register-service bytes:

$$
D_R=s[6mnK+2mn+2K(m+n)].
$$

Broadcast saves shared-memory reads, but each receiving thread still gets its own operand register write. This model omits address and loop instruction service, operand forwarding, separate read/write ports, and direct hardware copies. It is a declared abstraction, not measured physical register traffic. Any replacement must change both the demand formula and the meaning of the register service rate consistently.

### Convert every rate to complete output tiles per second

Let B_H, B_S, and B_R be the three service rates in bytes per second, under the preceding counting conventions. Let P be scalar multiply-adds per second, counting one multiply-add as one operation, not two floating-point operations. All four rates must cover the same hardware scope, such as an entire hypothetical device. Per-block storage limits remain per block.

A completed tile needs mnK multiply-adds. The four edge capacities are therefore:

$$
c_H=B_H/D_H.
$$

$$
c_S=B_S/D_S.
$$

$$
c_R=B_R/D_R.
$$

$$
c_P=P/(mnK).
$$

Bytes per second divided by bytes per tile gives tiles per second. Converting units before constructing the graph is essential: ordinary flow conservation cannot reconcile edges measured in different units.

## 3. Solver contract: one chain per legal shape

For each legal (m,n,k), build five vertices connected by four directed edges: source → after HBM → after shared service → after register service → sink. Assign the capacities c_H, c_S, c_R, and c_P in that order. Ordinary flow conservation requires equal incoming and outgoing flow at every intermediate vertex.

Every source-to-sink flow crosses every edge, so it cannot exceed the smallest capacity. Conversely, placing that smallest capacity on all four edges obeys every capacity constraint and conserves flow. Thus the maximum flow F is:

$$
F=\min(c_H,c_S,c_R,c_P).
$$

The implementation must nevertheless run an actual augmenting-path maximum-flow solver and independently compare its answer with this analytical cut. The chain says that every output consumes four resources. It does not describe four sequential GPU execution phases. This is an ideal steady-state upper bound, allowing service overlap between blocks and omitting startup, drain, synchronization, and dependency delays.

Tiles of different areas must not be ranked by tiles per second. Let Q be completed output elements per second, with larger values better:

$$
Q=mnF.
$$

The objective is to return every legal (m,n,k) maximizing Q over a supplied finite candidate set. At fixed M, N, and K, this also maximizes useful arithmetic throughput and minimizes the ideal time MN/Q. That time is a resource-service lower bound, not an exact runtime prediction.

Do not put candidate shapes on parallel paths to express “choose one shape.” Ordinary maximum flow could use several paths simultaneously and double-count hardware capacity. Shape selection happens in an outer enumeration; each graph describes only one shape.

The program accepts JSON containing dimensions, element bytes, warp width, per-block resource limits, register reservation, four rates, and lists of candidate m, n, and k values. Dimensions and budgets must be positive integers; rates must be positive and finite; r must be at least three. Invalid input raises an error. Illegal candidates remain in the results with specific rejection reasons.

JSON and CSV outputs record legality, storage usage, three demands, four capacities, solver flow, analytical flow, Q, and all tied bottlenecks. Legal candidates are sorted by decreasing Q, then increasing m, n, and k for display. Scores within relative tolerance 1e-12 are treated as tied when selecting all winners; display order does not express a preference. If no candidate is legal, the winner list is empty.

## 4. Worked example, checks, and limits

### A synthetic example

Set M=N=K=256, s=4 bytes, W=32. A block may use 1024 threads, 48 KiB of shared memory, and 64 KiB of registers; r=8. One KiB is 1024 bytes. The hypothetical whole-device rates are B_H=100 GB/s, B_S=1000 GB/s, B_R=2000 GB/s, and P=100 billion multiply-adds/s. One GB is one billion bytes.

For m=16, n=32, k=8, the block uses 512 threads, 1536 shared-memory bytes, and 16384 register bytes. It reads 12288 input elements and stores 512 outputs, giving D_H=51200 bytes. Shared service includes 4096 A reads, 131072 B reads, and 12288 staging writes, giving D_S=589824 bytes. Register demand is D_R=3248128 bytes, and arithmetic work is 131072 multiply-adds.

The four capacities are approximately 1953125, 1695421, 615739, and 762939 tiles/s. Register service is the smallest, so Q is approximately 315.3 million output elements/s. These are model calculations; no GPU kernel has been executed or timed.

### Why this baseline cannot choose a unique k

Increasing k increases staging work per chunk in direct proportion, but reduces the number of chunks K/k by the same factor. Their product is unchanged. Every output also still needs K multiply-adds. Thus D_H, D_S, D_R, and arithmetic work contain no k. Only storage legality depends on k, so all legal depths for fixed m and n must tie.

This is the smallest distinguishing test: evaluate several legal k values at fixed m and n. A difference in model scores indicates an implementation error. A difference in real hardware timing would show that synchronization, residency, or another omitted mechanism matters; it would not be a prediction of this baseline. Small graphs and exact counts test the software directly without an expensive course simulation.

### Completion and evidence boundaries

The demonstration enumerates m in {4,8,16,32,64}, n in {32,64,128}, and k in {1,8,32,128,256}. Verification covers resource rejection, malformed inputs, broadcast counts, full-K demands, units, ranking across different tile areas, and all winning ties. An independent tiny scalar multiplication checks that chunking preserves every output's full reduction. This is not GPU kernel verification. There are no training data or fitted parameters; the analytical cut and independent event counts supply the confirmation evidence.

Keep this document as the baseline contract. Preserve the program and generated rankings.json and rankings.csv alongside it, and record completed checks in a short separate result note. Real-device prediction would first require a matched timing comparison between two legal depths at fixed m and n, plus bank-service and residency evidence. Until then, the study establishes an optimum only inside this finite synthetic model, not a real speedup or a unique best depth.
