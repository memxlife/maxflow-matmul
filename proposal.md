# Research proposal: verifier-guided modeling for homework accelerator optimization

**Historical snapshot.** Statements about pending grades describe the time of drafting. See [recorded results](experiment_results.md) for completed outcomes. Referenced course files are not included in this documentation snapshot.

**Status:** proposed research plan with explicitly identified preliminary evidence. The target is the homework accelerator and its unchanged simulator, not the native architecture of the remote RTX 5090 server. No new optimization round will start before this proposal is delivered. Graders already running may finish and preserve their results.

## 1. The question and the simple idea

Can a language model help a mathematical optimizer find better accelerator programs faster by changing what the optimizer understands about the machine?

Consider a matrix multiplication inside a Transformer. A larger input tile may reduce repeated memory reads. A simple bandwidth model therefore favors it. But the larger tile might leave fewer independent workgroups, delay the first useful operation, increase bank contention, or concentrate enough activity to violate the power limit. The optimizer can solve its equations correctly and still choose a worse program. The missing step is to identify which physical or scheduling effect the equations omitted, then test whether adding that effect improves subsequent decisions.

This proposal studies a repeated process with three distinct responsibilities. A large language model, or LLM, proposes a mechanism and a precise change to the mathematical model. A mixed-integer optimizer chooses legal configurations inside that model. “Mixed-integer” means that some variables make indivisible choices, such as selecting a tile, while others can represent quantities such as modeled time. An executable generator produces programs, and the unchanged course evaluator checks their mathematics, races, runtime and eligibility. The evaluator's disagreement with the model becomes evidence for the next revision.

The outcome we want is **more verified homework performance within a fixed research-time budget**. We do not count attractive generated code, fewer simulator calls, or a higher predicted score as success by themselves. Model construction, language-model interaction, optimization, generation and validation all consume time.

A simple implementation must be the starting point of the main prospective experiment. The existing advanced incumbent is valuable as a separate reference and as a continuation study, but it cannot replace the requested progression from a basic design to added mechanisms. We will not invent timestamps for the work that produced that incumbent.

## 2. Concrete target and controlled conditions

### 2.1 The two complete workloads

The authoritative workload specification (local evidence, not included) fixes two cases:

| Case | Model | Timed work | Why its optimization may differ |
| --- | --- | --- | --- |
| M1/P1, prefill | Three layers; width 256; eight attention heads; feed-forward width 1024 | Two 64-token prompts and their one-token continuations | Many input rows permit weight reuse and parallel work |
| M2/D1, decode | Two layers; width 128; four attention heads; feed-forward width 512 | Eight sequential new tokens for one input after a 128-token history | Small matrix products, repeated weights and step dependencies matter |

Prefill processes an existing prompt; decode produces subsequent token positions one at a time. The supplied inputs determine each new token's input vector, so this is teacher-forced execution rather than sampling text. The timing boundary, required hidden outputs, new keys and values, and commit points remain exactly those of the homework. Historical decode keys and values are constructed outside timed decode as specified by the reference.

Both cases use FP32 program values and the unchanged numerical reference. Optimizing a selected matrix product or one Transformer layer is only a diagnostic step. The research objective covers the entire pair, including normalization, attention, nonlinear activation, residual addition, memory movement and synchronization.

### 2.2 Fixed hardware for the main experiment

Use the exact fixed hardware file (local evidence, not included): sixteen streaming multiprocessors, or SMs; 256 KiB of shared memory per SM; eight shared-memory banks with two read ports and one write port per bank; four transfer engines per SM with queue depth two; 32 vector lanes; eight full-rate HBM channels; and no global cache. Preserve every other hardware field. Its checked area is 23.62586371072 square millimeters.

An SM executes workgroups. A workgroup owns private registers and a declared amount of shared memory. On this machine, each workgroup has 16,384 register words in total, divided into 32 partitions of 512 words. At most four workgroups can physically reside on one SM, subject also to shared-memory reservations. An instruction's register partition number is not a CUDA thread identifier.

Shared-memory banks service different addresses. The transfer engines move data between storage spaces, while the on-chip network connects SMs to external memory. HBM addresses map to channels by taking the integer part of byte_address/256 and then its remainder modulo eight. The eight logical SM pairs, (0,1) through (14,15), may preferentially schedule traffic to matching channels, but they do not own those channels exclusively. Merely balancing byte counts is not evidence of faster execution.

The remote machine may run the simulator or other research tools. Its RTX GPU throughput does not define the target's compute, memory or power behavior. Native RTX diagnostics already produced during scope discovery are preserved separately and excluded from homework claims.

### 2.3 What is authoritative

Use the preserved, unmodified v0.7.3 instruction specification (local evidence, not included), functional executor (local evidence, not included), race checker (local evidence, not included), event scheduler (local evidence, not included), power model (local evidence, not included), and scorer (local evidence, not included).

A race means two operations access overlapping data without sufficient ordering and at least one writes it. A functionally plausible result does not excuse a race. Likewise, a fast simulation does not establish numerical correctness. The complete grade supplies the final local verdict. The model is a teaching accelerator model, not a silicon measurement, and local admission remains distinct from grading-server confirmation.

## 3. Motivation from the supplied examples, without assuming their method

The supplied LLM4Branch paper (user-supplied manuscript; not redistributed here) separates executable policy structure from its parameters. An LLM proposes a branching-policy program skeleton; an inner parameter search evaluates candidate parameter vectors using solver feedback. Its reported setup uses eight training instances per benchmark, four of them for parameter tuning, 200 outer iterations, and 50 Bayesian-optimization iterations in the inner loop. The principal training objective is geometric-mean branching-node count; one difficult benchmark instead uses a primal–dual gap under a time limit. Thus reduced node count is not automatically reduced wall time. These are the paper's stated methods, not measurements reproduced here.

We adopt the useful separation between structural proposals and parameter optimization. We do not copy its budgets, assume that Bayesian tuning solves hardware constraints, or claim that it designs accelerator execution models. Here the LLM may revise resource representation, legal variables, dependencies and approximation boundaries, while a constrained mathematical optimizer selects accelerator configurations. The primary efficiency measure includes the complete research wall time.

The supplied OpenAI slide (user-supplied image; not redistributed here) illustrates a progression from functional kernels through a chip/simulator evaluator to end-to-end hardware validation. It labels an MLA example from 0.31% to 88.94% of its stated roofline and reports 1.5–1.8× speedups for GPT-OSS attention and mixture-of-experts blocks. The image does not disclose a reproducible optimization algorithm or sufficient roofline assumptions. We adopt its readable milestone presentation, not its numerical targets or an inferred algorithm.

Neither example establishes novelty or success for this proposal. A broader related-work review would be needed before making a field-wide novelty claim. The bounded contribution under test is whether revising an explicit accelerator model helps this homework search under equal resources.

## 4. Research questions and falsifiable hypotheses

The central question is whether adaptive modeling improves the quality of verified designs found per unit research time. Four hypotheses make that question testable.

| Hypothesis | Smallest useful comparison | Supporting outcome | Outcome that rejects or narrows it |
| --- | --- | --- | --- |
| H1: model revision helps when a missing interaction changes rankings | Same candidate family and budget, with a frozen model versus a revised model | Revision selects a candidate with better full verified performance earlier | Extra modeling time yields no better result, or the revised ranking remains wrong |
| H2: a constrained optimizer contributes beyond LLM proposals | Same generated mechanisms, with optimizer selection versus budget-matched random or LLM selection | More legal, useful candidates and better final eligible score at equal elapsed time | Selection overhead erases gains or simpler selection performs equally well |
| H3: adding mechanisms progressively improves diagnosis | Matched staged additions versus allowing all mechanisms at once | Fewer unexplained failures and better verified progress per hour | Staging delays combinations that the unrestricted method finds sooner |
| H4: full-workload feedback prevents misleading local promotion | Operator-only ranking versus connected and full-workload feedback | It avoids candidates whose local gains vanish or violate full eligibility | Local ranking already predicts all relevant outcomes, making added feedback unnecessary for this family |

These are hypotheses, not promised outcomes. A flat curve, a rejected candidate, or an ablation that matches the proposed method is a valid result. The smallest next experiment should distinguish a named explanation, rather than merely create another design.

## 5. What we optimize and what “best” means

Let T_P and T_D be the complete simulated cycle counts of a candidate's prefill and decode programs. Let B_P=30,996,995 and B_D=1,302,032 be the frozen reference cycle counts in the course baseline manifest (local evidence, not included). These are scoring constants, not the newly measured simple baseline on fixed hardware.

For an eligible pair, the official score rewards the geometric mean of the two speedups:

$$
S=1000\sqrt{\frac{B_P}{T_P}\frac{B_D}{T_D}}.
$$

A factor-of-two prefill gain and a factor-of-two decode loss cancel in this score. That is why optimizing one operator or reporting a simple average of speedups is insufficient.

Eligibility requires all correctness checks, area at most 24 square millimeters, peak power at most 20 W under the prescribed 1,000-cycle window, and each case's latency at most twice its frozen baseline. The course runs at 500 million cycles per simulated second. Report throughput using the scorer's convention: 128 prompt tokens divided by complete P1 time, and eight decode tokens divided by complete D1 time. P1 timing still includes its continuation even though that continuation is not added to the throughput numerator.

Let t be cumulative elapsed research time since a recorded start. Let V(t) be the set of exact program pairs that have completed all correctness and eligibility checks by time t. The primary curve is:

$$
S_{\mathrm{best}}(t)=\max_{x\in V(t)} S(x).
$$

Here x denotes a complete candidate pair on the fixed hardware. Before the first eligible result, the curve is unavailable, not zero. Once an eligible incumbent exists, rejected and slower candidates cannot lower the best-so-far curve. Separately plot correctness-checked cycle counts even for ineligible candidates, using clear failure markers. Never attach a course score to a candidate that failed a gate.

The initial practical target remains an eligible score of at least 50,000. Reaching it would meet the project target, but it would not by itself prove that the proposed research method beats another search method.

## 6. The first mathematical model, and why integers eventually matter

### 6.1 Keep the maximum-flow example as a bound

The earlier matrix contract (local evidence, not included) assigns one flow unit to a complete output tile. Every tile consumes HBM, shared-memory, register and arithmetic services. Dividing each resource rate by its demand produces four capacities in tiles per second. A serial chain's maximum flow is their minimum. Multiplying by the output tile's area permits comparison of differently sized tiles.

This is a useful resource bound, but it does not select a unique reduction depth when that depth cancels from the traffic formulas. It also does not express buffer lifetimes or mutually exclusive implementation choices. Enumerating four legal tile configurations is simpler and more transparent than forcing that example into a mixed-integer solver. We will retain enumeration as an exact cross-check whenever the domain is small.

### 6.2 Represent executable choices before building equations

For each workload, retain its operation graph: operations are nodes and required producer-to-consumer orderings are edges. For example, normalized activations must exist before their matrix multiplication reads them. Preserve shapes, addresses, arithmetic order, output ownership, and lifetime of every changed intermediate.

For operation o, let I_o be a finite set of executable implementation options. An option specifies tile dimensions, operand storage, number of buffers, workgroup partition and permitted instruction family. Its record includes resource demands, capacity requirements, generated-source estimate, and the assumptions used to estimate duration. Options that change an intermediate layout must include compatible producers and consumers together.

Let x_oi be a binary variable: one means operation o uses option i; zero means it does not. Exactly one option is chosen:

$$
\sum_{i\in I_o}x_{oi}=1.
$$

This is where an integer decision is necessary: choosing half of a one-buffer implementation and half of a two-buffer implementation is not an executable program. The same issue occurs when placing an indivisible workgroup on an SM. For a workgroup g and SM s, let z_gs be one if that group is assigned there. Its placement choices sum to one. Initially, placement may remain fixed while storage and partition decisions are explored; it becomes a variable only when evidence makes it consequential.

### 6.3 Initial resource constraints

For workload c, where c is P or D, let D_coir be the demand of implementation i for operation o on resource r. Let b_r be that resource's service per cycle. All demands use a compatible unit: bytes, arithmetic operations, or bank-word accesses. Let the nonnegative variable T_hat_c be the modeled complete cycle count. Every resource supplies a lower bound:

$$
b_r\widehat T_c\geq\sum_o\sum_{i\in I_o}D_{coir}x_{oi}.
$$

This inequality says that total demand cannot exceed what a resource can serve within the modeled duration. It does not promise that those demands overlap perfectly. An aggregate bound is deliberately optimistic; it is a starting model and a diagnostic, not the evaluator.

Add necessary capacity constraints before solving. Every RF view must stay inside one of 32 legal partitions and its 512-word extent. A workgroup's shared reservation must fit 256 KiB. Concurrent reservations on an SM must fit its total capacity, with at most four physical residents. An implementation with two live buffers pays for both. Exact register-coordinate checks and intermediate lifetime checks remain in the generator, because a scalar count alone cannot prove legal addressing.

The first implementation uses prechecked options whose local layouts already satisfy these constraints. Incompatible pairs, such as a changed producer layout with an unchanged consumer, are excluded by linear constraints. If x_a and x_b denote two incompatible binary choices, their sum is at most one.

Source text is also a resource. Enforce 8 MiB of emitted source, at most ten million expanded instructions and loop nesting at most 32. Option-level size estimates are conservative screening constraints; the actual emitted source is checked before admission. A failed exact check creates an exclusion for that combination or a corrected size model. Never enlarge the course parser limit.

### 6.4 Add dependencies and scheduling only when needed

A resource bound can miss an accumulator dependency or a late data arrival. When such a discrepancy changes a decision, add operation start times and chosen durations. A consumer's start must be no earlier than its producer's finish. Tasks using the same exclusive resource cannot overlap; a binary order variable chooses which goes first. For a bounded set of tasks, these are ordinary mixed-integer linear scheduling constraints with a justified finite scheduling horizon.

Transfer queues, multiple engines and buffer reuse create cumulative or interval constraints. For those subproblems, a CP-SAT scheduler may be clearer than a large time-indexed mixed-integer linear program. CP-SAT is a discrete constraint solver, not a continuous MILP solver. State which solver is used and what it proves. The research tests adaptive representations; it does not require every subproblem to use the same solver.

Do not split a banked resource merely to make the model look detailed. Split it when address mapping, ports or contention explain a measured ranking error. Similarly, channel balance, network service and workgroup residency need explicit representations when they affect execution. Model construction must remain cheaper than the mistakes it prevents.

### 6.5 Initial objective and the exact boundary of its guarantee

The true objective is the eligible course score. Its geometric-mean formula is nonlinear in the modeled times. The initial linear optimizer will use a bounded set of decode budgets. For each budget, constrain modeled decode time to that budget and minimize modeled prefill time. Compare the returned feasible pairs using the exact predicted score formula, then evaluate selected programs.

Start with decode budgets equal to 0.8, 0.9 and 1.0 times the current comparable decoder's measured cycles. These are experimental search settings, not hardware limits. Retain the incumbent as a feasible fallback and preserve the best measured component when hardware and case state are independent. This samples a tradeoff frontier; it does not prove a global optimum outside the tested options and budgets. On small spaces, enumerate all options and confirm the solver's result.

Peak window power is especially important. Average energy divided by total cycles is not a substitute for the 1,000-cycle peak constraint. The initial model can screen known unsafe combinations or use conservative, validated power envelopes, but it cannot certify exact peak power without a schedule and the authoritative calculation. Until that representation is available, correctness, peak power and full eligibility remain mandatory evaluation constraints. Record such gaps explicitly instead of claiming that a solver-feasible design is submission-ready.

## 7. The adaptive outer loop and the optimizer's inner loop

The inner loop solves a frozen mathematical problem. It receives versioned options, coefficients and constraints; emits selected choices, solve time, feasibility status and any optimality bound; and generates executable programs. An optimality bound limits the solver's objective inside that model. It is not a hardware roofline or a guarantee of the real course score.

The outer loop asks what the next model needs to understand. One iteration has six steps:

1. **Propose a causal mechanism.** Explain what repeated work, movement or waiting it removes and what cost it adds. Name an observation that would falsify the explanation.
2. **Freeze one model revision.** Add, remove or refine a variable or constraint. Record why it matters, its assumptions, and the exact implementable domain.
3. **Solve and generate.** Let the optimizer select admissible configurations. Use a small enumeration check where possible. Hash the emitted hardware and programs.
4. **Run the cheapest distinguishing check.** Check source and storage first, then bounded numerical/race tests and connected timing where applicable. Retain at most three materially distinct proposals: predicted best, uncertain, and structurally different.
5. **Validate the promising full pair.** An independent agent owns one immutable full grade. The main search may continue lightweight work only when it does not materially delay that grade.
6. **Explain agreement or disagreement.** Distinguish illegal generation, incorrect arithmetic, race, missing resource contention, power failure and statistical or operational noise. Update the model or reject the mechanism; do not merely tune coefficients until one example fits.

For example, fewer weight loads might predict a decoder improvement. If full timing gets worse and power rises, the next model should represent transfer overlap or activity concentration. It should not silently reduce the estimated cost of those loads. A separate matched probe can distinguish these explanations.

Each iteration records one question, its prediction, the experiment, the outcome and the decision it changes. The LLM may propose new mechanisms and models, but it cannot declare its own code correct or promote a partial result.

## 8. Mechanism coverage and staged experiments

Maintain a small knowledge table with columns for mechanism, supported instructions, known requirements, current evidence, model representation and next distinguishing test. This prevents repeated focus on matrix tiles while another stage dominates the complete program.

| Resource or stage | Mechanisms to consider | Important interaction to preserve |
| --- | --- | --- |
| Dense products | Tile width/depth, register reuse, shared weight panels | Register limits, first-use delay, instruction tails |
| Transfers and storage | Single/double buffering, prefetch distance, DMA overlap | Queue depth, source lifetime, overwritten buffers |
| Work placement | Rows per group, SM assignment, channel-aware intermediates | Lost parallelism, network load, producer placement |
| Attention | Query/key partition, softmax stability, value-product schedule | Causality, KV ownership, reduction order |
| Normalization and activation | Row partition, permitted fusion, parameter retention | Extra loads, changed consumer release, power windows |
| Complete workload | Cross-layer and cross-token retention, commit placement | ABI, scratch lifetime, future-token visibility |

The prospective sequence begins as follows. The stage order is a controlled experimental choice, not a claim that it is universally best.

| Stage | New freedom | Fixed comparison and admission |
| --- | --- | --- |
| 0: simple legal course program | None; m=8,n=8,k=16 dense tiles, one workgroup, direct HBM loads | Both full workloads, unchanged fixed hardware; establish correctness and actual baseline |
| 1: tile selection | n in {8,16}, k in {16,32}, m fixed at 8 | Optimizer chooses within RF constraints; complete grade decides whether it helps |
| 2: shared weight panels | Retain a K-by-16 weight panel across multiple output-row tiles | Keep Stage 1 arithmetic and RF tiles; preserve decoder when it has no row reuse |
| 3: parallel work distribution | Multiple workgroups and SM placement | Preserve producer/consumer dependencies and count duplicate data movement |
| 4: overlap | Two buffers, bounded prefetch distance and DMA scheduling | Prove live-buffer separation; test real producers and consumers |
| 5: broader full-program changes | Attention, normalization, activation and retention choices | Expose only supported transformations; validate the complete pair |

Stages 0 and the advanced normalization candidate are already being graded. Stage 1 and Stage 2 sources have been generated but their dispatch is paused for this proposal. Later stages are planned, not implemented commitments. The scalar maximum-flow example remains a teaching and unit-checking baseline; moving to course tensor instructions is an explicit change of implementation family.

At each stage, reject a feature if it fails correctness or offers no verified benefit within the allocated budget. A regression does not justify replacing the best program merely to maintain a visually rising mechanism list.

## 9. Fair comparisons, repeated searches and held-out evidence

Compare four search methods: the proposed LLM plus adaptive model plus optimizer; the same optimizer with a frozen initial model; LLM-directed code/configuration proposals without optimizer selection; and random search or exact enumeration where the domain is small. Give every method the same starting program, fixed hardware, legal instruction set, evaluator, source budget and initial evidence.

Use two distinct comparisons. A controlled comparison gives all methods the same mechanism catalog and tests selection and model revision. An exploratory comparison permits new mechanisms and tests the broader workflow. Do not use the exploratory result to claim that the optimizer alone caused the gain: the candidate spaces differ.

Apply the same wall-time and concurrency limits to every arm. Also report LLM calls/tokens when available, solver time, CPU-hours, evaluator calls and completed full grades. A method using many concurrent simulators must not look cheaper merely because its wall clock is shorter. Failures and timeouts consume its budget. Reuse of historical evidence must be identical across arms or separated into a warm-start condition.

The initial feasibility study is one search trajectory and cannot establish statistical superiority. A planned comparison uses three independent search seeds per method, with three hours per trajectory and the same two-hour evaluation cutoff plus one-hour completion allowance. No proposal is launched after the cutoff. Every admission still uses its actual completion time; work finishing outside three hours is censored from that budget's curve. These longer repetitions are a proposed study budget, not jobs launched by this document.

The simulator's cycle result is normally deterministic for fixed source and authority; repeating it does not create independent search trials. Replication therefore varies the search seed or LLM proposal trajectory. Report all trajectories, median progress and their range. Three runs give descriptive variability, not strong significance evidence.

For numerical robustness, use seed 7 for required local admission and seeds 19 and 43 for additional confirmation. Do not call those already inspected seeds a hidden test. Before a formal comparison, an independent reviewer should select additional nonnegative fixture seeds and keep their outputs unavailable to search until the final candidates are frozen. The official workload dimensions remain the score target. Tests on unseen matrix shapes or sequence lengths are a separate generalization experiment with their own contract and cannot replace or modify the official score.

## 10. Curves, bounds and milestone evidence

Produce three primary panels against the same cumulative research clock: best eligible prefill throughput, best eligible decode throughput, and best eligible combined score. Annotate a milestone only after its complete verdict arrives. Also show evaluated but rejected candidates in a companion plot, including correctness failures, power violations, timeouts and slower designs.

Start the prospective clock before mechanism/model work. Include LLM thinking and tool interaction, source inspection after the start, optimization, generation, validation, failed attempts and waiting for results. Record stage durations separately for diagnosis, but do not replace elapsed time with their sum when tasks run concurrently. A previously validated incumbent may appear at time zero only on a clearly labeled continuation plot. Previously generated stages are warm-start information and their prior costs must be disclosed.

A roofline is an optimistic resource ceiling, not a measured best result. For a fixed program, dividing useful work by the maximum of resource-demand/service lower bounds gives a conditional throughput ceiling. State every included demand, service unit, hardware scope and omitted effect. If a transformation changes traffic, its fixed-source memory bound also changes; do not present the old line as a universal limit. A candidate-dependent modeled bound belongs on a separate panel or explicitly changing line.

The solver's lower bound on modeled cycles is a different object again. Plot it separately from the authority's measured cycles. Do not report “percentage of roofline” until numerator and denominator refer to the same useful work and assumptions. No native RTX figure or the supplied slide's percentages enter the homework plots.

Keep a machine-readable evidence ledger containing clock origin, proposal time, model version, solver result, input hashes, authority hashes, generation result, checks, start/completion timestamps, resource use and promotion decision. Preserve raw reports so a reader can regenerate every plotted point. A timeout has no cycle result unless the authority completed and wrote one.

## 11. Preliminary evidence and current limits

The following observations motivate the proposal but do not establish the proposed method's overall effectiveness:

- The published synthetic tile study (local evidence, not included) checked a serial maximum-flow model and depth ties. It is not a GPU or homework runtime experiment.
- The eligible advanced incumbent (local evidence, not included) reports 875,199 prefill cycles, 56,040 decode cycles and a 28,685.88 local score on the fixed hardware. Its source hashes were checked against the imported baseline receipt (local evidence, not included).
- A CP-SAT normalization proposal (local evidence, not included) chose four rows per workgroup from matched operator costs. The connected first layer (local evidence, not included) took 112,237 cycles versus 130,629 for the incumbent layer, with peak power about 18.035 W. Changing partition count also changes later placements. Its full paired grade is pending at proposal writing; no full speedup is claimed.
- That candidate initially produced 8,528,916 source bytes and failed the 8 MiB limit. A static-loop naming repair reduced it to 7,914,630 bytes, with expanded instruction equivalence checked per complete workgroup. This exposed a missing generation constraint rather than a performance mechanism.
- A decoder load-merging candidate (local evidence, not included) passed full race checks and numerical seeds 7, 19 and 43, but took 57,186 cycles versus 56,040 and reached 20.206588 W. It is slower and ineligible. Fewer load instructions alone did not predict a better full design.
- The simple baseline source (local evidence, not included) and tile-selection result (local evidence, not included) establish a runnable prospective sequence. The initial full grade is in progress; subsequent grades are paused for proposal delivery.

These examples support investigating missing interactions and exact validation. They are not a fair multi-method comparison, a proof of generalization, or evidence that a score of 50,000 is reachable.

## 12. Resource budget, stopping rules and deliverables

For the current feasibility work, allow at most two concurrent local heavy evaluations while monitored memory remains comfortable. Each full grade is limited to 30 minutes and 4 GiB per process. Lightweight screens use at most 90 seconds and 2 GiB; inner optimization uses two seconds initially, increasing only when a recorded solve gap justifies it. Already running jobs finish under their existing guards. Dispatch of new rounds remains paused until proposal delivery.

A completed incumbent grade took about twenty minutes, while the decoder diagnostic completed in about twenty seconds. These are workload-specific measurements, not guarantees for the simple baseline, whose expanded prefill is larger. Use the first baseline grade to estimate the remaining sequence cost. If completing the sequence requires a materially larger agreed resource window, present its concrete estimate before expanding it. Use remote CPU resources only with preserved evaluator/runtime provenance, and record the host change in timing comparisons of search efficiency.

Stop a candidate on a source/hash mismatch, numerical error, race, exceeded resource guard or authoritative eligibility failure. Stop a mechanism branch after a matched negative result unless a specific new explanation yields a different distinguishing test. Stop the initial feasibility phase after a complete basic-to-tuned comparison and one additional mechanism test, or when the recorded budget ends; report uncompleted work explicitly. Meeting the 50,000 target is a project milestone. Testing whether the method is better than alternatives still requires the equal-budget study.

Deliver a versioned problem/model contract, executable generators and solver interfaces, exact simple and advanced reference programs, immutable evaluation receipts, a failure ledger, and reproducible performance-versus-time figures. The final report should explain which model changes altered useful decisions, their cost, and which hypotheses survived. A partial curve must be called partial.

Later hardware co-design is a separate experiment requiring an explicit permitted hardware domain. It must not quietly alter this fixed-hardware control, reduce the eight HBM channels or their service rate, or transfer success from a different architecture. Even an impressive homework score supports only a conclusion about the stated teaching model and workloads until validated elsewhere.
