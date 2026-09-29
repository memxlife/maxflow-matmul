# Learning how to search: a self-evolving solver for AI inference co-design

**Research proposal.** This document proposes a general research program for AI inference infrastructure. It does not report a completed system or a demonstrated performance advantage. The earlier homework-specific [proposal](proposal.md) remains a separate historical document.

## 1. Why the search procedure should become a research object

An inference system must turn a trained model into useful responses within a latency, energy and cost budget. The designer can change the hardware, how the model runs on it, and how incoming requests share it. These choices interact. More arithmetic units may help long prompts but sit idle during small decode steps. More on-chip memory may avoid external reads but leave less chip area for computation. A larger batch may improve reuse while making individual users wait longer.

The decisions also operate on different timescales. Software and serving policies can be revised after deployment; chip structures are much harder to change. A design team therefore commits hardware against assumptions about future models and demand. NVIDIA describes modern chip verification as involving years of validation and many design iterations; that observation supports a long development horizon, not a universal duration for every chip. [NVIDIA's account of chip verification](https://blogs.nvidia.com/blog/vera-cpu-eda/).

Meanwhile, workload structure itself changes. For example, DeepSeek-V3 combines a mixture-of-experts model, which activates a subset of parameter blocks for each token, with an attention representation intended to reduce memory demand. Such changes affect which computation, storage and communication resources are useful. They illustrate why tuning for a fixed model is not the same as designing infrastructure for an evolving workload population. [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437).

The resulting risk is a mismatch: a carefully optimized architecture may arrive well suited to yesterday's workload. Faster design exploration can help, but only if verification keeps pace. Producing thousands of speculative designs is of little value when their feasibility, numerical behavior or physical cost remains uncertain.

This proposal asks whether search experience can become a reusable engineering asset. Instead of learning only which design is good, the system would learn which decomposition, search region, solver, heuristic and evaluation strategy tends to find good **verified** designs under particular conditions. When the workload changes, it could reuse that knowledge while testing whether its assumptions still hold.

The proposed research object is a **self-evolving solver**: a versioned procedure that changes its own search strategy in response to evidence. Its first implementation can evolve small executable heuristics and a persistent evidence store. Retraining the language model's weights is optional and is not assumed.

## 2. The central question and the opportunity

The central research question is:

> Under a fixed total exploration and verification budget, can a system that learns how to combine domain reasoning, structured optimization and uncertain experiments find better inference designs—and adapt to workload changes faster—than the same tools controlled by a fixed search procedure?

A large design space alone does not answer this question. Millions of variables can be manageable when the constraints have exploitable structure, while a small nonlinear black-box problem can be expensive. We should neither declare mixed-integer optimization impractical in general nor assume that one monolithic formulation can usefully represent every design decision.

The opportunity comes from repeated structure. Matrix operations recur across models. Buffer capacities impose exact limits. Producer–consumer dependencies constrain schedules. Workload classes often share bottlenecks, although their relative importance changes. This structure can make a bounded subproblem tractable even when the complete co-design problem is not.

A language model can propose promising decompositions and mechanisms using prior knowledge. A mathematical solver can enforce constraints and search a defined subproblem systematically. Bayesian or stochastic optimization can explore expensive choices without exact equations. Evaluators can reveal where the decomposition or prediction fails. The conjecture is that retaining experience about **how these tools should be combined** will be more useful across tasks than retaining only a table of winning designs.

The conjecture has clear failure cases. A fixed method may already exploit all useful structure. New workloads may differ too much for transfer. Policy learning may consume more resources than it saves. An adaptive procedure may overfit its simulator or repeatedly neglect unexpected regions. Any of these outcomes would limit or reject the proposed approach.

## 3. Distinguish a design, a model and a search policy

Three objects must remain separate.

A **design** is an executable or realizable system choice: for example, a memory hierarchy, a collection of compiled kernels, and a serving policy. A **performance model** predicts what a design will cost. A **search policy** chooses what to investigate next and how to investigate it.

Suppose a simulator shows that a large weight tile is slower than expected. Correcting the tile's estimated latency changes the performance model. Reducing the chosen tile changes the design. Learning to inspect residency before spending a full simulation budget on large tiles changes the search policy. All three may be useful, but only the third directly addresses the central contribution of this proposal.

A search action can select a design region, expose a previously fixed variable, choose a decomposition, allocate time to a solver, choose a branching heuristic, request a diagnostic, select an evaluation fidelity, or reopen a previously neglected region. A branching heuristic decides which discrete choice a solver should investigate next. It may change speed without changing the feasible set or the solver's proof rules.

An adaptive system must not quietly convert its preferences into facts. A region excluded by a proved capacity violation is different from a region postponed because a predictor considers it unpromising. The latter remains eligible for exploration and reconsideration.

## 4. A concrete first setting: a changing generative-inference service

The general scope includes models, operators, compiler mappings, schedules, memory and interconnect architectures, arithmetic units, and serving policies. The first study should follow one complete path through those layers rather than attempt to optimize everything immediately.

Use a generative-language-model service with both prefill and decode. Prefill processes a prompt; decode repeatedly produces subsequent token positions. The service receives requests with different prompt lengths, output lengths and arrival times. Its key/value cache stores attention information reused by later token positions. A larger cache consumes capacity but can avoid recomputation or external movement.

Begin with a parameterized accelerator family containing arithmetic clusters, a configurable on-chip memory hierarchy, external-memory interfaces and an inter-cluster network. Candidate hardware changes are central to this study. They must come from a realizable component catalog or a clearly labeled architecture model; an arbitrary bandwidth number without area, power or implementation support is not a design.

Software choices include matrix tiles, operand placement, fusion, buffer count, prefetch distance, parallel decomposition and request batching. The serving objective concerns completed requests that meet a latency requirement, rather than peak arithmetic throughput alone. This distinction has direct precedent: DistServe studies separate prefill and decode resource management to improve goodput, meaning useful request throughput under latency requirements. We use that interaction as a motivating example, not as evidence for our proposed search policy. [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin).

The first workload collection should contain a dense model family, several prompt/output-length distributions and both steady and bursty request arrivals. Introduce a mixture-of-experts family later as a more substantial shift involving conditional computation and communication. Fix numerical quality requirements before allowing precision or algorithm changes.

A prospective design sequence can start with a simple correct implementation, then add tile selection, shared-memory reuse, parallel scheduling and overlap. That sequence makes the effect of each mechanism visible. It is an experimental progression, not a required search order for every policy. The adaptive policy must be allowed to discover that a different order is better.

### An illustrative decision, not an experimental result

Assume a baseline service misses its response-time target when long prompts arrive in bursts. The LLM suggests three explanations: insufficient arithmetic capacity, interference between prefill and decode, or excessive cache movement. It proposes a decomposition that separates architecture allocation from per-phase kernel scheduling.

A mixed-integer subproblem allocates a bounded number of arithmetic and memory components subject to area and capacity limits. A scheduling solver chooses legal buffer use and overlap for representative operations. Bayesian optimization selects uncertain batching thresholds and a few hardware/software combinations for detailed evaluation. An event simulator then checks whether the proposed separation actually reduces interference.

If the larger memory option helps decode but harms prefill enough to reduce useful request throughput, the system retains that counterexample. It can revise the design and its performance model. More significantly, it can revise the search policy: similar future workloads should first measure phase interference, rather than begin by tuning arithmetic tiles. That revised procedure is accepted only if it helps on separate workload episodes after its own development cost is counted.

## 5. The design problem across workloads and timescales

Let h denote hardware choices, such as component counts and memory organization. Let a denote a software adaptation policy: it selects compiled mappings and serving settings from information available at deployment, such as the model shape and current request queue. Let w denote one workload scenario, including its model, request distribution and operating requirements.

The design is x=(h,a). Hardware is shared across scenarios; the software policy may adapt using declared observations. It cannot see future arrivals or evaluator answers that a deployed system would not know.

For scenario w, let L(x,w) be a specified tail-latency statistic, E(x,w) the energy per accepted request, and G(x,w) the goodput. Let C(h) represent a declared hardware cost estimate, including its costing assumptions. A scenario distribution P describes how frequently scenarios occur. A multiobjective design problem seeks a useful tradeoff among:

$$
\mathbf f(x;P)=\left(\mathbb E_{w\sim P}[L(x,w)],\mathbb E_{w\sim P}[E(x,w)],C(h),-\mathbb E_{w\sim P}[G(x,w)]\right).
$$

The minus sign makes higher goodput desirable when the other quantities are minimized. This equation defines a comparison, not a claim that expectations alone protect every workload. Report per-scenario performance and impose explicit service requirements for critical scenarios. Averaging a tail statistic across scenarios must never be called the tail statistic of their mixture.

A design is feasible only when its applicable constraints hold: functional and numerical requirements; memory capacity and lifetimes; resource and concurrency rules; supported compiler and instruction behavior; area, power, thermal and interconnect limits; and declared service-level requirements. Some are exact combinatorial checks. Others require simulation, synthesis or measurement. Their evaluator and uncertainty must be part of the constraint definition.

A **Pareto frontier** contains designs for which no other verified design is at least as good in every objective and strictly better in one. The main study preserves that frontier. A single utility may also be used when a deployment owner supplies explicit priorities; its weights must be fixed before comparing search methods.

### Uncertain future workloads

Hardware decisions must survive more than one forecast. Let U be a declared set of plausible workload distributions, constructed from observed workloads and explicit stress scenarios. A robust version of a scalar design loss J evaluates its worst expected loss within that set:

$$
J_{\mathrm{robust}}(x)=\max_{P\in\mathcal U}\mathbb E_{w\sim P}[J(x,w)].
$$

The set U, the loss J and its units are part of the experiment contract. This protects only against shifts represented in U; it does not predict unknown future architectures or demand. Compare robust and nominal design choices to expose the cost of flexibility.

Distinguish three decision horizons: architecture choices before physical commitment, compilation and provisioning choices before deployment, and online serving choices during execution. A faithful formulation enforces this ordering. It must not choose a different chip for each test request or assume future demand is already known.

### Why several optimization tools are needed

The general problem is mixed discrete/continuous and may be nonlinear. Integer choices include component counts, tile sizes and assignments. Continuous quantities may include modeled allocation fractions or control parameters, but voltage/frequency settings and other physical controls must use their actual legal operating domain.

A mixed-integer linear program is useful when a chosen subproblem can be expressed with linear relationships and indivisible decisions. A CP-SAT solver is useful for discrete constraints and scheduling. Neither label makes an inaccurate cost model authoritative. Bayesian optimization maintains an uncertain prediction of expensive outcomes and chooses informative or promising trials. Stochastic search supplies alternatives when representations are irregular or uncertainty models are unreliable. Tiny spaces should be enumerated directly.

For example, selecting one implementation per operator uses binary variables whose sum is one. Resource-demand bounds constrain the chosen implementations against hardware capacity. A buffer lifetime or task order may require additional discrete decisions. A serial maximum-flow resource bound does not express those choices by itself. Its value is as a cheap bound inside a larger procedure, not as a reason to use integer optimization where enumeration would suffice.

## 6. The second optimization problem: learn how to spend the search budget

Now make the search procedure itself an explicit object.

Let e denote a co-design episode: a workload collection, allowed architecture family, deployment requirements and evaluator access. Let π denote a search policy. It observes the current evidence and remaining budget, then chooses the next action. Let A_π(e,B) be the set of designs it has admitted at the required verification level by total budget B.

For comparisons requiring one number, let Q(A) be a prespecified measure of the quality of a verified design set. It may be the best feasible deployment utility or normalized Pareto hypervolume. Hypervolume measures the objective-space region dominated by admitted designs relative to fixed reference values; those values and normalization must be identical across methods.

Let C_learn(π) be the resource cost of developing the policy, including LLM use, tuning and verification. If that cost is amortized over n deployment episodes, a meta-level objective is:

$$
\max_{\pi}\;\mathbb E_e\left[Q\!\left(A_\pi(e,B)\right)\right]-\lambda\frac{C_{\mathrm{learn}}(\pi)}{n}.
$$

Here λ is a declared conversion from policy-development cost to the comparison utility; it is not a universal economic constant. Report the uncombined quality and cost numbers as well. Also impose explicit episode limits on elapsed time, CPU/GPU-hours, LLM spending and expensive evaluator calls. A single scalar budget does not make different resources interchangeable.

This objective separates two sources of improvement. The design optimizer tries to improve x within an episode. The policy optimizer tries to improve the procedure π across episodes. A new performance predictor helps the central thesis only if its use improves search decisions enough to justify acquisition and maintenance cost.

There is no free outer loop. Failed policies, prompt construction, policy testing, numerical tuning and discarded evidence all count. Report a cold-start condition that includes these costs and a warm-start condition that reuses a frozen policy. Show the number of future episodes needed to recover the up-front investment instead of asserting that reuse must eventually pay off.

## 7. A feasible self-evolving solver

### 7.1 Responsibilities and trust boundaries

| Component | Decision it owns | What it cannot establish by itself |
| --- | --- | --- |
| LLM proposer | Mechanisms, regions, decompositions, model changes and executable search heuristics | Correctness, physical feasibility, a valid bound or actual speedup |
| Mixed-integer/constraint solver | A structured subproblem with explicit constraints and objective | Global optimality outside that formulation or fidelity |
| Bayesian/stochastic optimizer | Expensive black-box settings and uncertain trial selection | A proof that an untested region is infeasible or inferior |
| Evaluators and checkers | Scoped measurements and verification judgments | Universal correctness beyond their semantics, coverage or abstraction |
| Search-policy controller | Which tool, region, fidelity and budget to use next | Permission to weaken the acceptance rules |
| Evidence store | Versioned observations, assumptions, failures and policy outcomes | Truth inferred from an unsupported textual summary |

The evaluator's acceptance contract is outside the evolving policy's write authority. Search code may request more evidence, but it cannot change a tolerance, remove a failing scenario, or substitute a cheap predictor for the promised verification level.

### 7.2 Executable policy representation

The first policy is a small program assembled from a restricted set of actions. Its input includes workload descriptors, architecture capabilities, candidate coverage, solver progress, prediction errors, observed evaluation cost and remaining budget. Its output selects an action and its parameters.

Actions include choosing a search region, choosing a decomposition template, selecting a solver/backend, allocating a timeout, proposing a neighborhood, requesting a fidelity level, reopening a region, or retaining the current procedure. Later actions may select an approved branching or neighborhood heuristic through a documented solver interface. Generated heuristics may reorder valid exploration; they may not falsify bounds or bypass feasibility checks.

The LLM produces a policy skeleton and an explicit parameter schema. Numerical optimization then tunes thresholds, allocation weights or exploration rates on training episodes. Programs run in a sandbox with restricted evaluator interfaces and bounded execution. Tests reject policies that exceed budgets, leak held-out information, make invalid calls or alter the trusted checker.

Start by evolving region selection and evaluation allocation. These choices are easier to observe and attribute than simultaneous changes to every internal solver rule. Add decomposition and branching decisions only after the initial policy can be evaluated reproducibly. Policy evolution need not mean changing every component on every episode.

### 7.3 How feedback changes the procedure

Maintain a versioned population containing the incumbent policy and a small number of challengers. For each challenger, replay or rerun matched training episodes from the same initial evidence. Compare complete verified progress under equal resources, not just its selected candidate's predicted score. Retain a challenger only when the prespecified aggregate result and regression guard justify promotion. Store its counterexamples even when rejected.

Feedback should say what failed and where. A slow solver suggests investigating formulation or decomposition. Many illegal generated programs suggest missing construction constraints. Frequent high-fidelity ranking reversals suggest inadequate predictors or an overaggressive fidelity policy. A policy that repeatedly chooses the same architecture family may need wider exploration, even if its local model remains accurate.

Keep two forms of persistent knowledge. First, store factual receipts: exact designs, inputs, tool versions, measurements and checker outcomes. Second, store conditional search lessons, such as “for this workload class, measure cache movement before allocating more arithmetic capacity.” A lesson carries its supporting cases, counterexamples and applicability conditions. It is a hypothesis to test when transferred, not a permanent rule.

### 7.4 Preserve exploration and reversibility

A learned policy can prioritize a promising region without erasing alternatives. The pilot will reserve a declared fraction of evaluations for diverse or uncertain candidates and test that fraction in an ablation. It will also reopen deferred regions when workload descriptors shift, model residuals grow, or a periodic coverage review finds long-neglected alternatives.

A certified exclusion records a checkable proof or bound under named assumptions. A heuristic exclusion records only a postponement and an expiration/revisit condition. If assumptions change, even a formerly certified exclusion must be rechecked. A solver bound is valid only for its encoded problem; a surrogate's low prediction is never a pruning certificate.

## 8. Allocate evaluation effort according to uncertainty and consequences

A multi-fidelity evaluator offers several levels of cost and detail. A quick analytical estimate may screen thousands of configurations; a cycle-level simulator may examine contention; synthesis may estimate physical area and timing; hardware measurement may test selected realizations. These levels are not automatically ordered in accuracy for every phenomenon.

The search policy should ask two questions before an expensive evaluation: could its result change the selected design or the search strategy, and is this evaluator capable of resolving that uncertainty? Testing another cheap approximation does not resolve a question about an effect all cheap approximations omit.

An acquisition rule can combine expected verified improvement, uncertainty reduction and evaluation cost. Its numerical values are predictions, so compare it against simpler cost-aware and random allocation rules. Learn fidelity relationships from paired evaluations and keep uncertainty visible. If cheap and detailed rankings disagree, increase diverse detailed sampling rather than assuming the cheap evaluator is uniformly biased.

Correctness and performance require different treatment. A low-cost performance estimate can be tentative. A failed numerical or concurrency test cannot be averaged away by a high predicted speedup. Conversely, a timeout without a verdict is incomplete evidence, not proof of incorrectness.

Verification acceleration is a research target in its own right. Cache exact repeated checks using input and tool hashes. Reuse a proof only when its assumptions still apply. Decompose checks only when interfaces and invariants justify composition. Use an early rejection when a sound bound already proves a violation. Never equate “fewer checks” with “faster verification” unless the accepted guarantee is unchanged.

## 9. What verification means at each level

Every result should carry an evidence level rather than the unqualified label “verified.”

| Level | Main question | Evidence and boundary |
| --- | --- | --- |
| Functional/numerical | Does the program compute the intended result to the declared tolerance or quality criterion? | Reference comparisons, property tests, and formal equivalence where applicable; finite testing is not a universal proof |
| Memory/concurrency | Are addresses, lifetimes, ownership and synchronization valid? | Static checks, race analysis or formal properties under an execution model |
| Performance model | Does predicted cost agree sufficiently with a trusted evaluator to guide decisions? | Held-out prediction error and ranking tests; agreement is empirical, not semantic correctness |
| Architecture simulation | Does the executable design obey and perform under the architecture model? | Versioned simulation of full representative workloads; no automatic silicon claim |
| RTL/implementation | Does the register-transfer-level circuit implement the intended architecture and meet physical constraints? | Formal/refinement checks, simulation, synthesis and timing/power analysis with explicit coverage and assumptions |
| Hardware/system | Does the available implementation meet service requirements in practice? | Controlled measurements, variability estimates and full serving tests |

RTL describes the behavior of registers and combinational logic in a hardware implementation. A formal check can establish a specific invariant or equivalence for a bounded or symbolic model; it does not certify every physical or deployment property. Likewise, area and power estimates depend on technology libraries, operating points and implementation assumptions.

A changed component needs an explicit connection to the unchanged parts. For example, moving data into a different memory layout requires matching producers, consumers and address rules. A local kernel check does not establish correct model execution. A hardware simulation does not establish the correctness of a later compiler lowering unless that connection is also checked.

The trusted base consists of declared specifications, checker implementations, reference data, physical libraries and measurement procedures. It can contain defects. Preserve independent checks and adversarial tests rather than presenting the system as universally correct because one evaluator accepts it.

## 10. Intellectual foundations and the capability to test

Existing work supplies important parts of the proposed system:

- **Architecture/mapping representation.** Timeloop separates accelerator architecture, mappings and evaluation, giving a foundation for systematic exploration of hardware/software choices. It supports the importance of explicit representation; it does not establish that our search-policy evolution will help. [Timeloop](https://research.nvidia.com/publication/2019-03_timeloop-systematic-approach-dnn-accelerator-evaluation).
- **Structured optimization.** CoSA formulates accelerator scheduling as constrained optimization. It demonstrates the value of choosing a representation suited to a solver. It is a methodological foundation, not a proof that every inference co-design problem is linear or tractable. [CoSA](https://arxiv.org/abs/2105.01898).
- **Black-box search and resource allocation.** BOHB combines model-based configuration selection with resource allocation across trials. It motivates combining informed proposals with bounded evaluation, while our workload, verification obligations and policy-transfer questions differ. [BOHB](https://proceedings.mlr.press/v80/falkner18a.html).
- **Bottleneck-guided design exploration.** AutoDSE uses bottleneck-guided coordinate optimization for FPGA accelerator design. It is an important comparison against any claim that domain reasoning or feedback-driven search is new by itself. [AutoDSE](https://arxiv.org/abs/2009.14381).

The user-supplied manuscript **LLM4Branch: Large Language Model for Discovering Efficient Branching Policies of Integer Programs**, by Zhinan Hou, Xingchen Li, Yankai Zhang, Tianxun Li and Keyou You (not redistributed here), is particularly relevant to the executable-policy idea. It uses an LLM to propose a branching-policy skeleton and numerical optimization to tune its parameters from solver feedback. Its reported experiment uses eight training instances per benchmark, four for inner tuning, 200 outer iterations and 50 Bayesian-optimization steps in the inner loop. Its principal objective is branching-node count, with a gap-based objective for one difficult benchmark. That objective is not the same as total search wall time. We adopt the separation between program structure and parameters, but do not attribute accelerator co-design or our claimed transfer behavior to that paper.


The proposed contribution is therefore a testable capability: learning and reusing the orchestration of regions, representations, solvers and verification effort across changing inference tasks. It is not novelty by combining tool names. A broader related-work review, including algorithm configuration and automated algorithm design, is required before making a priority or state-of-the-art claim.

## 11. Research questions and decisive experiments

### RQ1: Does adapting the search policy help beyond adapting the cost model?

Freeze a manageable candidate space and the same available evidence. Compare a fixed policy with an evolving policy while giving both the same trainable performance predictor. Then freeze the predictor and repeat. This separates learning where/how to search from merely improving prediction.

The hypothesis survives if policy evolution improves verified progress on held-out episodes at equal total cost. It fails in this setting if gains disappear when candidate spaces, predictor quality and policy-development cost are matched. The first experiment should use a space small enough that a reference frontier can be exhaustively evaluated for research analysis; search policies must not receive that hidden frontier.

### RQ2: Does adaptive decomposition make structured optimization more useful?

Expose several valid decompositions of the same design space: architecture first, workload partition first, and a joint bounded formulation. Keep final candidate expressiveness matched. Measure solve time, feasible proposals, verification cost and admitted design quality.

The hypothesis is that workload-dependent selection of decomposition helps when different interactions dominate. A counterexample is a single fixed decomposition that consistently performs as well after meta-search overhead. Solver difficulty alone is not evidence of design quality, and a larger represented space is not evidence of a better optimizer.

### RQ3: Can evaluation allocation save cost without creating false confidence?

Compare a fixed fidelity schedule, a cost-aware uncertainty policy, and an evolving allocation policy. Include designed cases where cheap models misrank candidates. Measure missed improvements, false promotions, total expensive evaluations and final high-fidelity frontier quality.

Success requires preserved acceptance standards. A method that appears fast because it admits insufficiently checked designs fails this question. If cheap-fidelity ranking is unreliable across the domain, the right result may be a narrow policy that uses it only for sound exclusions.

### RQ4: Does reusable search knowledge help after workload or hardware change?

Train policy versions on one set of workload/hardware families, then freeze them before evaluating new families. Compare transfer against a reset policy with equal per-episode resources, and include the amortized training cost separately.

Test both moderate changes, such as longer prompts or a changed memory/compute ratio, and structural changes, such as expert routing. Transfer succeeds only within the observed range. A policy that helps familiar tasks but suppresses exploration on unfamiliar ones exhibits negative transfer; that result should trigger applicability checks or rollback, not be hidden by an aggregate average.

## 12. Evaluation design: fair budgets, leakage controls and useful curves

Use fixed-domain and open-domain comparisons separately. In fixed-domain comparisons, all methods receive the same mechanisms and legal candidates. Compare a strong fixed hybrid procedure, evolving-policy hybrid, LLM-only proposal selection, structured optimization with a fixed decomposition, Bayesian/stochastic search, and enumeration on small spaces. In open-domain comparisons, allow new mechanisms and representations, but attribute gains to the complete method rather than to its optimizer alone.

Prespecify equal concurrency, wall-time and resource budgets. Report LLM calls and available token/cost accounting, solver compute, simulator/verification compute and human intervention. Match access to historical data and evaluator caches. Include policy-development time in cold-start comparisons. Avoid giving one arm a warm cache or a curated mechanism library unavailable to the others.

Partition tasks into policy-training, validation and final test episodes by workload family and hardware regime, not by randomly splitting nearly identical tile configurations. Validation selects policy versions and hyperparameters; final tests do not. A held-out episode may allow online adaptation using its allotted evaluations, but those observations must not be recycled to redesign the frozen meta-policy and rerun the same test as if it were unseen.

For workload drift, release changes chronologically. At each point the policy sees only prior observations. Report time to recover service requirements, loss during adaptation and the cost of any required hardware replacement. A software-only adjustment on fixed hardware and redesigning the hardware are distinct outcomes. Replay old episodes after updates to detect catastrophic forgetting: improvement on recent tasks that damages previously supported ones.

The principal plot shows best admitted utility or verified Pareto hypervolume against total elapsed search time. Show separate latency/goodput, energy and cost panels so a favorable aggregate cannot conceal a bad tradeoff. Add milestone annotations for changes in design **and** changes in search policy. Plot failed, rejected and unfinished trials separately. A policy is credited only when a result completes the required verification, not when it proposes an appealing design.

Use a second cost axis or companion plot for total resource use. Parallel evaluations consume overlapping wall time but additive compute resources. If runs end without a feasible design, record that the first feasible design had not been found when the budget expired; its discovery time remains unknown; do not silently omit it. Noisy hardware measurements need repeated trials and uncertainty estimates. Deterministic simulator results do not acquire statistical independence through repeated identical runs.

Any roofline or mathematical bound must state its work units, assumptions and scope. A surrogate prediction, a mixed-integer solver's bound for an approximate formulation, and a physical resource ceiling are different objects. Plot them separately. Never manufacture a percentage of roofline to resemble a motivating slide.

## 13. A phased, feasible research program

**Phase A: establish the falsifiable core.** Build a small explicit co-design benchmark with a reference frontier and complete evaluator receipts. Freeze a strong hand-designed hybrid policy. Implement evolving region selection and evaluation allocation with small executable policies. The go/no-go decision is whether held-out verified progress improves once meta-search cost is included. If not, identify whether the policy class, transfer assumption or opportunity itself is inadequate before building a larger platform.

**Phase B: introduce changing workload distributions.** Move to full prefill/decode service traces and a parameterized memory/compute architecture. Add workload drift and online software adaptation. Test applicability detection, reopening of neglected regions and robust design tradeoffs. Preserve one frozen-policy control throughout.

**Phase C: evolve decomposition and solver behavior.** Allow approved formulation templates, solver allocations and branching/neighborhood heuristics. Keep feasibility and bound computation inside trusted backends. Use matched candidate spaces to determine whether search changes, rather than expressiveness changes, explain improvements.

**Phase D: strengthen physical and end-to-end validation.** For a small selected architecture family, connect simulation to synthesizable components, RTL properties, physical estimates and available hardware measurements. This phase depends on suitable implementation and verification infrastructure. It is not required to claim that every arbitrary proposed architecture has been physically realized.

For the pilot, propose three search seeds per method on small benchmark episodes, a fixed two-hour online budget per episode, and a separately reported cap of 24 worker-hours for policy development. These are planning limits to be revised after a cost-only evaluator pilot and frozen before method comparison. Larger studies should use enough independent episodes to report uncertainty across workload families, not merely three executions of the same task. No such experiments are launched by this document.

Stop or narrow a phase when its core comparison shows no benefit at matched cost, when evaluator fidelity is insufficient to decide the hypothesis, or when verification cost exceeds the available resources. Preserve the strongest fixed method as a fallback. A negative result about one policy representation should be distinguished from a claim that all adaptive search is impossible.

## 14. Risks and their experimental controls

| Risk | How it can mislead the study | Control |
| --- | --- | --- |
| Unsafe heuristic pruning | Good unfamiliar regions disappear before evaluation | Separate certified exclusions from postponements; reserve exploration and log reopenings |
| Simulator exploitation | The policy finds model artifacts instead of robust designs | Independent evaluators, held-out discrepancies and stronger validation of finalists |
| False confidence | Cheap agreement is mistaken for physical or semantic proof | Explicit evidence levels, uncertainty calibration and unchanged acceptance rules |
| Search-policy overfitting | The controller memorizes benchmark identities or cached outcomes | Family-level splits, provenance checks and controlled feature access |
| Catastrophic forgetting | Recent gains erase useful earlier behavior | Replay suite, versioned incumbents and rollback |
| Expensive self-improvement | Meta-search costs more than deployment savings | Cold/warm accounting and measured amortization break-even |
| Confounded credit | New candidates, better predictors and better search change together | Matched-domain ablations and separate open-domain reporting |
| Invalid physical freedom | Unrealistic bandwidth or frequency appears optimal | Realizable catalogs, implementation constraints and explicit unresolved parameters |
| Verification shortcuts | A faster curve reflects weaker checking | Frozen acceptance contracts and audits of reused evidence |
| Unpredictable workload change | Robustness is claimed outside the stress set | State distributional limits; test structural shifts and report adaptation failures |

The largest practical risk is building an elaborate controller before demonstrating that a simple adaptive policy is useful. The phase order deliberately tests that assumption first.

## 15. Deliverables and bounded success criteria

The first deliverables are a workload/architecture episode specification, an explicit design representation, a restricted executable search-policy interface, a versioned evidence store, a fixed hybrid baseline, an adaptive policy implementation and reproducible evaluation scripts. Subsequent deliverables add solver-policy extensions and stronger physical-validation paths only when earlier results justify them.

A successful study must show, on untouched episodes, either better verified design quality at the same total cost or lower total cost to reach a prespecified verified target, without weakening correctness or feasibility requirements. It must identify which search-policy change produced the gain through an appropriate ablation. A transfer claim additionally requires improvement on declared new workload or hardware families after accounting for policy-learning cost.

Evidence is insufficient if there is only one hand-selected success, only a better predicted score, only more candidate evaluations, or a speedup that disappears when policy development is charged. Failure includes persistent negative transfer, missed feasible regions caused by uncorrected heuristic pruning, or a need to relax verification to obtain progress.

The intended result is not a universally optimal autonomous chip designer. It is a tested account of when reusable, evolving search procedures help engineers navigate changing inference workloads—and when a fixed solver, a simple enumeration or a conventional experiment remains the better choice.

### Scope of existing local work

Earlier local matrix and homework experiments are optional sandboxes for checking interfaces and timing/provenance conventions. They are not results for this general proposal, and their hardware limits and scores do not define its design space. Existing graders may finish under their original guards; no additional homework round or implementation campaign is authorized by this proposal draft.
