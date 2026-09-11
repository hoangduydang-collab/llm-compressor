# Sep 7 - Sep 11

## Duy

### What I worked on

- **GPQA rerun under the AA harness.** Same two checkpoints as last week, ours
  and PhalaCloud's, scored by NVIDIA's packaged AA clone instead of our own
  protocol.
- First GLM-5.3 number we have in the same units as a published figure.
- **Expert-parallel GPTQ and the serving-format work both moved.** EP GPTQ's
  deciding gate is queued behind a GPU node.

### Key results, GPQA Diamond under AA methodology

**Scores.** 198 Diamond items x 5 repeats = 990 completions per arm. Both arms
990/990 successful, every response HTTP 200.

| Arm | Score (pass@1, avg of 5) | Stderr |
|---|---:|---:|
| **Ours (W4AFP8)** | **81.82%** | ±1.27 |
| PhalaCloud (W4AFP8) | 79.39% | ±1.28 |
| AA published GLM-5.3 (max), reference | ~91.7% | |

- Ours leads by 2.42 points, same direction as last week's cheap suite, now on
  990 completions instead of 198.
- The gap is not statistically resolved: about 1.3x the standard error of the
  difference. Directional, not a proven win.
- Last week's 61.11% / 55.56% pair is not comparable. Single greedy samples, 32K
  output budget, our own extraction.

**Ours is also cheaper to run.** Not the question we asked, but the more
interesting finding.

| | Ours | PhalaCloud |
|---|---:|---:|
| Avg completion tokens | 26,463 | 30,745 |
| Hit the 131,072-token cap | 120 / 990 (12.1%) | 155 / 990 (15.7%) |
| Wall clock, one 8xH100 node | 37.1 h | 46.2 h |

- Ours answers in 16% fewer tokens and truncates less often, so it scores higher
  while generating less.
- Truncation is probably most of the gap to AA's 91.7%. Ours ceilings at 87.9%
  if every length-finish scores zero.
- That last point is unverified, not measured. The harness kept only aggregates,
  no per-item scores.
- The unquantized comparison is still missing. No BF16 GLM-5.3 fits an 8xH100
  node, so a defect shared by both arms stays invisible.

**Provenance.**

- `nvidia-simple-evals==26.3` / `nemo-evaluator` 0.2.8, task
  `gpqa_diamond_aa_v3`, temperature 0.6, top_p 1.0, max_new_tokens 131,072,
  thinking on, concurrency 8.
- Serve identical on both arms: SGLang 0.5.17, tp=8, context 164,800, FP8 KV,
  mem_frac 0.75, no speculative decoding.
- We wrote no prompt and no extraction regex. That is the point of using the
  packaged harness.
- Artifacts:
  `/mnt/cephfs/hoangduy/results/glm53-quality-paired/20260902t0431z/client-{ours,phala}/aa-gpqa-v3/results-formal-198-c8`

### Issue 1, expert-parallel GPTQ (current priority)

**The problem.**

- MiniMax-M3 never needed expert parallelism. It fit on one 8xH100 node with
  data-parallel GPTQ, so the question never came up.
- GLM-5.3 does not fit. Every rank holds a Hessian for every expert in the layer
  it is working on, and the run OOM'd on a full node.
- The earlier EP implementation was left unfinished and the practice run was
  abandoned rather than risked.

**Progress.**

- Expert-local Hessian accumulation and solving implemented, opt-in behind
  `quantization.gptq_expert_parallel`.
- Reuses the fork's existing EP adapter and MoEQuant's expert-local/replicated
  ownership split rather than new math.
- CPU regression passes, 212 tests. Tiny-fixture NCCL correctness and
  save/reload pass.

| Gate | State |
|---|---|
| CPU regression, 212 tests | pass |
| Representative 256-expert EP4/EP8 memory gate | **not run, needs a full node** |
| Full EP8 checkpoint | not run |

- The passing gates prove correctness and round-tripping. They do not prove the
  OOM is fixed. They ran on tiny real-GLM fixtures.
- Hypothesis to test: routed Hessian residency drops from roughly 19 GiB/rank at
  EP4 to 9.5 GiB at EP8, with logits within `rtol=2e-3` of a DDP4 control.
- The eight-GPU chain that measures it is written and submitted. Pod
  `glm53-ep-gptq-20260910t120700z` has been Pending 17 h.
- It is fail-closed, so only a passing representative gate starts the full run.

### Issue 2, produce the serving formats during quantization

**The problem.**

- **SGLang's W4AFP8 path accepts exactly one checkpoint layout, and nothing
  close to it.** Producing a correct 4-bit model is not enough; it has to be
  spelled the way the loader expects or the server will not start.
- That layout is specific down to the tensor: routed experts as INT4 group-128
  nibble pairs in INT8 storage, attention and shared experts as E4M3 with
  128x128 block scales named `weight_scale_inv` rather than `weight_scale`, the
  DSA indexer in FP8, and the MTP layer present.
- The indexer is the sharpest example. `W4AFp8Config.from_config` ignores
  `ignored_layers`, so declaring the indexer BF16 cannot work. The runtime
  builds it quantized regardless.
- Our AWQ output missed several of those, so making it serveable cost a
  whole-checkpoint conversion (2 h 26 m), a 19-shard indexer repatch (32 m) and
  an MTP graft. Roughly 3.5 h of pure post-processing, plus one unloadable
  checkpoint along the way.
- M3 needed the same kind of patching for vLLM, so this is a pattern. W4AFP8 is
  a runtime contract, not a recipe label.

**Progress.**

- Most of what was needed already existed: the FP8 quantizer, repacker, indexer
  repair, MTP graft and verifier were all in the repo. Only the wiring was
  missing.
- Indexer `wk` / `wq_b` now sit in the AWQ recipe's `FP8_BLOCK` pass, so the
  required format is produced during quantization instead of repatched after.
- The converter now preserves native block FP8 instead of rebuilding those
  weights from BF16. The old path reconstructed the AWQ fold and assumed
  per-channel scales.
- Verified on CPU: 116 tests pass, exit 0, with versions and file hashes
  recorded.

**Still open.**

- Direct native export, which would remove the second whole-checkpoint write.
- Expert activation-scale alignment. The MoE path wants static per-group scales;
  our preset is dynamic per-token.
- MTP assembly and qualification against the pinned serving runtime.
- CPU conformance is not a serving qualification.

### What is blocking me: GPU resources

This is the binding constraint right now, not design and not implementation.

- **The EP GPTQ memory gate needs all eight GPUs on one node.** The request is
  atomic and every GPU node has exactly eight, so a partially free node cannot
  take it and the pod pends indefinitely. Pending 17 h so far.
- **Evaluation is expensive at these budgets.** This one GPQA table cost 83 h of
  wall clock on a full node, about 28 GPU-days, because the arms cannot share a
  node and 131K-token thinking traces run 18-22 min per request.
- **Zhou Yu's two serving nodes relieve half of that.** Paired arms can run
  concurrently from next week, so each benchmark costs wall clock once rather
  than twice.
- It does not lift the context ceiling. 164.8K is the measured FP8 KV pool on
  8xH100, and AA-LCR's output-policy parity wants 256K.
- I have a full-node holder designed and approved. It queues for a whole node
  and holds it on a renewable 48-hour lease, which turns "check if a node is
  free" into "be next in line". It does not create capacity.
- Quantization is still the squeezed side. It needs a predictable 8-GPU
  allocation of its own, not one shared with eval work.

### Plan for next week

**Three AA benchmarks, now that Zhou Yu's two serving nodes are available to
us.** All three run on the same open-sourced AA harness we used this week, so
there is no bespoke prompt, extraction or judge work to build.

- **Rerun GPQA Diamond.** Two endpoints means both arms run at once instead of
  back to back, so the 83 h becomes roughly 46 h.
- **AA-LCR v1.1**, 100 questions x 3 repeats. Closes the long-context gap and is
  the first real test of the FP8 DSA indexer.
- **HLE text-only**, 2,158 questions x 1 attempt. Needs the AA equality-checker
  judge wired up; that is the one dependency outside our cluster.

**Everything else.**

- **Run the representative EP GPTQ gate the moment a node frees.** A pass
  unlocks the full EP8 run; a fail is the memory diagnosis we have never
  actually had.
- **Keep issue 2 moving on CPU.** Native export and the activation-scale
  question both advance without a GPU, which is why they go first while the
  cluster is full.
