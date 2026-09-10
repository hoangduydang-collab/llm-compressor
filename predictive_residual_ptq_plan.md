# Anchor-Predictive Residual PTQ with Hadamard Rotation
## Agent implementation and experiment plan

**Version:** 1.1 · **Prepared:** September 10, 2026  
**Status:** Experimental specification, not a validated quantization method. No LLM experiments have been executed as part of preparing this plan.  
**Primary question:** Does a cheap within-group column predictor improve 2–3-bit weight quantization beyond a matched Hadamard-only baseline, after counting all stored information?

---

## 1. Objective, scope, and operating contract

Implement a reproducible, pure post-training experiment for the representation

\[
\widehat W = \widehat A B + L^\top\widehat Z.
\]

Here, one stored anchor column represents each group of input columns; each remaining column has a stored scalar prediction coefficient; and the remaining error is rotated on the **output dimension** before low-bit scalar quantization. The rotation is shared by all groups of a matrix.

The working hypothesis is that subtracting a cheap shared column direction reduces the information that the residual quantizer must represent. Randomized Hadamard rotation may redistribute residual outliers, but it does not reduce residual energy or manufacture column correlation. A negative result on ordinary contiguous groups is a valid and useful outcome.

### Fixed decisions for version 1

- Use **contiguous column groups**, initially `G=64`. No global column reordering or clustering. The primary real-model target is the locally installed **Qwen3-30B-A3B MoE** checkpoint; do not download a substitute model unless explicitly requested.
- Use **one actual column as the anchor**, with a **signed scalar coefficient** for every non-anchor column. No learned basis, multiple references, prediction chain, or cross-expert predictor.
- Store anchors at **4 bits**, coefficients in **FP16**, and residuals at **2, 3, or 4 bits**. The main scientific target is 2–3 bits; 4 bits is a control.
- Apply a normalized, randomized, block-diagonal Hadamard transform on output rows, initially `h=128`. Do not right-multiply the weight matrix.
- Keep activations, attention computation, KV cache, biases, embeddings, normalization parameters, and the output language-model head at their baseline precision in the initial experiment. This is **weight-only PTQ**, not end-to-end W4A4 or W4A8.
- Do not fine-tune model parameters, learn rotations, or use GPTQ compensation inside the proposed method in version 1. A deterministic weight-MSE clipping search is allowed and must be identical across internal baselines.
- Separate **quality**, **serialized size**, and **runtime**. None is a substitute for another.

### Agent instructions

Inspect the existing repository and reuse working infrastructure before creating replacements. Prefer established Hadamard kernels, model loading, and evaluation code. Keep the experiment independent of the particular agent framework. Pin package versions and source commits after a smoke test; do not assume that a repository's current default branch reproduces an older paper.

Run deterministic correctness gates before real-model sweeps. Implement resume, failure logging, and bounded sweeps. Do not delete original checkpoints or modify a shared Python environment. Never fabricate unavailable measurements, silently change precision or matrix orientation, or describe a BF16 reconstruction as a packed low-bit kernel.

The end product is an evidence-backed **go / revise / stop** decision, together with the implementation and reproducible artifacts—not a requirement to obtain a positive result.

## 2. What to reuse, and what is actually new in this experiment

QuaRot provides the rotation motivation and an official implementation. QuIP# provides a related randomized-Hadamard quantization reference, but its full algorithm also uses vector codebooks and fine-tuning. Our simple rotation-only baseline is **not** a reproduction of either complete method. [S1–S3]

Reuse `Dao-AILab/fast-hadamard-transform` for the accelerated transform after validating a small reference implementation. Its API applies the transform on the last tensor dimension and accepts a scale argument; its documented implicit padding must not be relied upon for the invertible codec described below. [S4]

Reuse Transformers/Safetensors for model inspection and loading and the LM Evaluation Harness for downstream evaluation. LLM Compressor already exposes rotation-related recipes and is a useful baseline/integration reference. Do not implement an unrelated compression framework around the prototype. [S5–S7]

Low-rank-plus-quantized representations have prior art, including LQER. The present anchor representation is itself a restricted, block-structured low-rank predictor plus quantized residual; do not claim that the general decomposition is new. [S8]

## 3. Exact mathematical specification

### 3.1 Matrix convention and groups

Use

\[
W\in\mathbb R^{m\times n},\qquad X\in\mathbb R^{n\times T},\qquad Y=WX+b.
\]

This corresponds to a PyTorch linear weight of shape `[out_features, in_features]`. A usual PyTorch input is `[tokens, in_features]`, so its computation is `X_row @ W.T + bias`.

Partition the input columns into consecutive groups `I_g` of size at most `G`. Let `K=ceil(n/G)` be the group count. Each group has one anchor index `a_g`. The last group may be shorter; do not discard it or count fictitious weights.

### 3.2 Closed-loop encoding: use what the decoder actually has

First encode and decode the anchor:

\[
\widehat a_g=\operatorname{DQ}_4(\operatorname{Q}_4(W_{:,a_g})).
\]

For a non-anchor column `j` in group `g`, fit

\[
\alpha_j^*=\frac{\widehat a_g^\top W_{:,j}}{\widehat a_g^\top\widehat a_g},
\qquad
\widehat\alpha_j=\operatorname{FP16}(\alpha_j^*).
\]

The actual encoded residual is

\[
\boxed{r_j=W_{:,j}-\widehat a_g\widehat\alpha_j.}
\]

**Both the anchor and the coefficient must already have their stored precision before computing this residual.** Computing it against an original FP32 anchor or an unrounded coefficient would give an encoder/decoder mismatch.

Encode the transformed residual:

\[
z_j=Lr_j,\qquad \widehat z_j=\operatorname{DQ}_b(\operatorname{Q}_b(z_j)).
\]

Decode a non-anchor column as

\[
\boxed{\widehat W_{:,j}=\widehat a_g\widehat\alpha_j+L^\top\widehat z_j.}
\]

For the anchor column itself, decode `W_hat[:, a_g] = a_hat_g`. Its coefficient is implicitly 1 and its residual is implicitly zero. **Do not store an anchor coefficient or anchor residual.** Anchor quantization error remains and must be measured; it is not corrected by a hidden residual.

For non-anchor columns, in exact arithmetic,

\[
\widehat W_{:,j}-W_{:,j}=L^\top(\widehat z_j-z_j).
\]

Thus anchor/coefficient approximation is incorporated into the residual instead of being added as an unaccounted reconstruction error. A poorly stored predictor can still enlarge the residual and make quantization harder.

### 3.3 Full representation and structured inference identity

Collect decoded anchors as `A_hat` of shape `[m, K]`. Define `B` conceptually as a `[K, n]` matrix: each input column has only one nonzero entry, belonging to its group; that entry is its stored coefficient, or 1 for an anchor column.

Let `Z_hat` conceptually have shape `[m, n]`, with zero columns at anchors and stored decoded transformed residuals elsewhere. Then

\[
\widehat W=\widehat A B+L^\top\widehat Z,
\]

and

\[
\boxed{\widehat W X=\widehat A(BX)+L^\top(\widehat Z X).}
\]

Do not allocate dense `B` in production code. `BX` is a groupwise weighted reduction of activations. The residual multiplication can gather the non-anchor input columns and use the compact residual matrix.

This identity supplies a possible execution path without permanently reconstructing all weights. It is **not** evidence of a speedup. The additional reduction, anchor product, inverse transform, packing format, and memory traffic must be measured separately.

### 3.4 Rotation definition, normalization, and inverse

For each matrix, define

\[
L=\operatorname{blockdiag}(H_{h_1},\ldots,H_{h_q})D,
\]

where each `H_h` is a **normalized** Sylvester Hadamard matrix, `H_h.T @ H_h = I`, and `D` is a diagonal matrix of deterministic random signs. Default block sizes are 128; split any remainder into power-of-two blocks, for example `96 = 64 + 32`.

This makes `L` square and orthogonal for every positive output dimension without padding the representation. The same block partition and sign vector must be used for all column groups and matched baselines of that matrix and seed.

Forward and inverse operations are different:

```text
forward(v): signs first, then normalized block-Hadamard
inverse(v): normalized block-Hadamard first, then signs
```

Although an ordinary normalized Hadamard is self-inverse, `H @ D` is generally **not** self-inverse. The inverse is `D @ H.T`.

Store the actual sign vector, bit-packed, rather than depending solely on a library RNG reproducing the same sequence on another platform. Record the seed as provenance too.

The kernel operates on the last dimension. To transform output rows of a matrix, reshape/transpose into `[columns, output_blocks, h]`, transform the last axis, then restore `[m, columns]`. Never accidentally transform the input-column axis. Use normalization `scale=1/sqrt(h)` for a kernel returning an unnormalized Hadamard product. [S4]

The following hold before quantization:

\[
\|LR\|_F=\|R\|_F,\qquad (LW)^\top(LW)=W^\top W.
\]

Rotation can change coordinate-wise extremes and scalar quantization error; it cannot improve the original column angles or reduce residual L2 norm. Measure the benefit rather than assuming every rotation helps.

The inverse output transform must occur **before** the layer's original bias/nonlinearity consumes the result. Do not push a transform through SwiGLU, attention, normalization, or residual additions without a separate invariance proof. Version 1 makes no whole-network QuaRot rewrite.

## 4. Anchor selection: projection error, not signed cosine medoids

The predictor permits negative coefficients. If two columns are negatives of one another, they are perfectly predictable—not dissimilar for this purpose. Consequently, maximizing a sum of signed cosine similarities is the wrong anchor-selection objective.

Use **decoded-predictor reconstruction SSE** as the primary objective. For each candidate anchor `i` in a group:

1. Quantize/dequantize candidate `W[:, i]` using the same INT4 anchor codec used at deployment, giving `c_i`.
2. Fit every non-anchor coefficient against `c_i`, then cast it to FP16.
3. Compute

\[
J_i=\|W_{:,i}-c_i\|_2^2+
\sum_{j\in I_g\setminus\{i\}}
\|W_{:,j}-c_i\widehat\alpha_{ij}\|_2^2.
\]

4. Choose the lowest-scoring candidate, breaking ties by the smallest original column index.

This objective includes anchor error and reflects the stored predictor, but does not repeatedly quantize every candidate's residual. Final residual quantization happens only after selecting the anchor. Anchor selection is independent of residual bit width and Hadamard seed, allowing a consistent anchor control.

Efficient implementation: quantize candidate columns in a batched operation; compute `C.T @ W_group` and column norms; evaluate candidate SSEs from these terms. Confirm the winning candidate with an explicit residual calculation to avoid cancellation artifacts. Complexity is approximately `O(m G^2)` per group, or `O(m n G)` per matrix. Avoid a global `n x n` similarity matrix.

As a diagnostic only, the unquantized projection score is

\[
\sum_j\frac{(W_{:,i}^\top W_{:,j})^2}{\|W_{:,i}\|_2^2}.
\]

Squared correlation, including negative correlation, is what matters.

Handle zero decoded anchors explicitly: define their non-anchor coefficients as zero, rather than dividing by zero. If casting a coefficient produces a non-finite value, invalidate that candidate and log the event. For an all-zero group, store a zero anchor and zero residuals under the normal format. Do not silently change to higher precision.

## 5. Scalar quantization and storage format

### 5.1 Make “2-bit” unambiguous

Use a transparent, fixed-width signed-integer reference quantizer:

\[
q_{\min}=-2^{b-1},\qquad q_{\max}=2^{b-1}-1,
\]

\[
q=\operatorname{clip}(\operatorname{round}(v/s),q_{\min},q_{\max}),
\qquad \widehat v=sq.
\]

For 2 bits this means **four levels** `[-2, -1, 0, 1]`, not a hidden ternary quantizer `[-1, 0, 1]`. Record the exact rounding convention. The signed range is asymmetric; do not describe it as a balanced four-level codebook.

For each quantization group, derive the initial unclipped scale

\[
s_0=\max\left(\frac{\max(0,\max v)}{q_{\max}},
\frac{\max(0,-\min v)}{-q_{\min}}\right).
\]

Evaluate 41 clipping multipliers uniformly spaced from 0.20 to 1.00. For each candidate, cast the scale to its storage dtype **before** computing integer codes and decoded MSE. Choose the scale with the smallest actual decoded weight MSE. All internal baselines use the identical procedure and search budget.

Default scale storage is FP16. All-zero groups use scale zero and integer code zero. Reject and report nonzero groups for which stored scales become zero, non-finite, or otherwise invalid; a separately labeled FP32-scale fallback is permitted only if its bytes are included. Never silently alter the format to make a run pass.

Before declaring a positive result, rerun the promoted 2-bit comparison with a balanced, full-four-level uniform codebook as a sensitivity check. For example, decoded levels may be `s * [-1, -1/3, 1/3, 1]`, with a zero-scale exception for an all-zero group. Label this codebook separately; do not call it an ordinary signed INT2 GEMM representation or pool it with integer-kernel results.

### 5.2 Three different grouping parameters

Keep these independent in code and in reports:

| Parameter | Meaning | Initial value |
|---|---|---:|
| `G` | Input-column predictor group size | 64 |
| `S` | Number of values sharing a residual/ordinary quantization scale | 128 |
| `h` | Output-row Hadamard block size | 128 |

Anchor scale groups use `S_a=128` consecutive **output elements within each anchor column**. Quantization scales for ordinary weights or compact residuals are per output row over `S` consecutive stored input-column positions.

Concatenate non-anchor residual columns in increasing original-column order before scalar grouping. The compact residual matrix has shape `[m, n-K]`. Scale groups may cross predictor-group boundaries. Do not accidentally restart residual scale groups at every predictor group, which would change the rate and potentially manufacture a comparison advantage.

Mask a final partial scale group when fitting scales. Do not include padded values in its MSE or count them as original parameters. If a kernel layout later requires padding, charge the padded bytes separately.

### 5.3 Packed payload schema

Store a small versioned JSON manifest plus packed numeric tensors. Safetensors can hold uint8 bitstreams and ordinary metadata tensors; it does not itself pack INT2/INT3 values. [S7]

Required fields:

```text
format_version
original_shape: [m, n]
original_weight_dtype
predictor_group_size, group_lengths
anchor_local_indices              # uint8 for G <= 256
anchor_integer_bitstream          # 4 bits per anchor element
anchor_scales                    # FP16, explicitly shaped
coefficients                     # FP16, non-anchor columns only
residual_integer_bitstream        # b bits per residual element
residual_scales                   # FP16, explicitly shaped
rotation_block_sizes
rotation_sign_bitstream
residual_bits, integer_code_range, rounding_mode
scale_group_sizes, scale_dtypes, packing_order
seed, model_revision, module_name, codec_config_hash
```

Define a simple reference packing order: row-major logical values, convert signed codes to unsigned `u=q-q_min`, then pack little-endian bits into a continuous uint8 stream. Record logical lengths; leave final unused bits zero. For anchors, specify their matrix orientation explicitly in the manifest. Round-trip decode must not need the original model weights.

Test 3-bit codes that cross byte and word boundaries. An `int8` tensor whose entries happen to lie in a 2-bit range is **not** a 2-bit serialized payload.

Do not add entropy coding in this experiment. The scientific question is whether a usable fixed-width representation gains quality per stored bit.

## 6. Baselines and controls

Use the same source weights, target modules, scalar quantizer, clipping search, scale precision, and paired rotation seeds.

| ID | Representation | Purpose |
|---|---|---|
| `BF16` | Original checkpoint precision | Model quality reference |
| `RAW` | Direct `DQ_b(Q_b(W))` | Ordinary scalar PTQ |
| `ROT` | `L.T @ DQ_b(Q_b(L @ W))` | Rotation without prediction |
| `ANCHOR_RAW` | Same selected INT4 anchors; direct b-bit non-anchor weights | Control for protecting anchors |
| `ANCHOR_ROT` | Same selected INT4 anchors; rotated b-bit non-anchor weights | **Primary control for the proposed method** |
| `PRED` | Stored anchor + coefficient + unrotated quantized residual | Prediction without rotation |
| `PRED_ROT` | Stored anchor + coefficient + rotated quantized residual | Proposed method |

`ANCHOR_ROT` uses the exact same anchors selected for `PRED_ROT`, the same compact non-anchor layout, and the same rotation and scale-group definitions. It omits prediction coefficients because it does not need them; report that small byte difference honestly. This prevents “we protected a few important columns at 4 bits” from being mistaken for a predictive-coding gain.

The principal comparisons are:

\[
\texttt{PRED\_ROT} \;\text{vs}\; \texttt{ANCHOR\_ROT}
\quad\text{and}\quad
\texttt{PRED} \;\text{vs}\; \texttt{ANCHOR\_RAW}.
\]

Also compare the full rate–distortion curve against `RAW` and `ROT`. Winning only against unrotated naive quantization is insufficient.

Bounded diagnostic variants: fixed coefficient `alpha=1`, FP16 anchors with their full storage cost, and an unquantized best rank-1 SVD approximation as a **geometry oracle**, not an implementable rate-matched baseline. Run these only to explain a result, not as an unbounded search for a win.

Once an internal signal survives, add an established GPTQ baseline through a maintained, compatible implementation. The original repository supplies algorithm and low-bit references; its model scripts are not automatically compatible with every modern architecture. Match calibration, target modules, and reported bit accounting. [S9]

A full QuIP# comparison can be added for a broader research claim, but document codebooks and fine-tuning. Do not label our `ROT` baseline “QuIP#” or compare a data-free method and a tuned method as though their optimization budgets were identical. [S3]

## 7. Honest bit accounting

Let `K=ceil(n/G)`, `n_r=n-K`, anchor bits `b_a=4`, and coefficient bits `b_alpha=16`. Ignoring byte alignment and manifest text for the moment, the proposed payload is

\[
\begin{aligned}
B={}&bmn_r+b_amK+16m\lceil n_r/S\rceil\\
&+16K\lceil m/S_a\rceil+b_\alpha n_r+8K+m.
\end{aligned}
\]

The last two terms assume uint8 anchor indices and one packed sign bit per output coordinate. Use actual group lengths and metadata dtypes where these assumptions do not hold. Rotation block descriptors and format metadata also consume space.

Report all of:

1. **Logical payload bpw:** `8 * actual_tensor_payload_bytes / (m*n)`.
2. **Serialized bpw:** `8 * total_codec_file_bytes / (m*n)`, including headers and alignment.
3. **Whole-model bpw:** include unquantized parameters and avoid double-counting tied tensors.
4. **Runtime memory:** measured resident memory and temporary workspace, with the execution mode stated.

An illustrative 4096-by-4096 matrix with `G=64`, `S=S_a=128`, 4-bit anchors, 2-bit residuals, FP16 coefficients/scales, uint8 anchor indices, and packed signs uses approximately **2.1623 payload bpw before headers/alignment**. Ordinary 2-bit weights with FP16 scales every 128 values use **2.125 bpw**. The proposed representation is therefore not “exactly 2 bits per weight,” and the comparison is not automatically equal-rate.

Use matched-rate controls within a preregistered 1% relative payload tolerance where possible, and always publish rate–distortion plots. A more expensive representation must deliver enough quality improvement to justify its bytes. If the baseline has fewer bytes, report that rather than padding it with meaningless metadata. `ANCHOR_ROT` will generally be a particularly close rate match, but verify rather than assume this for every shape.

An FP16-anchor diagnostic can reveal predictability but is not evidence that the INT4-anchor deployment format works. A 4-bit-residual version with metadata can exceed four effective bits per weight.

## 8. Resources, model selection, and data

### 8.1 Environment and primary model: local Qwen3-30B-A3B

Use the **already installed local Qwen3-30B-A3B checkpoint** as the only required real-model dependency for version 1. Do not download Qwen2.5 or another smoke/confirmation model merely to follow the old ladder. Synthetic tensors are sufficient for codec correctness before touching the checkpoint.

At Stage 0, locate the local checkpoint, resolve its exact path and variant, and pin a content identity using its `config.json`, tokenizer files, safetensor index/filenames, and hashes or immutable repository revision if available. Record whether the installed checkpoint is `Qwen3-30B-A3B`, `Qwen3-30B-A3B-Base`, an FP8 derivative, or another variant. The primary experiment requires original BF16/FP16-quality weights for PTQ; if the only local checkpoint is already quantized, stop and report that instead of treating it as the original baseline.

The published Qwen3-30B-A3B architecture has 48 decoder layers, 128 experts per MoE layer, 8 experts selected per token, hidden size 2048, and MoE intermediate size 768. [S10, S11] Under these dimensions, each expert contains three logical MLP matrices (`gate`, `up`, `down`) with about 4.72M weights total, and all expert MLP tensors across 48 layers contain about 29.0B scalar weights. This makes **MoE experts the primary quantization target**, not a later extension.

Treat expert tensors as logical 2D PyTorch linear weights `[out_features, in_features]` before applying the codec:

- `gate_proj`: `[768, 2048]`
- `up_proj`: `[768, 2048]`
- `down_proj`: `[2048, 768]`

Depending on the Transformers/checkpoint version, experts may appear as per-expert tensors or fused 3D tensors such as `gate_up_proj[num_experts, 2*moe_intermediate, hidden]` and `down_proj[num_experts, hidden, moe_intermediate]`. Current Transformers code supports a fused 3D expert representation. [S11] Build a small adapter that exposes each expert's logical `gate`, `up`, and `down` matrices without copying the entire expert bank. If `gate` and `up` are fused, split them before scientific reporting; do not let the output-row Hadamard silently mix the two SwiGLU branches in version 1.

Start with one GPU if the exact checkpoint and working precision fit with safe scratch headroom; otherwise shard model evaluation across the minimum number of GPUs needed on one node. Matrix-level screening should load/process one tensor or expert at a time and does not require the full model resident on GPU. Never require all GPUs merely because they are available.

Record GPU name, UUID, memory, compute capability, MIG status, driver, CUDA, PyTorch, Transformers, Hadamard extension, evaluation harness, OS, CPU/RAM, git commit, and package lockfile. Keep different GPU types/SKUs in separate runtime archives. Re-run performance baselines on every allocation; do not mix H100 and A100 timings.

For large tensor banks, keep original weights on CPU/checkpoint storage and use FP32 scratch on GPU. Accumulate anchor-selection dot products in output-row chunks if needed, then make a second pass to encode residuals. Do not select anchors from an arbitrary row slice and present them as full-column anchors.

### 8.2 Data separation

For Qwen3-30B-A3B experiments, use the checkpoint's fixed tokenizer and a pinned `Salesforce/wikitext` dataset revision with configuration `wikitext-2-raw-v1`. [S14] Proposed partitions are:

- **Calibration:** 128 deterministic windows of up to 2048 tokens from the training split. Used for external calibrated baselines and optional later calibration-only diagnostics.
- **Development:** 32 fixed validation windows of up to 2048 tokens. Used for activation-based screening and hyperparameter selection.
- **Final evaluation:** the full test token stream, accessed only after configuration selection is frozen.

If a split cannot supply the prescribed number of non-overlapping windows, use all available windows and record the true count; do not sample duplicates without reporting it. Save token IDs, sampling offsets, special-token policy, and hashes. Initial predictor fitting is weight-only, so it does not consume calibration activations to select anchors or coefficients.

This is a controlled first experiment, not a claim of broad-domain quality. A second domain/downstream suite is required before generalizing a positive result.

## 9. Staged execution plan

### Stage 0 — Inventory and deployment feasibility audit

Inspect existing utilities, model checkpoints, hardware, and available storage. Create the experiment manifest and a smoke-tested dependency lock. Produce `integration_audit.md` before the expensive sweep.

The audit must distinguish supported standard quantizers from this custom format. Current vLLM documentation provides out-of-tree quantization registration hooks, but those hooks do not supply an anchor/residual kernel automatically. Record the required loader, linear-layer implementation, and eventual MoE/parallelism changes. [S12]

A full plugin or fast kernel is not a prerequisite for the first quality experiment. The audit should establish the gap early, not consume the entire experiment budget trying to close it.

**Exit condition:** immutable model/config identifiers, resource checks, clear module orientation, and a documented research-versus-deployment boundary.

### Stage 1 — Reference codec and synthetic correctness tests

Implement the mathematical codec, packing, decoder, and accounting before downloading or evaluating a large model. Include tests for:

- Hadamard normalization, inverse, norm preservation, and preservation of column Gram matrices.
- Identity with bypassed residual quantization; non-anchor columns must reconstruct up to arithmetic error, while anchors retain their intentional storage error.
- The non-anchor error identity from Section 3.2.
- Equality of reconstructed dense multiplication and `A_hat @ (B @ X) + L.T @ (Z_hat @ X)`, including bias placement.
- Exact positive/negative collinear columns, a rank-1-plus-noise positive control, independent Gaussian columns as a weak-predictability control, and residuals with injected coordinate outliers.
- Zero matrices, zero anchors, tiny values, negative coefficients, non-contiguous tensors, non-power-of-two row counts, short predictor groups, and short scale groups.
- Integer level coverage, clipping, stored-scale rounding, all 2-/3-/4-bit pack/unpack values, crossing-byte boundaries, and malformed metadata rejection.
- Fresh-process encode/save/load/decode equivalence without original weights.
- Exact agreement between serialized tensor lengths and the bit-accounting function.

Use FP64 small-array reference tests with tight tolerances, then FP32 tests with dimension-aware tolerances. Disable reduced-precision FP32 matmul modes such as TF32 for reference metrics and record the arithmetic policy; runtime benchmarks may use their own explicitly documented settings. Separately report expected BF16 rounding differences; do not loosen every test until a wrong transform passes. Run the optimized Hadamard extension against the reference before using it.

**Exit condition:** all algebraic and storage tests pass, including an intentional wrong-axis and wrong-inverse test that the suite catches.

### Stage 2 — Qwen3 MoE weight-only geometry and quantization screening

Start with `G=64`, `h=128`, `S=128`, 4-bit anchors, FP16 coefficients/scales, residual bits `{2,3,4}`, and seed 0.

Use a deterministic pilot panel from the installed Qwen3-30B-A3B checkpoint:

- **Layers:** first, middle, and last MoE layers (resolve to actual indices; normally 0, 23/24, and 47 for 48 layers).
- **Experts per layer:** 8 fixed, evenly spread expert IDs, e.g. `{0,16,32,48,64,80,96,112}` when there are 128 experts.
- **Matrices per expert:** logical `gate_proj`, `up_proj`, and `down_proj` separately.
- **Attention controls:** `q_proj` and `o_proj` from the same three layers are useful secondary controls, but expert matrices are primary.

This gives roughly 72 expert matrices in the first pilot, enough to estimate whether the predictor works consistently across expert identity, layer depth, and projection orientation without immediately sweeping all 6,144 expert instances.

For every matrix and predictor group, measure prediction geometry **before** expensive model evaluation. Preserve original output dimension; correlation depends on it. Record layer index, expert ID, projection role, shape, and whether the source tensor was fused or per-expert.

Run all internal baselines and record error and actual bits. Use FP32 codec calculations and metrics. Do not choose seeds from results: use one paired seed for the pilot, then paired seeds `{0,1,2}` for confirmation.

In addition to aggregate results, report distributions across experts. A method that wins only on a few unusually correlated experts should not be promoted as a general MoE codec. Compare `gate/up` against `down` separately because their input dimensions and activation distributions differ.

An unstructured random matrix should not routinely look highly compressible by one anchor. For independent isotropic vectors and a fixed independent reference, the expected explained energy fraction is approximately `1/m`; choosing the best of several references can increase this, so use it as a sanity scale rather than an exact threshold.

**Exit condition:** a complete geometry/error/size table for the pilot expert panel, including negative results and the protected-anchor controls.

### Stage 3 — Routing-aware activation-weighted screening

Run the original-precision Qwen3-30B-A3B model on the **development** set and collect both router decisions and the actual inputs seen by expert projections. Keep the router itself unchanged and in baseline precision.

For each sampled MoE layer, record per-expert token counts and routing-weight statistics. Keep the fixed expert panel from Stage 2 for paired comparison, and optionally add a small **routing-stratified diagnostic panel** (for example high-traffic, median-traffic, and low-but-nonzero-traffic experts) selected using development data only. Do not replace the fixed panel after seeing quantization results.

Hook the actual input to each logical projection:

- `gate_proj` and `up_proj` receive the expert's routed hidden states.
- `down_proj` receives the post-SiLU gated intermediate activation, so it requires a separate capture point.

Use a deterministic reservoir of up to 4096 routed token vectors **per sampled expert/projection** where available. For rarely used experts, report the actual sample count rather than duplicating tokens. An expert with zero development traffic can still contribute weight-only metrics but has no activation-weighted result.

For each candidate matrix approximation, compute

\[
E_W=rac{\|W-\widehat W\|_F^2}{\|W\|_F^2},
\]

\[
E_X=rac{\|(W-\widehat W)X\|_F^2}{\|WX\|_F^2}.
\]

Use the original `W` and complete original-column ordering. Accumulate squared norms across token chunks in a stable precision. For a zero denominator, report absolute error plus an explicit flag rather than returning a fabricated ratio.

Evaluate the **full matrix** output error, not just independent groups: cross-group error terms can matter. A diagonal input-variance proxy may be logged as a cheap diagnostic but cannot replace `||(W-W_hat)X||`.

Report both unweighted-across-expert summaries and **routing-weighted summaries** so highly used experts do not disappear in an average over 128 experts. Do not use router traffic to change the weight precision in version 1; it is an analysis variable only.

These layerwise activations come from the original model and do not capture accumulated error from quantized upstream experts. Label them as local screening results; the next stage addresses propagation.

**Exit condition:** paired activation-weighted results across expert projection families and Hadamard seeds, with expert traffic/coverage reported explicitly.

### Stage 4 — Bounded refinement, then Qwen3 end-to-end MoE evaluation

Do not immediately run the entire Cartesian product of all hyperparameters. Use this sequence:

1. Confirm the default configuration over paired rotation seeds on the fixed expert pilot.
2. On development data only, vary `G` over `{32,64,128}` with other settings fixed.
3. For at most two surviving `G` choices, vary `h` over `{64,128,256}`.
4. Vary scale group size over `{64,128,256}` only to examine the rate–distortion frontier or resolve a clear scale-granularity issue.
5. Apply the alternate full-level scalar codebook check to the promoted 2-bit pair.
6. Confirm the selected configuration on **held-out layer/expert combinations** from the same Qwen3 checkpoint that were not used for hyperparameter selection.

Do not automatically add reordering, multiple anchors, learned transforms, cross-expert prediction, or a new predictor. Those are follow-on hypotheses.

For the first end-to-end quality test, quantize **all MoE expert gate/up/down weights** using the frozen configuration while keeping attention, router, embeddings, norms, and `lm_head` unchanged. This scope targets the dominant model storage while isolating the expert codec. Only after that survives should a second scope add supported attention linears.

For accuracy evaluation, decode the custom payload back into the checkpoint's working precision and replace expert weights. This is **accuracy emulation**, not a memory/speed benchmark. Load each candidate from the original checkpoint or verify complete restoration between trials; never quantize an already quantized model accidentally. Preserve router weights and behavior exactly.

Run original precision, the matched rotation control, `PRED`, and `PRED_ROT`, with additional controls as needed. Check logit finiteness, router-output consistency before expert computation, and generation sanity, then evaluate perplexity on development data. Freeze final choices before accessing the test split.

For perplexity, use a fixed context length of 2048 and stride 512. Score each target token once, mask overlap/context-only labels correctly, respect the model's next-token label shift, and compute total NLL divided by the number of scored tokens—not an unweighted average of batch losses. Save both NLL and perplexity. Sliding-window choices and tokenization materially change perplexity, so reproduce these settings across every candidate. [S13]

Use the LM Evaluation Harness for a frozen, small zero-shot suite such as HellaSwag, PIQA, and ARC-Easy after the perplexity gate. Record task revisions, prompt settings, few-shot count, sample counts, and exact score field (`acc` versus normalized accuracy). A capped development subset must be labeled as a subset; it is not a published full-benchmark score. [S6]

**Exit condition:** end-to-end Qwen3 quality and complete size results for the experts-only scope from a frozen protocol, or a documented rejection explaining where local gains failed to transfer.

### Stage 5 — Expand expert coverage and test attention only after expert success

Because Qwen3-30B-A3B already contains thousands of expert instances, use **held-out experts and layers within this same checkpoint** as the primary generalization test instead of requiring a second model.

Recommended confirmation sequence:

1. Evaluate the frozen codec on at least four held-out layer depths not used in the first/middle/last pilot.
2. Within each held-out layer, use fixed expert IDs disjoint from the pilot IDs where practical.
3. Add routing-stratified experts as a secondary analysis to check whether predictor quality correlates with expert traffic.
4. If the experts-only full-model evaluation succeeds, expand weight-only screening to a substantially larger expert sample before investing in a kernel.
5. Only then evaluate attention projections under a separate codec scope; do not infer that expert results automatically transfer to attention.

The version-1 method remains **within-expert column prediction**. Do not introduce cross-expert shared bases/deltas in the same experiment, even though Qwen3's repeated expert structure makes that an obvious follow-on study.

If a second checkpoint is already locally available at zero additional setup cost, it may be used as an optional external confirmation, but version 1 must not depend on downloading one.

**Exit condition:** evidence that the selected codec generalizes across held-out experts/layers of Qwen3, or a specific failure mode that motivates the next predictor design.

### Stage 6 — Runtime experiment, only for a surviving representation

First implement a structured floating-point reference forward using the stored representation:

```text
u_g = sum_j_in_group(alpha_j * x_j)    # alpha_anchor = 1
anchor_output = A_hat @ u
residual_output = Z_hat_compact @ x_nonanchor
output = anchor_output + inverse_rotation(residual_output) + bias
```

This validates the execution structure but is still not a packed-kernel speed result if `Z_hat_compact` is stored densely at runtime.

Then assess whether existing low-bit kernels can consume the compact residual layout, scales, and anchor format. A custom adapter or fused kernel may be required. Never claim that arbitrary 2-/3-bit packed values automatically map to an accelerated Tensor Core format.

Benchmark decode-like token counts `{1,8,32}` and prefill-like counts `{128,512,2048}` for real projection shapes. Compare original BF16, a strong ordinary low-bit kernel, rotation-only execution, and the structured predictor. Include unpack/dequantization, activation gathers/reductions, inverse transforms, launch overhead, workspace, and output writes. Distinguish cold load/packing time from steady-state inference.

Warm up, synchronize CUDA, use CUDA events for device timing, repeat independent measurement groups, and report medians plus variability. Record allocated and reserved memory, persistent buffers, and peak workspace. Do not benchmark other candidates concurrently on the same GPU or equate torch dense-code timing with future kernel performance.

A shared `L` permits one inverse transform after the summed residual product. Different transforms per predictor group would invalidate that simplification and are outside version 1. Tensor/expert parallel deployment must respect both predictor-group and output-Hadamard-block boundaries; document any needed communication before a vLLM integration claim.

**Exit condition:** a measured runtime result or a specific integration blocker—not a speculative speedup estimate.

## 10. Metrics and decision rules

### Geometry metrics

Record full-group and **non-anchor-only** residual energy fractions. Excluding anchors prevents the trivial exact prediction of one column per group from inflating the apparent benefit. Separate ideal original-anchor projection from the actual decoded-anchor/FP16-coefficient residual.

Report explained energy, coefficient magnitude quantiles, negative coefficient fraction, selected-anchor norms, residual RMS, residual max/RMS, per-scale-group maxima, clipping fraction, zero-code fraction, and actual quantization error. Log selected groups' singular-value spectra or leading rank-1 energy fraction as diagnostics. An SVD oracle has neither the anchor constraint nor the deployment bit budget.

### Primary comparison metrics

For each paired matrix/configuration/seed, compute

\[
r_X=E_X(\texttt{PRED\_ROT})/E_X(\texttt{ANCHOR\_ROT}),
\qquad
r_W=E_W(\texttt{PRED\_ROT})/E_W(\texttt{ANCHOR\_ROT}).
\]

Report each layer, projection family, median/geometric-mean paired ratio, fraction of wins, and parameter-weighted aggregate error. Do not average ratios with an effectively zero baseline error without flagging them. Summaries must not hide a regression in one projection family.

### Preregistered engineering gates—not statistical laws

**Advance to full Qwen3 experts-only end-to-end evaluation** when the main representation provides roughly 10% or greater median development output-error reduction (`median r_X <= 0.90`), wins on at least two-thirds of sampled matrices, and has no projection family with median regression beyond 10%, at matched payload within 1% where feasible. Confirm across three paired seeds. A strong measured rate–distortion improvement outside the 1% tolerance can also justify promotion, but state the different rate explicitly.

**Revise or stop the current contiguous-group version** when gains are within about 2% of the rotation control, vanish after stored-anchor/coefficient accounting, appear only with FP16 anchors whose cost dominates, or require systematically more bytes without a clear quality tradeoff. Weak local geometry alone is a warning, not a proof: use the actual quantization/output errors to decide.

**Advance to broad expert coverage and kernel work** only when the frozen Qwen3 experts-only evaluation has a repeatable improvement over the matched control and no meaningful downstream regression. Suggested screening tolerance is at most one absolute percentage point degradation in the aggregate downstream score, interpreted with uncertainty rather than treated as proof of equivalence. Inspect per-task changes too.

Use paired sequence-level NLL differences and bootstrap intervals over evaluation windows/documents for perplexity comparisons; do not treat every adjacent token as an independent replicate. Bootstrap by layer/block for aggregate local-error comparisons rather than presenting thousands of correlated weight entries as independent trials. Rotation-seed variability is a different source of uncertainty and should be reported separately.

Do not set an arbitrary “near-lossless” label based only on lower weight MSE. End-to-end accuracy, rate, scope of quantized parameters, and uncertainty must support any such statement.

## 11. Suggested repository structure and interfaces

This is a proposed implementation layout, not a claim that these commands already exist.

```text
predictive_ptq/
  README.md
  pyproject.toml
  configs/
    smoke.yaml
    screen.yaml
    full_model.yaml
  src/predictive_ptq/
    config.py
    inventory.py
    model_io.py
    grouping.py
    anchors.py
    transforms.py
    scalar_quant.py
    codec.py
    packing.py
    accounting.py
    activations.py
    metrics.py
    model_replace.py
    structured_linear.py
    evaluate.py
    reporting.py
    cli.py
  tests/
    test_transforms.py
    test_predictor.py
    test_codec.py
    test_packing.py
    test_accounting.py
    test_linear_equivalence.py
    test_model_restore.py
  scripts/
    run_screen.sh
    run_evaluate.sh
  results/<run_id>/
    manifest.json
    integration_audit.md
    resolved_config.yaml
    metrics.jsonl
    failures.jsonl
    payloads/
    plots/
    report.md
```

Core interfaces should separate codec correctness from model integration:

```python
select_anchors(W, config) -> AnchorSelection
encode_matrix(W, config, anchor_selection=None) -> MatrixPayload
decode_matrix(payload, *, dtype, device) -> Tensor
payload_nbytes(payload) -> dict[str, int]
apply_rotation(tensor, rotation_spec, *, inverse=False) -> Tensor
structured_linear(inputs, payload, bias=None) -> Tensor
score_matrix(W, W_hat, inputs=None) -> dict[str, float]
replace_target_weights(model, decoded_weights, selection) -> ReplacementAudit
```

`MatrixPayload` must contain everything required for decoding; its decoder must never access a hidden FP32 anchor, original weight, external permutation, or calibration tensor. Separate mathematical metadata from provenance metadata and count both when reporting serialized size.

Use content-addressed run IDs from model revision, module set, codec configuration, seed, and source commit. Resume only matching completed records. Failed jobs should be retried under an explicit policy, not silently overwritten. Log NaNs, OOMs, unsupported modules, and skipped tasks as failures or partial results, never as zeros.

## 12. Default configuration and commands for the agent to implement

A companion YAML file is supplied with the principal defaults. Treat the complete document as authoritative for mathematical and comparison details. Resolve model/dataset revisions before execution and save a fully concrete copy of the configuration in each run directory.

The intended CLI contract is:

```bash
# These commands are acceptance targets for the implementation.
python -m predictive_ptq.cli inventory --config configs/smoke.yaml
pytest -q
python -m predictive_ptq.cli screen --config configs/screen.yaml --resume
python -m predictive_ptq.cli score-activations --config configs/screen.yaml --resume
python -m predictive_ptq.cli evaluate --config configs/full_model.yaml --resume
python -m predictive_ptq.cli report --run-dir results/<run_id>
```

Do not launch full-model jobs before the required screening gates. A single orchestration script may invoke these stages, but each stage must also be independently runnable and resumable.

## 13. Required deliverables and report format

Deliver runnable code and tests, a lockfile/source revisions, resolved configs, actual packed payloads for the promoted matrices, a fresh-process decode check, the integration audit, raw results, and an experiment report.

Each matrix result record should include model/revision, module name, shape, method, `b/G/S/h`, anchor/coefficient/scale precision, seed, anchor-selection policy, payload and serialized bytes, local metrics, activation sample count and split hash, elapsed quantization time, peak workspace, and a success/failure status. Per-group geometry can be stored separately to keep matrix summaries compact.

The report should begin with a direct conclusion: **go**, **revise**, or **stop for this version**, followed by the evidence. Include:

- A scope table listing exactly which modules/parameters were and were not quantized.
- Geometry results excluding anchors, paired error plots, and error-versus-actual-bpw plots.
- Prediction-plus-rotation versus the matched protected-anchor rotation control.
- Perplexity/NLL and downstream scores under a frozen protocol, or an explicit explanation of why that stage was not reached.
- Actual storage and runtime status, including when quality was measured through BF16 reconstruction.
- Failure cases, resource use, implementation limitations, and the smallest justified next experiment.

Do not hide negative matrices, cherry-pick one rotation seed, compare different calibration/tokenization settings without disclosure, or turn a proposed kernel into a measured result. Preserve the raw results even when the hypothesis fails.

## 14. Copy-paste handoff prompt

> Implement and execute the attached “Anchor-Predictive Residual PTQ with Hadamard Rotation” plan. Start by inspecting the repository, checkpoints, and hardware and reusing existing working infrastructure. The version-1 method uses contiguous column groups, one stored INT4 anchor per group, signed FP16 coefficients, and 2-/3-/4-bit scalar quantization of an output-axis randomized Hadamard residual. Compute residuals against the decoded anchor and stored coefficient. Use decoder-aware projection-SSE anchor selection, not signed cosine medoids. Preserve the exact rotation inverse, scale grouping, and storage accounting specified in the plan.
>
> Build and pass the codec, transform, packing, and matrix-product equivalence tests before model sweeps. Compare against RAW, ROT, ANCHOR_RAW, ANCHOR_ROT, and PRED, with ANCHOR_ROT as the principal control. Count every byte. Use the already installed local Qwen3-30B-A3B checkpoint; do not download a substitute model. Start with the fixed first/middle/last-layer × 8-expert pilot, then routing-aware activation screening, held-out experts/layers, and finally an experts-only end-to-end quantization if the gates pass. Treat gate/up/down as separate logical matrices even if checkpoint/runtime tensors are fused. Keep the router and non-target weights unchanged. Do not add global reordering, learned predictors, cross-expert prediction, multiple anchors, fine-tuning, or custom kernel development merely to force a positive result. Keep a vLLM integration gap audit, but distinguish accuracy emulation from actual packed inference. Save reproducible configs, source revisions, packed round-trip examples, raw metrics, failures, and a go/revise/stop report. A well-supported negative result is successful completion of the initial experiment.

## 15. Primary references and implementation resources

References were checked on September 10, 2026. Package support is version-specific; pin the versions used by the actual experiment. The equations, defaults, gates, and storage example in this plan are the proposed experiment design, not results reported by these papers.

**[S1] QuaRot paper:** Ashkboos et al., “QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.” Rotation/invariance motivation.  
`https://arxiv.org/html/2404.00456v2`

**[S2] QuaRot official implementation:** reuse and inspect rotation utilities, not an assertion that our residual codec is included.  
`https://github.com/spcl/QuaRot`

**[S3] QuIP# paper and official code:** Tseng et al., ICML 2024. Randomized Hadamard processing, vector codebooks, and additional optimization distinguish the full method from our scalar baseline.  
`https://proceedings.mlr.press/v235/tseng24a.html`  
`https://github.com/Cornell-RelaxML/quip-sharp`

**[S4] Fast Hadamard Transform:** Dao-AILab official implementation and API.  
`https://github.com/Dao-AILab/fast-hadamard-transform`

**[S5] LLM Compressor transform documentation:** existing SpinQuant/QuaRot-style and QuIP-style transform recipes.  
`https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/transform/`

**[S6] LM Evaluation Harness:** EleutherAI's official evaluation framework.  
`https://github.com/EleutherAI/lm-evaluation-harness`

**[S7] Safetensors documentation:** tensor loading, partial tensor access, and storage.  
`https://huggingface.co/docs/safetensors/en/index`

**[S8] LQER:** Zhang et al., “LQER: Low-Rank Quantization Error Reconstruction for LLMs,” ICML 2024; related decomposition/inference prior art, not the same anchor algorithm.  
`https://proceedings.mlr.press/v235/zhang24j.html`  
`https://github.com/ChengZhang-98/lqer`

**[S9] GPTQ official implementation:** algorithm and existing low-bit baselines; verify modern model integration separately.  
`https://github.com/IST-DASLab/gptq`

**[S10] Qwen3 official release:** architecture summary for Qwen3-30B-A3B, including 48 layers, 128 total / 8 activated experts, and 30B / 3B total/activated scale.  
`https://qwenlm.github.io/blog/qwen3/`

**[S11] Qwen3 MoE Transformers implementation/config:** logical expert shapes and fused expert storage representation; verify against the locally installed Transformers version and checkpoint.  
`https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py`  
`https://huggingface.co/Qwen/Qwen3-30B-A3B`

**[S12] vLLM quantization documentation:** supported schemes and out-of-tree quantization registration. A plugin hook is not a working custom kernel.  
`https://docs.vllm.ai/en/latest/features/quantization/`

**[S13] Transformers perplexity documentation:** tokenization, context windows, and strided evaluation.  
`https://huggingface.co/docs/transformers/en/perplexity`

**[S14] WikiText official dataset repository:** dataset configurations, splits, and provenance.  
`https://huggingface.co/datasets/Salesforce/wikitext`
