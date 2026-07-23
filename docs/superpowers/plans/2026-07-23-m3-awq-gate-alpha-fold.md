# M3 AWQ r7 — function-preserving down-side smoothing via the gate path ("gate-alpha fold")

- Status: DESIGN — conditional; do not implement until the r6 decision gate below fires.
- Date: 2026-07-23
- Prereq reading: BUGS_AND_FIXES.md "AWQ up->down smoothing fold is not
  function-preserving on MiniMax-M3"; `M3_AWQ_R6_REQUANT_HANDOFF.md`.

## Decision gate (when to build this at all)

Implement r7 only if the r6 evaluation (up->down mapping removed) shows in-house
AWQ still **materially behind GPTQ** on the reasoning suite (GPQA recovery or
budget-exhaustion gap beyond the paired noise band). If r6 ≈ GPTQ, ship r6 and
close; the mapping this design rescues was worth a median ~6% down-weight-MSE
on a flat activation landscape — do not build serve plumbing to reclaim it
unless the eval says that 6% matters.

## Problem being solved

We want the down_proj input-group reshaping that AWQ's per-expert up->down
mapping provided (`down_cols *= s_r`, median 6% / p75 10% weight-MSE reduction
in the r5 telemetry), but the compensating `1/s_r` cannot live on `up_proj`:
M3's expert activation `h = (clamp(up, ±L) + β)·glu` (β=1.0, L=7.0) is affine
and clamped in `up`, so an up-side fold changes the function (the r5 bug).

## The r7 scheme (proposed by Duy; "Case A" = exact variant)

Carry `1/s_r` through the **gate** path's homogeneous factor instead:

1. `W_gate[r,:] /= s_r` (per intermediate channel r, per expert)
2. per-channel sigmoid slope `alpha_r = 1.702 * s_r`
3. per-channel gate clamp `limit_r = 7.0 / s_r`
4. `W_down[:,r] *= s_r` (unchanged — this is the part that helps quantization)

Exactness (per channel, all inputs; `c_L(x) = min(x, L)`):

```
glu' = c_{7/s}(g/s) · σ(1.702·s · c_{7/s}(g/s))
     = (c_7(g)/s) · σ(1.702 · c_7(g))          = glu / s
h'   = (c_±7(u) + β) · glu/s                    = h / s
out  = (d·s) · (h/s)                            = d · h   ✓ identical function
```

β and the up-clamp are never touched — exact for every input, not just in
expectation. Without step 3 (per-channel clamp) the identity still holds for
all `g ≤ 7·min(1, s_r)`; the residual is confined to the rare `g > ~7` tail.
r7 uses Case A (both α and clamp co-scaled): the two vectors share one source
scalar, so the second one is free.

Quantization properties:
- down_proj: identical column reshaping as the r5 mapping → full benefit kept.
- gate_proj: per-row scale → absorbed exactly by its per-row-group int4 scales
  (no harm). Composes with the MoE-input mapping's column scaling (row and
  column scalings commute).
- AWQ grid-search semantics unchanged (loss still `Q(W_down·s)/s` on cached
  down inputs); only the fold destination moves from up to gate(+alpha).

## Design decisions / improvements over the raw idea

1. **Single-tensor ABI.** Store only `s_r` (fp32, one `[intermediate]` vector
   per expert per layer, ~11.2M params ≈ 45 MB fp32 model-wide; name:
   `experts.N.gate_smooth_scale`). Derive `alpha_r = 1.702*s_r` and
   `limit_r = 7.0/s_r` at load time in fp32 — avoids independent bf16
   roundings of two stored vectors and makes the invariant below exact.
2. **Self-checking invariant.** `alpha_r * limit_r == 1.702 * 7.0` identically
   by construction; the fold-consistency gate asserts it, plus
   `(gate_row_scale_implied) * s_r == 1` against the base checkpoint, plus a
   functional spot-check: random `x` through one expert pre/post fold, rel
   err ≤ bf16 tolerance (this is the check class that would have caught the
   r5 bug — weight-algebra checks provably cannot).
3. **Scale backstop.** Clamp `s_r` to `[1/8, 8]` at fold time (mirrors the r4
   dead-channel lesson: bound every fold).
4. **Selective application (optional).** The grid already reports per-expert
   error reduction; skip the fold where predicted benefit < threshold (e.g.
   <5%) to shrink the blast radius. Default on; decide from r6 telemetry.

## Serve-side implementation (vLLM, no CUDA required)

- Our path resolves `SWIGLUOAI_UNINTERLEAVE -> SiluAndMulWithClamp`, whose
  CUDA op takes scalar `alpha/limit/beta` — but whose `forward_native` is pure
  elementwise torch: `σ(self.alpha * gate)` **broadcasts a per-channel tensor
  alpha as-is**, same for the clamp. Plan: force the native path for M3
  experts and pass `[d]`-shaped tensors. Perf: unfused elementwise, memory-
  bound, small single-digit % of MoE time; fuse later only if measured.
- Per-expert dimension: the activation runs on the grouped all-experts tensor,
  so `alpha/limit` must be expanded row-wise by expert id using the expert
  boundary offsets available at the `apply_moe_activation` call site — the
  same seam `vllm_m3_patches.patch_vllm_w4a8_swigluoai_uninterleave` already
  wraps to inject the scalar swiglu params. This is the main plumbing item.
- Loader: map `gate_smooth_scale` tensors; extend the serve ABI gate.

## Calibration-side implementation

- M3-specific fold hook replacing the removed mapping: reuse the AWQ grid
  search (smooth target: gate; balance: down) but at fold time write
  `W_gate rows /= s_r`, `W_down cols *= s_r`, emit `gate_smooth_scale = s_r`.
- Linearized experts: `_apply_gate` override reading per-channel
  `alpha_r/limit_r` so calibration-time propagation matches serve exactly.
- Reexport: carry the new tensors through `reexport_minimax_m3_vllm`.

## Validation ladder (in order, each fail-closed)

1. Unit: fold identity on random tensors (fp32 exact; bf16 ≤ 1e-2 rel).
2. Single-layer smoke quant (include layer 8 — dead-channel path — and 30).
3. `pipeline/m3_verify_no_updown_fold.py` must PASS (down cols are scaled in
   r7 — NOTE: this gate must learn to divide out `gate_smooth_scale` first;
   extend it before the r7 smoke).
4. Functional spot-check gate (item 2 in design decisions).
5. Full requant + stuck-item probe + paired eval, same contract as r6.

## Effort estimate

Calibration-side ~2-3 days; serve-side patch + ABI + gates ~3-5 days;
validation ladder ~2 days cluster-elapsed. Only spend it if the r6 decision
gate fires.
