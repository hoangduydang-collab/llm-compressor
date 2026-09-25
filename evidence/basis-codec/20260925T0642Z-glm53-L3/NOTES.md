# Basis codec (shared basis + quantized coefficients, no residual) — GLM-5.3 layer 3

**Verdict: STOP.** No variant approaches INT4-g128 weight error at ~3 bpw. Every
arm is indistinguishable from the same arm on iid-Gaussian weights.

- Source: `zai-org/GLM-5.3-BF16` snapshot `304b8051…`, layer 3 (first MoE
  layer), all 256 routed experts, gate/up `[2048, 6144]`, down `[6144, 2048]`.
  Shared expert excluded.
- Script: `pipeline/basis_codec_layer_analysis.py`, sha256 `af72591e…71ffb`
  (identical to the copy the pod ran). Job: `pipeline/k8s/hd-basis-codec-glm53-l3.yaml`,
  1× H100 on ca-gpu06, 14 min wall (8 min of it loading 19.3 GB from cephfs at 40 MB/s).
- Full output: `/mnt/cephfs/hoangduy/results/basis-codec/glm53-L3-20260925t064203z/`
  (`results.json`, `summary.md`, `run.log`); `run.log` here is the pod log.

## Numbers (relative weight MSE)

| | gate | up | down | iid-Gaussian |
|---|---|---|---|---|
| **INT4-g128 @4.125 bpw (target)** | **1.13%** | **1.12%** | **1.21%** | 1.02% |
| INT3-g128 @3.125 | 4.13% | 4.13% | 4.32% | 3.82% |
| Dense PCA @3.125, best d | 4.06% (d16) | 4.04% (d16) | 4.01% (d128) | 3.82% |
| Dense AM/GM bound, any d in 8–128 | ≤0.001 bit | ≤0.001 bit | 0.000 bit | 0.000 bit |
| Sparse best @~3.03 (d8 K256 s2) | 5.43% | 5.44% | 5.20% | 5.39% |
| Eigen-experts, error floor at K=64 | 66.5% | 66.3% | 65.1% | — |

- **Dense:** the pooled block covariance is flat — the top quarter of directions
  holds 25–26% of the energy (flat = 25%). PCA gains ≤2% relative over the
  identity basis, well short of the 3.6× needed.
- **Sparse:** all three 3-bpw configs are *worse* than plain INT3, and match the
  Gaussian control to within 0.2 pp. The MOD fit converged smoothly with no dead
  atoms, so this is not a fitting failure.
- **Eigen-experts:** the mean expert holds 0.39% of the energy (= 1/256, i.e. no
  shared component). The floor sits below the flat line 1−K/256 (e.g. 41% vs 50%
  at K=128); unequal expert norms would produce this without shared directions —
  not separated here. Matching INT4 would need K≈256, i.e. storing every expert.
- Real weights are slightly *harder* than Gaussian (INT4 1.13% vs 1.02%;
  row-normalized excess kurtosis 0.22 gate/up, 1.18 down).

## Scope limits

Weight-space error only (no activation weighting), so this can rule the idea
out but could not have ruled it in. Eigen-experts measured without cross-expert
neuron alignment. Sparse coefficients quantized open-loop after fitting.
