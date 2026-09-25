# Weight-structure probes on GLM-5.3 routed experts — layers 3, 40, 76

**Conclusion:** no exploitable weight-space structure at any depth. Layers 40 and
76 are statistically indistinguishable from iid Gaussian weights; layer 3 (first
MoE layer) is the most structured and still nowhere near usable. Sub-4-bit gains
must come from the activation side (Hessian-weighted allocation, GPTQ-family,
fine-tuning, VQ shape gain), not from a shared basis or cross-expert prediction.

| | L3 | L40 | L76 | iid Gaussian |
|---|---|---|---|---|
| INT4-g128 rel MSE (gate/up/down) | 1.13/1.12/1.21% | 1.02/1.02/1.02% | 1.02/1.02/1.04% | 1.02% |
| row-normalized excess kurtosis (gate,up/down) | 0.22/1.18 | 0.00/0.01 | 0.00/0.05 | 0 |
| dense AM/GM bound, any d in 8–128 | ≤0.001 bit | 0.000 | 0.000 | 0.000 |
| best dense PCA @3.125 bpw | 4.01–4.06% | 3.82% | 3.82% | 3.82% |
| best sparse @3.03 bpw | 5.20–5.44% | 5.40% | 5.38–5.40% | 5.39% |
| eigen-experts floor at K=128 | 40–41% | 49% | 47–48% | 50% |
| shared / mean routed expert energy | 1.9–2.5× | 0.6–0.8× | 0.5–0.75× | 1× |
| shared+delta residual energy | 99.87% | 99.83% | 99.81% | 99.93% |
| neuron max-cos p99, vs shared / vs routed | 0.088/0.053 | 0.128/0.055 | 0.186/0.063 | 0.034 |

The only depth trend: a thin tail of routed neurons weakly aligned with shared
neurons grows with depth (p99 0.09 → 0.19), while the shared expert shrinks
relative to routed ones. Neither is large enough to exploit.

Unverified hypothesis for the near-perfect Gaussianity: Muon-style
(orthogonalized, spectrally flat) optimizer updates.

Runs (1× H100 each on ca-gpu06; raw output under
`/mnt/cephfs/hoangduy/results/basis-codec/`):

- `20260925T0642Z-glm53-L3/` — basis codec, layer 3 (`NOTES.md` has detail)
- `20260925T0709Z-glm53-L3-shared/` — shared expert, layer 3 (`NOTES.md`)
- `20260925T0723Z-glm53-L40/`, `20260925T0723Z-glm53-L76/` — both analyses,
  `pipeline/k8s/hd-layer-structure-glm53-l40-l76.yaml`; pod logs in `run-log.txt`
