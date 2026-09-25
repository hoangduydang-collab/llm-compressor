# Shared vs routed experts — GLM-5.3 layer 3

**Verdict:** the shared expert is not meaningfully related to the routed experts
in weight space. "Routed = shared + delta" removes 0.13% of routed energy — no
use for compression.

- Source: `zai-org/GLM-5.3-BF16` snapshot `304b8051…`, layer 3, 256 routed experts
  + 1 shared expert (same shape as a routed expert: intermediate 2048).
- Script: `pipeline/shared_expert_correlation.py` (sha256 `c85b2478…0946a`, as run).
  Job: `pipeline/k8s/hd-shared-expert-glm53-l3.yaml`, 1× H100 on ca-gpu06, ~10 min.
  First attempt failed at import (shipped the repo's non-empty `pipeline/__init__.py`);
  relaunched with an empty one, no weights had been read.
- Full output: `/mnt/cephfs/hoangduy/results/basis-codec/glm53-L3-shared-20260925t070931z/`.

A "neuron" = (gate row | up row | down column) ∈ R^18432. All tests are
permutation-invariant; each is compared against another routed expert and an
iid-Gaussian control.

| | vs shared | vs other routed | Gaussian |
|---|---|---|---|
| A: per-neuron max \|cos\|, median | 0.032 | 0.031 | 0.026 |
| A: per-neuron max \|cos\|, p99 | 0.088 | 0.053 | 0.034 |
| A: neurons with max \|cos\| > 0.5 | 1 of 524,288 | 0 | 0 |
| C: energy left after best-permutation scaled delta | 99.87% | 99.90% | 99.93% |
| B: gate energy in top-256 dirs (isotropic 4.2%) | 8.2% | 7.4% | 4.2% |
| B: down energy in top-256 dirs | 5.9% | 4.9% | 4.2% |

- **B** is above isotropic, but about equally for "shared" and "another routed"
  expert (up_proj is even lower for shared: 6.7% vs 7.3%). All experts lean
  toward a common hidden-space subspace; the shared expert is not special in it.
- **A** has a real but thin tail: routed neurons align with shared neurons a bit
  more than with another routed expert's (p99 0.088 vs 0.053). No copies.
- The shared expert has ~2–2.5× the energy of a mean routed expert
  (gate 1.94, up 2.47, down 2.41) and a slightly more concentrated spectrum
  (top-64 directions hold 12.8% of its gate energy vs 7.2% for Gaussian).

Scope: weight space only, one layer; says nothing about functional (output)
similarity on real activations.
