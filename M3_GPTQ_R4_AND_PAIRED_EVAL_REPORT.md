# GPTQ r4 and paired evaluation evidence

- GPTQ r4 run: `20260714T191900Z-m3-ddp-quant-smoke-r4-gptq`
- GPTQ commit: `7967bbd11180803d3a99190788d39510f7566765`
- GPTQ Slurm job: `12925` on `gpu-h101`
- GPTQ outcome: failed, return code `1`
- AWQ authorization: pulled from packet revision `2026-07-15-r4`; not launched

## GPTQ r4

The three-hour NCCL timeout fix allowed the run to proceed well beyond the
previous r3 failure. It then failed after about 7h27m during a later
distributed broadcast:

```text
WorkNCCL(SeqNum=20980, OpType=BROADCAST, NumelIn=3, NumelOut=3,
Timeout(ms)=10800000) ran for 10800033 milliseconds before timing out.
```

Rank 0 also reported:

```text
RuntimeError: unable to allocate shared memory(shm) for file
</torch_1140157_3143461634_13983>: Success (0)
```

The last visible model-preparation progress was linearizing approximately
layer 17, `15/57` MoE layers. No calibration partition, native quantization
metrics, provenance, or completion artifact was produced. The durable raw
logs are under:

`/mnt/nfs/hoangduy/logs/m3-distributed-quant-smoke/20260714T191900Z-m3-ddp-quant-smoke-r4-gptq/`

No retry or dynamic patch was performed.

## Paired quality evaluation

Run root:

`results/m3-quality/20260714T154500Z-m3-paired-gptq-awq-grouped-r3`

The four grouped benchmark arms completed successfully:

- AWQ reasoning, job `12917`, node `gpu-h123`: GPQA and IFEval, 100 each.
- GPTQ reasoning, job `12916`, node `gpu-h117`: GPQA and IFEval, 100 each.
- AWQ broad math, job `12915`, node `gpu-h101`: MMLU-Pro/GSM8K 100 each,
  AIME 2025 30.
- GPTQ broad math, job `12918`, node `gpu-h107`: MMLU-Pro/GSM8K 100 each,
  AIME 2025 30.

The two distributional-probe arms completed with return code `1` and no
completion artifact. They are not included as benchmark scores below.

| Task | AWQ | GPTQ |
|---|---:|---:|
| GPQA Diamond (`acc_norm`) | 0.24 | 0.28 |
| IFEval prompt strict | 0.84 | 0.88 |
| MMLU-Pro | 0.76 | 0.76 |
| GSM8K strict | 0.97 | 0.98 |
| AIME 2025 | 0.2333 (7/30) | 0.6333 (19/30) |

These are paired directional quick-evaluation results on seeded subsets
(AIME has 30 items), not directly comparable to public leaderboard scores.
This report records observations only and does not make a model-adoption or
performance decision.
