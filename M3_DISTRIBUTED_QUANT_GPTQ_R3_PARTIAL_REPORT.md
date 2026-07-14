# Partial evidence: distributed GPTQ smoke r3

- Protocol revision: `2026-07-15-r3`
- Required fix ancestor: `4801028f`
- Executed Git commit: `f481561945dc7fd881333f3cb8e1c6c2e32878e6`
- Run ID: `20260714T170500Z-m3-ddp-quant-smoke-r3`
- Classification: `PARTIAL_RETURN_GPTQ_ARM`
- Overall packet: still open because the authorized AWQ arm is running

## GPTQ result

- Slurm job: `12923`
- Step: `12923.0`
- Node: `h108-gpu-polaris.pod4.lab.bitdeer.ai`
- GPUs: eight H100 GPUs
- Arm return code: `1`
- Approximate elapsed time: 18m22s
- Result directory:
  `/mnt/nfs/hoangduy/results/m3-distributed-quant-smoke/20260714T170500Z-m3-ddp-quant-smoke-r3/gptq/MiniMax-M3-gptq-W4AFP8/20260714-170748`

The planner’s loader fix allowed the model load to proceed beyond the previous
`tie_word_embeddings` constructor error. The arm later failed in distributed
initialization with an NCCL watchdog timeout:

```text
[Rank 3] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=5, OpType=BROADCAST, NumelIn=2, NumelOut=2,
Timeout(ms)=600000) ran for 600017 milliseconds before timing out.
```

The first reported root-cause rank in the torchrun failure summary was rank 7;
the watchdog diagnostics also report rank 3. Torchrun subsequently terminated
the sibling ranks. No calibration partition, native quantization metrics,
provenance, or smoke completion artifact was produced.

## Raw evidence

Durable raw logs remain under:

`/mnt/nfs/hoangduy/logs/m3-distributed-quant-smoke/20260714T170500Z-m3-ddp-quant-smoke-r3/gptq/`

SHA-256:

- `torchrun.out` (1,248 bytes):
  `7efb2d6d789826d5a85c88a825f4b05f1d468e452c68bd72478f85ad67d13ca6`
- `torchrun.err` (112,676 bytes):
  `5dffed514fd9a10b762d6868574baf37b3ab803926bb41699b12ebd9eed12eb1`
- `resources.log` (13,417 bytes):
  `02613e2e1ef1ea2601d86165b57d7a4d3accb2fed903ce6c9df7e0c3393b3746`

The corresponding copied partial evidence is under:

`results/m3-distributed-quant-speedup/20260714T170500Z-m3-ddp-quant-smoke-r3-gptq-partial/`

## AWQ independence

AWQ remains active as Slurm job `12924` on `gpu-h108`. It was not canceled,
restarted, or altered. This report does not close the packet or make a
quantization-speed verdict; the final return remains pending AWQ completion.
