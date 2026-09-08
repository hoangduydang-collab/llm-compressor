# Local NCCL attempt, 2026-09-09

Implementation: `6c1f0b2a`. Controller: detached tmux, `controller.sh`.

Job **830661** was allocated two A100-40 GPUs on xgph12 but failed after 10 seconds
before the worker script started. Slurm reports `NonZeroExitCode`, exit `1:0`.
The controller reports “Unable to allocate resources: Job/step already completing
or completed”. Only the extern step appears in accounting; no pytest step,
worker preflight, checkpoint, or NCCL result exists. The underlying launch failure
is not established by these records. No automatic retry was made.

This is an infrastructure failure, not a test failure or an NCCL pass. The three
committed GPU cases still need execution. Raw controller and scheduler records
are retained here. The worker and controller scripts are shared-filesystem paths.
