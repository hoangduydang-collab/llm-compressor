# GPTQ chain checkpoint locator fix

## Problem

The representative EP4 lane completed quantization, saved a valid checkpoint,
and passed the pipeline's offline verification. The chain then entered its
failure hold because `checkpoint_for` searched only `<lane>/*/checkpoint`.
Pipeline output uses `<lane>/<model>/<timestamp>/checkpoint`.

## Design

Search recursively beneath the fresh lane root for directories named
`checkpoint`. Keep the existing newest-modification-time selection and fail
closed when no candidate exists. Each chain run has a new durable run root, so
recursive discovery cannot select an artifact from an earlier Job.

Add a rendered-template regression test that requires recursive discovery and
rejects the old one-level glob. Do not change quantization, validation, resource,
or hold behavior.

## Relaunch

After the fix is tested, reviewed, committed, and pushed, submit its replacement
Job pinned to `gpu04` while the failed Job still holds the node. Confirm the
replacement is Pending for insufficient GPUs, then delete the failed holder and
verify that the replacement acquires `gpu04`.
