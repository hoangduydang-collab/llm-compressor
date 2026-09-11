# GPTQ chain venv launcher fix

## Problem

The Rancher chain installs this repository into `/work/venv`, but `run_lane`
invokes the image's `/usr/local/bin/torchrun`. That entrypoint launches
`/usr/bin/python3`, so distributed ranks cannot see the editable
`llmcompressor` package installed in the venv. Representative EP4 therefore
fails with `ModuleNotFoundError: No module named 'llmcompressor'`.

## Design

Launch every distributed lane with the explicit venv interpreter:
`/work/venv/bin/python -m torch.distributed.run`. Add `llmcompressor` to the
environment preflight imports so a missing editable install fails before any
quantization lane starts.

This fixes the interpreter boundary directly. It does not reinstall PyTorch,
add a wrapper, change quantization behavior, or modify the failure-hold policy.

## Verification

Render the real Job template in a regression test and require:

- the distributed launcher uses `/work/venv/bin/python`;
- the system `torchrun` entrypoint is absent;
- preflight imports `llmcompressor`.

Run the focused renderer tests and validate the rendered container body with
`bash -n`.
