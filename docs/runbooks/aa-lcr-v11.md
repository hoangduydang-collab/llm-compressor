# AA-LCR v1.1 external-judge runbook

This run evaluates `glm-5.3-w4afp8` against the pinned AA-LCR v1.1 dataset.
The judge is the external OpenAI API, accessed only through `OPENAI_API_KEY`;
it is never a local model or an agent. Results are a public-methodology
reproduction, not an official AA-LCR leaderboard score.

## Preflight gates

Run the authoritative Rancher availability check before asking to schedule a
Job. It is informational only; the evaluation Jobs request no GPUs:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' scripts/gpu-free.sh --verify
```

Create/update the code ConfigMap from the reviewed commit. Do not include any
key in the ConfigMap:

```powershell
kubectl -n evaluation create configmap hd-aa-lcr-v11-code `
  --from-file=aa_lcr_v11.py=pipeline/aa_lcr_v11.py `
  --from-file=requirements-aa-lcr-v11.lock=pipeline/requirements-aa-lcr-v11.lock `
  --dry-run=client -o yaml | kubectl apply -f -
```

Create the API-key Secret by typing the key only at the secure prompt. This
command is shown for the operator and must not be executed by an agent without
fresh authorization:

```powershell
$secure = Read-Host "OpenAI API key" -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
  $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
  @{
    apiVersion = "v1"
    kind = "Secret"
    metadata = @{
      name = "hd-openai-aa-judge"
      namespace = "evaluation"
    }
    type = "Opaque"
    stringData = @{ OPENAI_API_KEY = $key }
  } | ConvertTo-Json -Depth 6 | kubectl apply -f -
} finally {
  if ($ptr -ne [IntPtr]::Zero) {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
  }
  $key = $null
  $secure = $null
}
```

Generate the hash-locked dependency file before staging, in an isolated Python
3.12 environment:

```powershell
py -3.12 -m venv .venv-aa-lcr-lock
.\.venv-aa-lcr-lock\Scripts\python.exe -m pip install pip-tools==7.5.0
.\.venv-aa-lcr-lock\Scripts\pip-compile.exe `
  --generate-hashes `
  --resolver=backtracking `
  --output-file=pipeline/requirements-aa-lcr-v11.lock `
  pipeline/requirements-aa-lcr-v11.in
```

Review and commit the generated lock before creating the ConfigMap. The staging
Job derives its venv name from the lock SHA-256, never deletes an existing
environment, installs with `--require-hashes`, and records the digest, Python
version, and `pip freeze`.

## Stage and canary

After authorization, create the CPU-only stage Job:

```powershell
kubectl apply -f pipeline/k8s/stage-aa-lcr-v11.yaml
kubectl -n evaluation logs -f job/hd-stage-aa-lcr-v11
```

Then authorize and apply the canary:

```powershell
kubectl apply -f pipeline/k8s/hd-aa-lcr-v11-canary.yaml
kubectl -n evaluation logs -f job/hd-aa-lcr-v11-canary
```

Inspect the canary before proceeding:

- Job succeeded with `backoffLimit: 0`; logs contain no environment values.
- `run.sqlite` has one candidate and one valid external OpenAI judgment.
- The retrieved judge model is `gpt-5.6-luna`; the judge preflight passed.
- The candidate server snapshots agree with the immutable run fingerprint.
- Published files exist under
  `/mnt/cephfs/hoangduy/results/glm53-aa-lcr-v11/glm53-w4afp8-aa-lcr-v11-canary-r1`.

## Full run and resume

Only after a reviewed canary, apply the full CPU-only Job:

```powershell
kubectl apply -f pipeline/k8s/hd-aa-lcr-v11-full.yaml
kubectl -n evaluation logs -f job/hd-aa-lcr-v11-full
```

The full result must contain 300 candidate rows and 300 valid judgments: 100
questions × 3 repeats. The CLI is resumable; rerun the same command to fill
only missing candidate or judge rows, while preserving the immutable
fingerprint:

```powershell
kubectl apply -f pipeline/k8s/hd-aa-lcr-v11-full.yaml
```

Do not change the run ID, endpoint identity, code revision, model, repeat
count, or judge contract when resuming. The CLI rejects checkpoints and result
directories that belong to another fingerprint, and refuses to summarize or
publish an incomplete run.

When the result is retained and no further evaluation needs the credential,
the operator may remove it:

```powershell
kubectl -n evaluation delete secret hd-openai-aa-judge
```
