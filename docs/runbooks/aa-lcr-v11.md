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

Build an immutable, content-addressed code ConfigMap from a clean, committed
tree. Do not include a key in the ConfigMap. `kubectl create` deliberately
fails if the derived name already exists: never update a ConfigMap used by a
run.

```powershell
git diff --quiet
if ($LASTEXITCODE -ne 0) { throw "Tracked working-tree files are dirty." }
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw "Tracked index files are dirty." }
$revision = git rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw "Cannot resolve HEAD revision." }
$configMap = "hd-aa-lcr-v11-code-$($revision.Substring(0, 12))"
$existing = kubectl -n evaluation get configmap $configMap --ignore-not-found -o name
if ($existing) { throw "Refusing to update existing immutable ConfigMap: $configMap" }

$codeFile = (Resolve-Path pipeline/aa_lcr_v11.py).Path
$lockFile = (Resolve-Path pipeline/requirements-aa-lcr-v11.lock).Path
$code = [Convert]::ToBase64String([IO.File]::ReadAllBytes($codeFile))
$lock = [Convert]::ToBase64String([IO.File]::ReadAllBytes($lockFile))
@{
  apiVersion = "v1"
  kind = "ConfigMap"
  metadata = @{ name = $configMap; namespace = "evaluation" }
  immutable = $true
  data = @{ AA_LCR_CODE_REVISION = $revision }
  binaryData = @{
    aa_lcr_v11.py = $code
    "requirements-aa-lcr-v11.lock" = $lock
  }
} | ConvertTo-Json -Depth 6 | kubectl -n evaluation create -f -
```

The Job templates contain the `AA_LCR_CODE_CONFIGMAP` token. Render it only
with `$configMap` above; the runtime Job must contain neither that token nor
any revision placeholder.

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
  --index-url https://pypi.org/simple `
  --output-file=pipeline/requirements-aa-lcr-v11.lock `
  pipeline/requirements-aa-lcr-v11.in
```

Review and commit the generated lock before creating the ConfigMap. The staging
Job derives its venv name from the lock SHA-256, never deletes an existing
environment, installs with `--require-hashes`, and records the digest, Python
version, and `pip freeze`.

The finding that `httpx2` is a legacy or invalid replacement is rejected.
On 2026-09-13, the official `openai==3.8.0` metadata resolved maintained
`pydantic`, `httpx2`, and `httpcore2`; a hash-checked dry run exited zero.
Do not replace `httpx2` with legacy `httpx`. Sources:
https://pypi.org/pypi/openai/3.8.0/json and
https://pypi.org/pypi/httpx2/2.12.0/json.

```powershell
py -3.12 -m pip install --dry-run --ignore-installed --require-hashes `
  -r pipeline/requirements-aa-lcr-v11.lock
```

## Stage and canary

After authorization, render and create a fresh Job from its `generateName`
template. Before every launch, the helper refuses to start if a Pending or
Running Pod already owns that run ID, preventing concurrent SQLite access.
Capture the generated name for logs; do not use `kubectl apply` for Jobs.

```powershell
function Start-AaLcrJob([string]$manifest, [string]$runId) {
  $active = @(kubectl -n evaluation get pods -l "aa-lcr-run-id=$runId" `
    --field-selector=status.phase=Pending,status.phase=Running -o name)
  if ($active.Count -gt 0) {
    throw "Run $runId already has an active Pod: $($active -join ', ')"
  }
  $rendered = (Get-Content $manifest -Raw).Replace(
    "AA_LCR_CODE_CONFIGMAP", $configMap
  )
  if ($rendered -match "AA_LCR_CODE_CONFIGMAP|REPLACE_WITH_GIT_COMMIT") {
    throw "Refusing to create a Job with unresolved runtime placeholders."
  }
  $jobName = $rendered | kubectl -n evaluation create -f - `
    -o jsonpath='{.metadata.name}'
  if (-not $jobName) { throw "Job creation did not return a name." }
  Write-Host "Created $jobName for $runId"
  kubectl -n evaluation logs -f "job/$jobName"
}

Start-AaLcrJob pipeline/k8s/stage-aa-lcr-v11.yaml aa-lcr-v11-stage
```

When staging has completed, authorize the canary:

```powershell
Start-AaLcrJob pipeline/k8s/hd-aa-lcr-v11-canary.yaml `
  glm53-w4afp8-aa-lcr-v11-canary-r1
```

Inspect the canary before proceeding:

- Job succeeded with `backoffLimit: 0`; logs contain no environment values.
- `run.sqlite` has one candidate and one valid external OpenAI judgment.
- The retrieved judge model is `gpt-5.6-luna`; the judge preflight passed.
- The candidate server snapshots agree with the immutable run fingerprint.
- Published files exist under
  `/mnt/cephfs/hoangduy/results/glm53-aa-lcr-v11/glm53-w4afp8-aa-lcr-v11-canary-r1`.

## Full run and resume

Only after a reviewed canary, create the full CPU-only Job:

```powershell
Start-AaLcrJob pipeline/k8s/hd-aa-lcr-v11-full.yaml `
  glm53-w4afp8-aa-lcr-v11-full-r1
```

The full result must contain 300 candidate rows and 300 valid judgments: 100
questions × 3 repeats. The CLI is resumable; rerun the same command to fill
only missing candidate or judge rows, while preserving the immutable
fingerprint:

```powershell
Start-AaLcrJob pipeline/k8s/hd-aa-lcr-v11-full.yaml `
  glm53-w4afp8-aa-lcr-v11-full-r1
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
