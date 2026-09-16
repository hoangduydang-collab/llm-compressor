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
version, and `pip freeze`. It also materializes the pinned `cl100k_base`
vocabulary into the shared versioned cache
`/mnt/cephfs/hoangduy/cache/aa-lcr-v11-tiktoken`. This prefetch runs even when
the lock-derived venv already exists, records nonempty cache-file SHA-256
inventory in `tiktoken-cache-sha256.txt`, and does not print cache contents.
Canary and full Jobs use that same `TIKTOKEN_CACHE_DIR` and fail closed before
the runner if it has no nonempty artifact. Do not disable TLS or certificate
verification: `tiktoken.get_encoding("cl100k_base")` verifies the upstream
vocabulary's expected hash.

AA-LCR candidate prompts preserve raw CSV question text, including trailing
whitespace and zero-width characters, exactly as the official loader does.
The five upstream v1.1 `input_tokens` metadata discrepancies (questions 5,
21, 62, 65, and 81) are pinned and validated as exact published/actual
`cl100k_base` tuples before endpoint traffic. Actual prompt tokens—not the
stale published metadata—define request length. The immutable run contract
and published `summary.json` expose the discrepancy map. Keep this validation
until a newly pinned upstream revision corrects those counts.

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
Running Pod or a nonterminal Job already owns that run ID, preventing concurrent
SQLite access. A Job can exist before its Pod has been created, so checking Pods
alone has a race.
Capture the generated name for logs; do not use `kubectl apply` for Jobs.

```powershell
function Start-AaLcrJob([string]$manifest, [string]$runId) {
  $jobsJson = kubectl -n evaluation get jobs -l "aa-lcr-run-id=$runId" -o json
  if ($LASTEXITCODE -ne 0) {
    throw "Cannot determine whether run $runId already has an existing Job."
  }
  try {
    $jobs = $jobsJson | ConvertFrom-Json -ErrorAction Stop
  } catch {
    throw "Kubectl returned malformed Job JSON for run $runId: $($_.Exception.Message)"
  }
  $nonterminalJobs = @(
    $jobs.items | Where-Object {
      $terminalConditions = @(
        $_.status.conditions | Where-Object {
          @('Complete', 'Failed') -contains $_.type -and $_.status -eq 'True'
        }
      )
      $terminalConditions.Count -eq 0
    }
  )
  if ($nonterminalJobs.Count -gt 0) {
    throw "Run $runId already has a nonterminal Job: $($nonterminalJobs.metadata.name -join ', ')"
  }
  $podsJson = kubectl -n evaluation get pods -l "aa-lcr-run-id=$runId" -o json
  if ($LASTEXITCODE -ne 0) {
    throw "Cannot determine whether run $runId already has an active Pod."
  }
  try {
    $pods = $podsJson | ConvertFrom-Json -ErrorAction Stop
  } catch {
    throw "Kubectl returned malformed Pod JSON for run $runId: $($_.Exception.Message)"
  }
  $active = @(
    $pods.items | Where-Object {
      @('Pending', 'Running') -contains $_.status.phase
    }
  )
  if ($active.Count -gt 0) {
    throw "Run $runId already has an active Pod: $($active.metadata.name -join ', ')"
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
  kubectl -n evaluation wait --for=create pod -l "job-name=$jobName" --timeout=180s
  if ($LASTEXITCODE -ne 0) {
    throw "Timed out waiting for a Pod created by Job $jobName."
  }
  kubectl -n evaluation logs -f "job/$jobName"
}

Start-AaLcrJob pipeline/k8s/stage-aa-lcr-v11.yaml aa-lcr-v11-stage
```

When staging has completed, authorize the canary:

```powershell
Start-AaLcrJob pipeline/k8s/hd-aa-lcr-v11-canary.yaml `
  glm53-w4afp8-aa-lcr-v11-canary-r1
```

Temperature `1.0` / `top_p=0.95` matches PhalaCloud's published AA-LCR
sampling. It is a sampling ablation, not the public-methodology default
(`0.6` / `1.0`), and must not resume the `0.6` checkpoint. After a committed
code ConfigMap, authorize:

```powershell
Start-AaLcrJob pipeline/k8s/hd-aa-lcr-v11-canary-t1.yaml `
  glm53-w4afp8-aa-lcr-v11-canary-t1p95-r1
```

Inspect `/mnt/cephfs/hoangduy/aa-lcr-v11-work/glm53-w4afp8-aa-lcr-v11-canary-t1p95-r1/run.sqlite`
for contract temperature `1.0` and `top_p` `0.95` before the 300-unit job:

```powershell
Start-AaLcrJob pipeline/k8s/hd-aa-lcr-v11-full-t1.yaml `
  glm53-w4afp8-aa-lcr-v11-full-t1p95-r1
```

### Phala sampling at a doubled output cap

32 of the 300 traces in the 0.6 run hit the 131,072 cap with empty answers, so
the `-t1-2x` pair reruns Phala sampling at `--candidate-max-tokens 262144`. It
is a **fresh measurement, not a resume**: the cap is in the run contract, so it
changes the fingerprint and gets its own `…-t1p95-2x-r1` run ids. The published
claim becomes `AA-LCR v1.1 sampling ablation (temperature=1, top_p=0.95,
max_tokens=262144)` with an explicit limitation that it is not comparable to a
131,072-cap AA-LCR result. Upper bound on what the extra budget can buy: the
0.6 run's 32 truncations, i.e. 246/300 = 82.0%.

```powershell
Start-AaLcrJob pipeline/k8s/hd-aa-lcr-v11-canary-t1-2x.yaml `
  glm53-w4afp8-aa-lcr-v11-canary-t1p95-2x-r1
Start-AaLcrJob pipeline/k8s/hd-aa-lcr-v11-full-t1-2x.yaml `
  glm53-w4afp8-aa-lcr-v11-full-t1p95-2x-r1
```

Two knobs exist because the doubled cap does not fit two concurrent requests:

- `--candidate-concurrency` is an **admission ceiling** (1–2, default 2), not a
  fixed width. AA-LCR prompts run 76,820–114,611 tokens, so a worst-case
  262,144-cap request needs ~376,755 KV tokens and two of them need ~753,510 —
  over the 598,848-token pool of the TP=16 serve. At the 131,072 cap two fit
  (~491,366) and the ceiling is reached normally.
- `--candidate-long-attempt-seconds` (default 900) is when the gate stops
  admitting a second attempt. Non-streaming `chat/completions` returns nothing
  until decode finishes, so elapsed time is the only in-flight signal that a
  trace is heading for the cap. The gate is reactive: it refuses to open a
  *second* slot while a long attempt runs, and cannot undo co-residency that
  already exists. Each `generate` phase prints an `admission gate:` line
  recording how often it held at one.

`build_run_contract` also fails closed before any endpoint traffic when
`largest prompt + max_tokens` exceeds the serve's reported
`max_total_num_tokens`. That guard rejects a 524,288 cap on this pool
(638,899 > 598,848) and accepts 262,144. Prompt lengths there are cl100k
counts, a proxy for the GLM tokenizer, so treat it as a floor on the real
footprint rather than a certificate.

The `-t1-2x` full Job gets `activeDeadlineSeconds: 172800` (48h) rather than
24h: doubling the cap roughly doubles generated-token volume and the gate
serialises exactly the long tail the cap raise exists to reach. Resume still
works if it dies on deadline — relaunch the same run id.

Inspect the canary before proceeding:

- Job succeeded with `backoffLimit: 0`; logs contain no environment values.
- Inspect the durable checkpoint at
  `/mnt/cephfs/hoangduy/aa-lcr-v11-work/glm53-w4afp8-aa-lcr-v11-canary-r1/run.sqlite`.
- `run.sqlite` has one candidate, one valid external OpenAI judgment, and one
  immutable successful judge-preflight audit.
- Requested, retrieved, and returned judge model are exactly `gpt-5.6-luna`;
  the recorded endpoint is exactly `https://api.openai.com/v1`.
- The candidate server snapshots agree with the immutable run fingerprint.
- Expected and observed served model are exactly `glm-5.3-w4afp8`.

The canary intentionally does not publish a headline bundle. Its JSON
completion record points operators to the durable one-unit checkpoint; only a
strict 100-question × 3-repeat full run may summarize and publish.

## Full run and resume

Only after a reviewed canary, create the full CPU-only Job:

```powershell
Start-AaLcrJob pipeline/k8s/hd-aa-lcr-v11-full.yaml `
  glm53-w4afp8-aa-lcr-v11-full-r1
```

The full result must contain 300 candidate rows and 300 valid judgments: 100
questions × 3 repeats. Full Jobs use `activeDeadlineSeconds: 86400` (24h)
because a from-scratch 300-unit run did not fit in 12h; the 0.6 campaign
only finished after a resume reset the Job clock. If a Job still dies on
deadline, sqlite is intact: relaunch the same run ID. The CLI is resumable;
rerun the same command to fill only missing candidate or judge rows, while
preserving the immutable fingerprint:

```powershell
Start-AaLcrJob pipeline/k8s/hd-aa-lcr-v11-full.yaml `
  glm53-w4afp8-aa-lcr-v11-full-r1
```

Do not change the run ID, endpoint identity, code revision, model, repeat
count, or judge contract when resuming. The CLI rejects checkpoints and result
directories that belong to another fingerprint, and refuses to summarize or
publish an incomplete run.

Publication prefers `renameat2(..., RENAME_NOREPLACE)`. CephFS returns
`EINVAL` for that flag; the runner then `os.rename`s only when the destination
is absent. To republish a finished checkpoint after a publish-only failure,
mount newer runner code but keep `AA_LCR_CODE_REVISION` equal to the
checkpoint fingerprint, extract the stored server snapshot as
`--endpoint-identity-file`, and run phase `summarize` against the same run ID.
Do not create a new run ID and do not replace an existing result directory.

Current low-severity limitation: every CLI phase, including `summarize`,
re-prepares the official pinned dataset before opening the checkpoint.
Therefore `summarize` currently needs dataset-network availability even though
all scoring primitives already exist in SQLite. This does not change the
published metric, but a future cleanup may load the existing checkpoint
without re-downloading the dataset.

## Published 0.6 public-methodology result

In-house W4AFP8, temperature 0.6, top_p 1.0, 100×3, Luna medium:

**Pass@1 71.33% (214/300).** Write-up:
[`docs/status/2026-09-14-glm53-aa-lcr-v11.md`](../status/2026-09-14-glm53-aa-lcr-v11.md).
Bundle:
`/mnt/cephfs/hoangduy/results/glm53-aa-lcr-v11/glm53-w4afp8-aa-lcr-v11-full-r1/`.

When the result is retained and no further evaluation needs the credential,
the operator may remove it:

```powershell
kubectl -n evaluation delete secret hd-openai-aa-judge
```
