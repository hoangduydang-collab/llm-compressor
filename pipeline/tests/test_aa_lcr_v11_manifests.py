import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
K8S = ROOT / "pipeline" / "k8s"
RUNBOOK = ROOT / "docs" / "runbooks" / "aa-lcr-v11.md"
STAGE = K8S / "stage-aa-lcr-v11.yaml"
CANARY = K8S / "hd-aa-lcr-v11-canary.yaml"
FULL = K8S / "hd-aa-lcr-v11-full.yaml"
EVAL_MANIFESTS = (CANARY, FULL)


def load_container(path: Path) -> dict[str, object]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document["spec"]["template"]["spec"]["containers"][0]


def test_manifests_never_embed_openai_key():
    for path in EVAL_MANIFESTS:
        text = path.read_text(encoding="utf-8")
        assert "sk-" not in text
        assert "OPENAI_API_KEY" in text
        key_env = next(
            item
            for item in load_container(path)["env"]
            if item["name"] == "OPENAI_API_KEY"
        )
        assert key_env["valueFrom"]["secretKeyRef"] == {
            "name": "hd-openai-aa-judge",
            "key": "OPENAI_API_KEY",
        }


def test_eval_jobs_request_no_gpu():
    for path in EVAL_MANIFESTS:
        resources = load_container(path)["resources"]
        assert "nvidia.com/gpu" not in resources.get("requests", {})
        assert "nvidia.com/gpu" not in resources.get("limits", {})


def test_jobs_use_pinned_service_ceph_and_safe_restart_policy():
    for path in (STAGE, *EVAL_MANIFESTS):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        text = path.read_text(encoding="utf-8")
        assert document["spec"]["backoffLimit"] == 0
        assert "model-cache-shared" in text
        assert "AA_LCR_CODE_CONFIGMAP" in text
        assert "glm-5-3-w4afp8-sglang:30000/v1/chat/completions" in text
        assert "/mnt/cephfs/hoangduy/runlogs" in text


def test_jobs_are_content_addressed_templates_without_fixed_names():
    for path in (STAGE, *EVAL_MANIFESTS):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        metadata = document["metadata"]
        assert "name" not in metadata
        assert metadata["generateName"].startswith("hd-")
        assert "REPLACE_WITH_GIT_COMMIT" not in path.read_text(encoding="utf-8")
        container = load_container(path)
        revision = next(
            item
            for item in container.get("env", [])
            if item["name"] == "AA_LCR_CODE_REVISION"
        )
        assert revision["valueFrom"]["configMapKeyRef"] == {
            "name": "AA_LCR_CODE_CONFIGMAP",
            "key": "AA_LCR_CODE_REVISION",
        }


def test_runtime_rendered_job_has_no_unresolved_code_tokens():
    for path in (STAGE, *EVAL_MANIFESTS):
        rendered = path.read_text(encoding="utf-8").replace(
            "AA_LCR_CODE_CONFIGMAP", "hd-aa-lcr-v11-code-0123456789ab"
        )
        assert "AA_LCR_CODE_CONFIGMAP" not in rendered
        assert "REPLACE_WITH_GIT_COMMIT" not in rendered


def test_canary_and_full_have_pinned_population_and_concurrency():
    canary = CANARY.read_text(encoding="utf-8")
    full = FULL.read_text(encoding="utf-8")

    assert "--limit 1" in canary
    assert "--repeats 1" in canary
    assert "--canary" in canary
    assert "--judge-preflight" in canary
    assert "--limit 100" in full
    assert "--repeats 3" in full
    assert "--candidate-concurrency 2" in full


def test_runbook_renders_immutable_configmaps_and_resumable_jobs():
    text = RUNBOOK.read_text(encoding="utf-8")

    assert "git diff --quiet" in text
    assert "git diff --cached --quiet" in text
    assert "git rev-parse HEAD" in text
    assert "AA_LCR_CODE_CONFIGMAP" in text
    assert "immutable = $true" in text
    assert "data = @{ AA_LCR_CODE_REVISION = $revision }" in text
    assert "binaryData" in text
    assert "create -f -" in text
    assert "generateName" in text
    assert 'get pods -l "aa-lcr-run-id=$runId" -o json' in text
    assert "--field-selector" not in text
    assert "$LASTEXITCODE -ne 0" in text
    assert "ConvertFrom-Json -ErrorAction Stop" in text
    assert "@('Pending', 'Running') -contains $_.status.phase" in text
    assert "httpx2" in text
    assert "https://pypi.org/pypi/openai/3.8.0/json" in text
    assert "https://pypi.org/pypi/httpx2/2.12.0/json" in text
    assert (
        "/mnt/cephfs/hoangduy/aa-lcr-v11-work/"
        "glm53-w4afp8-aa-lcr-v11-canary-r1/run.sqlite"
    ) in text
    assert "The canary intentionally does not publish a headline bundle." in text


def test_runbook_rejects_nonterminal_same_run_job_before_creation():
    text = RUNBOOK.read_text(encoding="utf-8")

    jobs_query = 'get jobs -l "aa-lcr-run-id=$runId" -o json'
    create_job = 'create -f - `'
    assert jobs_query in text
    assert "ConvertFrom-Json -ErrorAction Stop" in text
    assert re.search(
        r"\$jobs\.items\s*\|\s*Where-Object\s*\{(?s:.*?)"
        r"@\(.*?Complete.*?Failed.*?\).*?status.*?True",
        text,
    )
    assert text.index(jobs_query) < text.index(create_job)


def test_runbook_waits_for_created_pod_before_following_job_logs():
    text = RUNBOOK.read_text(encoding="utf-8")

    created = 'Write-Host "Created $jobName for $runId"'
    pod_creation_wait = (
        'kubectl -n evaluation wait --for=create pod -l "job-name=$jobName" '
        "--timeout=180s"
    )
    logs = 'kubectl -n evaluation logs -f "job/$jobName"'
    assert pod_creation_wait in text
    assert text.index(created) < text.index(pod_creation_wait) < text.index(logs)
