import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
K8S = ROOT / "pipeline" / "k8s"
RUNBOOK = ROOT / "docs" / "runbooks" / "aa-lcr-v11.md"
STAGE = K8S / "stage-aa-lcr-v11.yaml"
CANARY = K8S / "hd-aa-lcr-v11-canary.yaml"
FULL = K8S / "hd-aa-lcr-v11-full.yaml"
CANARY_T1 = K8S / "hd-aa-lcr-v11-canary-t1.yaml"
FULL_T1 = K8S / "hd-aa-lcr-v11-full-t1.yaml"
CANARY_T1_2X = K8S / "hd-aa-lcr-v11-canary-t1-2x.yaml"
FULL_T1_2X = K8S / "hd-aa-lcr-v11-full-t1-2x.yaml"
EVAL_MANIFESTS = (CANARY, FULL, CANARY_T1, FULL_T1, CANARY_T1_2X, FULL_T1_2X)
TIKTOKEN_CACHE_DIR = "/mnt/cephfs/hoangduy/cache/aa-lcr-v11-tiktoken"
CL100K_CACHE_FILE = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
CL100K_BPE_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
TIKTOKEN_READY_MARKER = f"{TIKTOKEN_CACHE_DIR}/cl100k_base.ready"


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


def test_base_jobs_pin_aa_generic_sampling_explicitly():
    # The module default moved to Z.ai's recommended 1.0/0.95 (the lab-override
    # branch of AA's rule). These run ids published 0.6/1.0 results, so they must
    # state it rather than inherit a default that no longer means that.
    for path in (CANARY, FULL):
        text = path.read_text(encoding="utf-8")
        assert "--candidate-temperature 0.6" in text
        assert "--candidate-top-p 1.0" in text


def test_full_jobs_use_24h_deadline():
    # 12h was enough only because the 0.6 campaign resumed into a second Job.
    # A from-scratch 300-unit run needs the extra headroom.
    for path in (FULL, FULL_T1):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert document["spec"]["activeDeadlineSeconds"] == 86400
    for path in (CANARY, CANARY_T1, CANARY_T1_2X):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert document["spec"]["activeDeadlineSeconds"] == 7200


def test_doubled_cap_full_job_gets_48h():
    # Twice the output cap is roughly twice the generated-token volume, and the
    # admission gate serialises the long tail that the cap raise exists to reach.
    document = yaml.safe_load(FULL_T1_2X.read_text(encoding="utf-8"))

    assert document["spec"]["activeDeadlineSeconds"] == 172800


def test_doubled_cap_jobs_pin_new_run_ids_cap_and_adaptive_concurrency():
    canary = CANARY_T1_2X.read_text(encoding="utf-8")
    full = FULL_T1_2X.read_text(encoding="utf-8")

    for text in (canary, full):
        assert "--candidate-temperature 1.0" in text
        assert "--candidate-top-p 0.95" in text
        assert "--candidate-max-tokens 262144" in text
        # Ceiling 2 with the gate free to hold at 1 while an attempt runs long.
        assert "--candidate-concurrency 2" in text
        assert "--candidate-long-attempt-seconds 900" in text
    assert "--canary" in canary
    assert "--limit 100" in full
    assert "--repeats 3" in full
    # A raised cap is a different measurement, so it gets its own run ids and
    # cannot land on the 131k checkpoints or their published result directories.
    assert "--run-id glm53-w4afp8-aa-lcr-v11-canary-t1p95-2x-r1" in canary
    assert "--run-id glm53-w4afp8-aa-lcr-v11-full-t1p95-2x-r1" in full


def test_phala_sampling_jobs_pin_new_run_ids_and_1_0_0_95():
    canary = CANARY_T1.read_text(encoding="utf-8")
    full = FULL_T1.read_text(encoding="utf-8")

    assert "glm53-w4afp8-aa-lcr-v11-canary-t1p95-r1" in canary
    assert "glm53-w4afp8-aa-lcr-v11-full-t1p95-r1" in full
    assert "--candidate-temperature 1.0" in canary
    assert "--candidate-temperature 1.0" in full
    assert "--candidate-top-p 0.95" in canary
    assert "--candidate-top-p 0.95" in full
    assert "--canary" in canary
    assert "--limit 100" in full
    assert "--repeats 3" in full
    assert "glm53-w4afp8-aa-lcr-v11-full-r1" not in full
    assert "glm53-w4afp8-aa-lcr-v11-canary-r1" not in canary


def test_manifests_stage_and_verify_shared_tiktoken_vocabulary_cache():
    for path in (STAGE, *EVAL_MANIFESTS):
        container = load_container(path)
        environment = {item["name"]: item.get("value") for item in container["env"]}
        assert {
            name: environment[name]
            for name in (
                "TIKTOKEN_CACHE_DIR",
                "CL100K_CACHE_FILE",
                "CL100K_BPE_SHA256",
                "TIKTOKEN_READY_MARKER",
            )
        } == {
            "TIKTOKEN_CACHE_DIR": TIKTOKEN_CACHE_DIR,
            "CL100K_CACHE_FILE": CL100K_CACHE_FILE,
            "CL100K_BPE_SHA256": CL100K_BPE_SHA256,
            "TIKTOKEN_READY_MARKER": TIKTOKEN_READY_MARKER,
        }

    stage = STAGE.read_text(encoding="utf-8")
    assert 'mkdir -p "$TIKTOKEN_CACHE_DIR"' in stage
    assert 'tiktoken.get_encoding("cl100k_base")' in stage
    assert 'test -s "$TIKTOKEN_CACHE_DIR/$CL100K_CACHE_FILE"' in stage
    assert 'sha256sum "$TIKTOKEN_CACHE_DIR/$CL100K_CACHE_FILE"' in stage
    assert 'mv "$MARKER_TMP" "$TIKTOKEN_READY_MARKER"' in stage
    assert 'tiktoken-cache-sha256.txt' in stage

    for path in EVAL_MANIFESTS:
        text = path.read_text(encoding="utf-8")
        assert 'test -d "$TIKTOKEN_CACHE_DIR"' in text
        assert 'test -s "$TIKTOKEN_CACHE_DIR/$CL100K_CACHE_FILE"' in text
        assert 'test "$(cat "$TIKTOKEN_READY_MARKER")"' in text
        assert 'sha256sum "$TIKTOKEN_CACHE_DIR/$CL100K_CACHE_FILE"' in text
        assert 'find "$TIKTOKEN_CACHE_DIR" -type f -size +0 -print -quit' not in text
        assert text.index('test -d "$TIKTOKEN_CACHE_DIR"') < text.index(
            '"$VENV/bin/python" -m pipeline.aa_lcr_v11 run'
        )


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
    assert "activeDeadlineSeconds: 86400" in text
    assert TIKTOKEN_CACHE_DIR in text
    assert 'tiktoken.get_encoding("cl100k_base")' in text
    assert "tiktoken-cache-sha256.txt" in text
    assert "fail closed" in text


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
