# Shared mixed calibration for AWQ and GPTQ

The owner approved using a simple mixed dataset and explicitly requested implementation
on 2026-09-12, following the proposed generic/SWE-chat design. This supersedes the
entropy-selection experiment. Implementation and push to duy-branch are authorized.

Use one CPU-only preparation command to turn configurable weighted sources into an
immutable directory containing data.jsonl and manifest.json. Example composition:
50% UltraChat train_sft and 50% SWE-chat coding-agent sessions by actual tokens,
256 windows of 2048 tokens, seed 42. The ratio is a starting choice, not an optimum.
No model forward passes, difficulty scoring, evaluation suite or GPU runs.

Add optional calibration.prepared_dataset to the existing shared loader. Both native
W4AFP8 paths load and verify the prepared bundle, then use existing rank partitioning.
Legacy single-source behavior is unchanged when unset. No changes to recipes, native
export or source-rtn MTP. Source loading reuses Hugging Face datasets; message rendering
reuses the target tokenizer chat template. Existing upstream input_ids passthrough and
Intel AutoRound code-calibration mixing establish the reuse pattern. The remaining
integration is fixed-budget provenance and SWE-chat's turn-table/session boundary.

Preparation sources support HF IDs/configs/revisions or local JSON/Parquet files, explicit
messages/text/SWE-chat-turn formats, positive weights, deterministic window quotas.
SWE-chat turns are grouped by session and ordered by turn_number before formatting;
retain assistant text/reasoning and tool calls/results, exclude metadata, reject malformed
required fields. Sampling is session-based, without scoring, with one full window per eligible document/session. SWE-chat windows are selected
from substantive assistant decision/work turns across the session, retaining preceding
context; exclude metadata-only or tool-output-only stretches. Never default to the
session prefix. Use structural eligibility and seeded selection, not entropy scoring. Short documents are skipped, not padded;
insufficient eligible data is an error. This avoids letting a few long sessions dominate.
SWE-chat requires authorized access; there is no silent source fallback. Its Arrow
turn table can require substantial one-time CPU/disk preparation, outside quantization.

Manifest: source specifications and realized counts, selected document/session IDs and
window offsets, seed, tokenizer/template fingerprint, sequence length/count, content
SHA256. Bundles fail validation on mismatched tokenizer, corruption or incompatible
num_samples/max_seq_length. Local outputs are never overwritten. Reuse identical bundle
for AWQ and GPTQ. Preserve identifiers for subsequent evaluation exclusions; preparation
does not claim benchmark decontamination or Phala recipe reproduction.

Verification uses real small local datasets and a lightweight tokenizer fixture: exact
budgets, deterministic reproduction, session/tool reconstruction, distributed disjoint
coverage, legacy loader behavior, failure paths, both native configs. No cluster spend.
