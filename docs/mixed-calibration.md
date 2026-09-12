# Mixed calibration for AWQ and GPTQ

Prepare a generic/coding-agent token bundle once on CPU and reuse it in either
quantization path. The example uses **50% UltraChat and 50% SWE-chat by tokens**:
128 windows from each source, 2,048 tokens per window, 524,288 tokens total.
The proportions are configurable; they are a starting choice, not a measured optimum.
SWE-chat already includes coding and agent interactions.

The existing generic-only configs still work. Native W4AFP8 export, block-FP8
preparation, AWQ/GPTQ settings and source-RTN MTP assembly are unchanged. This
feature adds no difficulty scoring, model forward passes or extra calibration
windows. It does not authorize a new GPU run or a dataset-comparison experiment.

## Prepare once on CPU

Use the same Python environment as quantization (Hugging Face `datasets` and
`transformers` are already dependencies). From the repository root:

```bash
python -m pipeline.prepare_calibration \
  --config pipeline/configs/calibration/glm53_generic_agentic_mix.yaml \
  --tokenizer /mnt/cephfs/.hf-cache/models--zai-org--GLM-5.3-BF16/snapshots/304b8051cfb2b260b61ce0cbe330e02a98e73639 \
  --output /mnt/cephfs/hoangduy/calibration/glm53-generic-agentic-256x2048-s42
```

Alternatively, load only the tokenizer from the Hub with
`--tokenizer zai-org/GLM-5.3-BF16 --tokenizer-revision 304b8051cfb2b260b61ce0cbe330e02a98e73639`.
No model weights are loaded. The output directory must be new.

SWE-chat is gated: use an HF identity that already has access, or supply an
existing authorized local dataset snapshot. The command does not accept terms,
request access or substitute a different dataset. The checked-in source revisions
were resolved from public Hub metadata on 2026-09-12.

Preparation uses CPU, RAM, disk and potentially network access. In particular,
SWE-chat's normalized conversation table is large and is loaded using HF's Arrow
cache; this is not a download of only 128 records. Session indexing uses lightweight
columns, then reads selected session contents. Run preparation once before allocating
quantization GPUs, retain the bundle and cache, and reuse them. Preparation cost
is separate from the existing quantization `dataset_preparation` phase, which still
includes bundle validation and rank partitioning. No full-source performance
measurement is claimed here.

## Use with either method

Two opt-in configs retain their corresponding native recipes and use the same
bundle directory:

- `pipeline/configs/glm53_ep_gptq_w4afp8_full_mixed.yaml`
- `pipeline/configs/glm53_distributed_w4afp8_awq_full_mixed.yaml`

They have distinct run names, output roots and offload roots from the generic-only
configs. In the executor's existing approved launch, substitute the relevant mixed
config; all launch/resource prerequisites still apply. Rancher launcher/mount
resolution remains with the executor under the owner's earlier instruction.

For any existing AWQ/GPTQ config, the only required data-path addition is:

```yaml
calibration:
  prepared_dataset: /absolute/path/to/prepared-bundle
  num_samples: 256
  max_seq_length: 2048
```

The existing CLI also supports
`--set calibration.prepared_dataset=/absolute/path/to/prepared-bundle`.
When set, the bundle supplies the tokens; legacy `dataset_id`, `dataset_split`
and `dataset_data_files` are not used. Unset it (or set YAML `null`) to restore
legacy source loading. Keep the configured count and length equal to the bundle.
The preparation seed fixes sample selection/order; the consumer does not resample it.

## Change sources or proportions

Copy the preparation YAML and change its `sources` list. Positive weights are
normalized into integer window quotas using largest remainders. Since every window
has the same length, window proportions equal token proportions, subject to rounding.
A positive source that receives no window is rejected. Use one source for a
prepared generic-only bundle, or add another compatible source for a three-way mix.

Supported formats are `messages` (chat messages), `text` (already rendered text)
and `swe_chat` (the published normalized conversation-turn table). The `column`
setting selects the messages/text column; `id_column` preserves a document ID.
A local SWE-chat source can replace the Hub entry:

```yaml
- name: coding_agentic
  weight: 0.5
  dataset_id: parquet
  dataset_split: train
  dataset_data_files: /absolute/path/to/conversations.parquet
  format: swe_chat
  id_column: session_id
```

Remove `dataset_config_name` and `dataset_revision` when switching that entry to
local Parquet. A local chat JSONL source instead uses `dataset_id: json`, a
`dataset_data_files` path and `format: messages`. Local SWE-chat data must use the
published normalized columns; raw Claude/Codex/Gemini transcript files are not that
schema. Normalized tool results already contain the publisher's truncation.

## Meaningful windows across SWE-chat sessions

The adapter groups turns by `session_id` and orders them by `turn_number`, keeping
assistant reasoning, responses, tool calls and tool results. It excludes metadata
rather than treating each database row as a separate calibration example. Rendering
uses the target tokenizer's chat template and structured tool arguments. SWE-chat
window selection requires token offset mappings (supported by GLM's fast tokenizer).

Sampling considers substantive assistant work across the session, with preceding
context. Anchor spans are located in the final rendered session and mapped to its
tokens; each selected window must overlap the chosen assistant payload. The manifest
records the merged message's constituent turns and the selected field/span, rather
than attributing the entire merged message to its last original turn. It does not
always take the first 2,048 tokens and does not select windows
consisting only of metadata or tool output. These are inexpensive structural rules,
not an assertion that the window is difficult or that the code is correct.

Each source contributes at most one window per eligible document/session. Short
examples are skipped rather than padded or joined across sessions. If there are too
few eligible examples for a source's quota, preparation fails rather than silently
reweighting or duplicating data. One-window sampling limits long-session dominance.

## Reproducibility and limits

Keep `data.jsonl` and `manifest.json` together. The manifest records the source
specifications, realized mixture, selected document/session IDs and window positions,
seed, tokenizer/template identity and content hashes. Quantization verifies the bundle,
including tokenizer identity and configured dimensions, then uses the existing
nonoverlapping rank partitioning. Per-rank token hashes continue to be written in
`calibration_partition*.json`.

A different tokenizer/template or modified token file is an error; rebuild a new
bundle rather than editing the manifest. AWQ and GPTQ should consume the exact same
bundle. Preserve selected session IDs for exclusion from any later evaluation.
This preparation step does not automatically decontaminate benchmark questions,
reproduce Phala's exact private sample selection, or prove a quality improvement.
The historical generic loader truncates rows and can have a smaller actual token
budget; it is not a matched-token experimental control for this bundle.

## Existing resources used

- [HF Datasets](https://huggingface.co/docs/datasets/loading) supplies Hub/local
  dataset loading, Arrow storage and indexed selection.
- [SWE-chat's published schema](https://huggingface.co/datasets/SALT-NLP/SWE-chat)
  defines the session and tool fields used by the adapter.
- [Phala GLM-5.3](https://huggingface.co/PhalaCloud/GLM-5.3-W4AFP8) provides the
  direct precedent for coding-agent AWQ calibration.
- [Intel AutoRound calibration loading](https://github.com/intel/auto-round/blob/main/auto_round/calib_dataset.py)
  supplies an established weighted-mixture/tokenized-data pattern; the existing
  llm-compressor dataset path also supports already-tokenized input. This integration
  adds provenance checks and the SWE-chat session adapter around existing libraries.

## Local validation

The calibration/config/distributed regression set passes 38 tests. The existing
Torch dependency emits 14 `torch.jit.script_method` deprecation warnings. A separate
CPU smoke with the real pinned GLM tokenizer and synthetic generic/agent traces
produced identical AWQ/GPTQ token hashes and selected agent windows thousands of
tokens into sessions. Reproducible inputs and results are in
[the CPU evidence directory](../results/mixed-calibration/20260912-cpu/README.md).
The full gated SWE-chat download, full quantization and model-quality evaluation
were not run.
