#!/usr/bin/env bash
# =============================================================================
# Profile: Qwen3-30B-A3B-Instruct-2507, in-house W2A16 (AutoRound DDP, int2 g128
# sym, compressed-tensors pack-quantized), served via vLLM+Humming (serve-sub4).
# CANDIDATE arm; BASELINE_REF resolves the BF16 profile in this directory.
# Serve (1xH100, same node as baseline):
#   CUDA_VISIBLE_DEVICES=2 vllm serve /mnt/nfs/hoangduy/projects/llm-compressor/artifacts/Qwen3-30B-A3B-Instruct-2507-autoround-W2A16-g128-ddp8 \
#     --served-model-name qwen3-30b-w2a16 --port 8411 \
#     --max-model-len 8192 --max-logprobs 20
# (serve-sub4 env required: PYTHONPATH=humming-main-site, cu13 LD_LIBRARY_PATH,
#  ninja on PATH, NFS HUMMING_CACHE_DIR — see llm-compressor evidence NOTES.)
# =============================================================================

MODEL_NAME="${MODEL_NAME:-qwen3-30b-w2a16}"
MODEL_PATH="${MODEL_PATH:-/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/Qwen3-30B-A3B-Instruct-2507-autoround-W2A16-g128-ddp8}"
MODEL_TOKENIZER="${MODEL_TOKENIZER:-$MODEL_PATH}"

MODEL_CAPS="${MODEL_CAPS:-nonreasoning}"

ENDPOINT_PORT="${ENDPOINT_PORT:-8411}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:${ENDPOINT_PORT}/v1}"
BASE_URL="${BASE_URL:-http://localhost:${ENDPOINT_PORT}}"
ENGINE="${ENGINE:-vllm}"
ENGINE_VERSION="${ENGINE_VERSION:-0.26.0-serve-sub4+pr48918+humming-main}"
GPU_SKU="${GPU_SKU:-H100-80GB}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_LABEL="${GPU_LABEL:-1xH100}"
PRECISION="${PRECISION:-W2A16}"
QUANT_RECIPE="${QUANT_RECIPE:-autoround-w2a16-g128-ddp8-iters200-n512}"

NONREASONING_TEMP="${NONREASONING_TEMP:-0}"
MAX_OUTPUT_NONREASONING="${MAX_OUTPUT_NONREASONING:-4096}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-8192}"

GENERAL_NUM_CONCURRENT="${GENERAL_NUM_CONCURRENT:-16}"
GENERAL_TASKS="${GENERAL_TASKS:-gpqa_diamond_cot_zeroshot ifeval}"
GENERAL_REQUEST_TIMEOUT_S="${GENERAL_REQUEST_TIMEOUT_S:-7200}"

TOP_LOGPROBS_K="${TOP_LOGPROBS_K:-20}"

RELIABILITY_METRICS="${RELIABILITY_METRICS-}"  # empty: this repo's run_ab does not wire probe datasets to run_quality (validator rejects)
RELIABILITY_MODE="${RELIABILITY_MODE:-unconstrained}"
RELIABILITY_TOOL_CALL_SOURCE="${RELIABILITY_TOOL_CALL_SOURCE:-probe}"

BASELINE_REF="${BASELINE_REF-qwen3-30b-a3b-2507-bf16}"
