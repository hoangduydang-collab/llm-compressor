#!/usr/bin/env bash
# =============================================================================
# Profile: Qwen3-30B-A3B-Instruct-2507, BF16 baseline, self-hosted vLLM.
# Baseline arm for the in-house sub-4-bit (W2A16 AutoRound) A/B.
# Serve (2xH100, same node as candidate):
#   CUDA_VISIBLE_DEVICES=0,1 vllm serve /mnt/nfs/hoangduy/hf_assets/Qwen/Qwen3-30B-A3B-Instruct-2507 \
#     --served-model-name qwen3-30b-bf16 --port 8410 --tensor-parallel-size 2 \
#     --max-model-len 8192 --max-logprobs 20
# =============================================================================

MODEL_NAME="${MODEL_NAME:-qwen3-30b-bf16}"
MODEL_PATH="${MODEL_PATH:-/mnt/nfs/hoangduy/hf_assets/Qwen/Qwen3-30B-A3B-Instruct-2507}"
MODEL_TOKENIZER="${MODEL_TOKENIZER:-$MODEL_PATH}"

MODEL_CAPS="${MODEL_CAPS:-nonreasoning}"   # Instruct-2507 = non-thinking variant

ENDPOINT_PORT="${ENDPOINT_PORT:-8410}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:${ENDPOINT_PORT}/v1}"
BASE_URL="${BASE_URL:-http://localhost:${ENDPOINT_PORT}}"
ENGINE="${ENGINE:-vllm}"
ENGINE_VERSION="${ENGINE_VERSION:-0.26.0-serve-sub4}"
GPU_SKU="${GPU_SKU:-H100-80GB}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_LABEL="${GPU_LABEL:-2xH100}"
PRECISION="${PRECISION:-BF16}"

NONREASONING_TEMP="${NONREASONING_TEMP:-0}"
MAX_OUTPUT_NONREASONING="${MAX_OUTPUT_NONREASONING:-4096}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-8192}"

GENERAL_NUM_CONCURRENT="${GENERAL_NUM_CONCURRENT:-16}"
GENERAL_TASKS="${GENERAL_TASKS:-gpqa_diamond_cot_zeroshot ifeval}"
GENERAL_REQUEST_TIMEOUT_S="${GENERAL_REQUEST_TIMEOUT_S:-7200}"

# distribution suite (flip-rate / top-k agreement / truncated-KL) in comparison mode
TOP_LOGPROBS_K="${TOP_LOGPROBS_K:-20}"

RELIABILITY_METRICS="${RELIABILITY_METRICS-}"  # empty: this repo's run_ab does not wire probe datasets to run_quality (validator rejects)
RELIABILITY_MODE="${RELIABILITY_MODE:-unconstrained}"
RELIABILITY_TOOL_CALL_SOURCE="${RELIABILITY_TOOL_CALL_SOURCE:-probe}"

# baseline arm: no BASELINE_REF
