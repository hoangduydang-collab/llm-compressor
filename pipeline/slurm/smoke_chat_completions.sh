#!/usr/bin/env bash
# Chat/completions smoke against a running ``vllm serve`` (Nemotron-style).
#
# Default matches the Nemotron Ultra curl that produced a clean short answer:
#   messages=[{"role":"user","content":"The capital of France is"}]
#   max_tokens=64, temperature=0.0
#   chat_template_kwargs.enable_thinking=false
#
#   bash pipeline/slurm/smoke_chat_completions.sh
#   MODEL=cyankiwi/MiniMax-M3-AWQ-INT4 bash pipeline/slurm/smoke_chat_completions.sh
#   PROMPT='What is 2+2?' MAX_TOKENS=32 bash pipeline/slurm/smoke_chat_completions.sh

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
MODEL="${MODEL:-cyankiwi/MiniMax-M3-AWQ-INT4}"
PROMPT="${PROMPT:-The capital of France is}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
# M3 / Nemotron reasoning models: disable thinking for a short usable smoke.
ENABLE_THINKING="${ENABLE_THINKING:-false}"
# Optional: MiniMax thinking_mode (enabled|disabled|adaptive). Empty = omit.
THINKING_MODE="${THINKING_MODE:-}"

echo "GET $BASE_URL/health"
curl -sf "$BASE_URL/health" >/dev/null
echo "health: ok"

# Build chat_template_kwargs JSON fragment.
if [[ -n "$THINKING_MODE" ]]; then
  CT_KWARGS=$(printf '{"enable_thinking": %s, "thinking_mode": "%s"}' \
    "$ENABLE_THINKING" "$THINKING_MODE")
else
  CT_KWARGS=$(printf '{"enable_thinking": %s}' "$ENABLE_THINKING")
fi

# Escape prompt for JSON (minimal: backslash + quotes).
PROMPT_JSON=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$PROMPT")

BODY=$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": $PROMPT_JSON}],
  "max_tokens": $MAX_TOKENS,
  "temperature": $TEMPERATURE,
  "chat_template_kwargs": $CT_KWARGS
}
EOF
)

echo "POST $BASE_URL/v1/chat/completions"
echo "  model=$MODEL max_tokens=$MAX_TOKENS temperature=$TEMPERATURE"
echo "  chat_template_kwargs=$CT_KWARGS"
echo "  prompt=$PROMPT_JSON"
echo ""

curl -s "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$BODY" | python3 -m json.tool
