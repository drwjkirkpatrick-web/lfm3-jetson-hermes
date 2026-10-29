#!/bin/bash
# LFM3-1.7B on Jetson 8GB — verified launch profile (Oct 2026)
# Usage: ./launch_lfm3.sh [q4km|qad] [port]
#
# Verified profile:
#   - 64K f16 KV (22.5 KB/token → 1.5 GB; clears Hermes agent 64K minimum)
#   - vendor sampling: temp 0.1 / top-k 50 / repeat-penalty 1.1
#     (greedy temp 0.0 makes the model loop inside its think block on
#      open-ended prompts — do NOT benchmark or serve at temp 0)
#   - max_tokens BUDGET: this model plans whole answers in reasoning_content
#     (~4.6K thinking tokens for a full HTML page) before emitting content.
#     One-shot generative work needs max_tokens >= 8000; tool calls fit in
#     512 (thinking bounded at 61-428 tok observed).
#   - thinking-off template override (model re-opens its own think block
#     anyway — bounded — and it lands cleanly in reasoning_content)
#   - ngl 99 full offload, NO_PINNED + UNIFIED_MEMORY required
QUANT="${1:-q4km}"
PORT="${2:-8090}"
DIR="$(cd "$(dirname "$0")" && pwd)"

case "$QUANT" in
  q4km) MODEL="$HOME/models/new-zoo/LFM3-1.7B-Q4_K_M.gguf";   ALIAS="lfm3-1.7b-q4km" ;;
  qad)  MODEL="$HOME/models/new-zoo/LFM3-1.7B-QAD-Q4_0.gguf"; ALIAS="lfm3-1.7b-qad"  ;;
  *) echo "usage: $0 [q4km|qad] [port]"; exit 1 ;;
esac

# pre-flight: GUI off, no stale server (bracket pattern avoids self-match)
sudo systemctl stop gdm.service 2>/dev/null || true
pkill -9 -f 'llama-serv[e]r' 2>/dev/null || true
sleep 8

GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_NO_PINNED=1 \
nohup "$HOME/llama.cpp/build/bin/llama-server" \
  -m "$MODEL" --alias "$ALIAS" \
  --host 127.0.0.1 --port "$PORT" \
  -ngl 99 -c 65536 -ctk f16 -ctv f16 \
  -b 512 -ub 512 -fa on \
  --jinja --fit off -np 1 -t 6 \
  --temp 0.1 --top-k 50 --repeat-penalty 1.1 \
  --chat-template-file "$DIR/templates/lfm3-nothink.jinja" \
  > /tmp/llama-server-lfm3.log 2>&1 &

echo "launched $ALIAS on :$PORT (pid $!) — log: /tmp/llama-server-lfm3.log"
echo "verify: sleep 20 && grep -c 'failed to allocate' /tmp/llama-server-lfm3.log  # must be 0"