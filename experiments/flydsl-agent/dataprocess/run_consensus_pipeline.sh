#!/bin/bash
# Multi-model consensus annotation pipeline (Plan §4.4)
#
# This script orchestrates the full annotation pipeline:
#   1. Rule-based expert annotation (already done)
#   2. Multi-model AI annotation (requires API keys)
#   3. Consensus voting + merge
#   4. Apply to manifest + rebuild datasets
#
# Usage:
#   # Set API keys for each provider
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   export OPENAI_API_KEY="sk-..."
#   export GOOGLE_API_KEY="AIza..."
#   export DEEPSEEK_API_KEY="sk-..."
#
#   # Run the full pipeline
#   bash run_consensus_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="/home/danyzhan/flydsl-agent-dataset"
METADATA_DIR="${DATASET_DIR}/metadata"
PROMPTS="${METADATA_DIR}/annotation_prompts.jsonl"

echo "=== Multi-Model Consensus Annotation Pipeline ==="
echo "Prompts: ${PROMPTS}"
echo ""

# Step 0: Rule-based (already done, skip if exists)
RESP_RULE="${METADATA_DIR}/responses_rulebased.jsonl"
if [ ! -f "${RESP_RULE}" ]; then
    echo "[Step 0] Running rule-based expert annotation..."
    python3 "${SCRIPT_DIR}/batch_annotate_cursor.py" "${PROMPTS}" "${RESP_RULE}"
else
    echo "[Step 0] Rule-based responses already exist: ${RESP_RULE}"
fi

# Step 1: Claude annotation
RESP_CLAUDE="${METADATA_DIR}/responses_claude.jsonl"
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "[Step 1] Running Claude annotation..."
    python3 "${SCRIPT_DIR}/consensus_annotate.py" annotate \
        --prompts "${PROMPTS}" \
        --output "${RESP_CLAUDE}" \
        --api-key "${ANTHROPIC_API_KEY}" \
        --model "claude-sonnet-4-20250514" \
        --model-name "claude" \
        --provider "anthropic" \
        --delay 0.5
else
    echo "[Step 1] SKIP: ANTHROPIC_API_KEY not set"
fi

# Step 2: GPT-4o annotation
RESP_GPT="${METADATA_DIR}/responses_gpt4o.jsonl"
if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "[Step 2] Running GPT-4o annotation..."
    python3 "${SCRIPT_DIR}/consensus_annotate.py" annotate \
        --prompts "${PROMPTS}" \
        --output "${RESP_GPT}" \
        --api-base "https://api.openai.com/v1" \
        --api-key "${OPENAI_API_KEY}" \
        --model "gpt-4o" \
        --model-name "gpt4o" \
        --provider "openai" \
        --delay 0.5
else
    echo "[Step 2] SKIP: OPENAI_API_KEY not set"
fi

# Step 3: Gemini annotation
RESP_GEMINI="${METADATA_DIR}/responses_gemini.jsonl"
if [ -n "${GOOGLE_API_KEY:-}" ]; then
    echo "[Step 3] Running Gemini annotation..."
    python3 "${SCRIPT_DIR}/consensus_annotate.py" annotate \
        --prompts "${PROMPTS}" \
        --output "${RESP_GEMINI}" \
        --api-key "${GOOGLE_API_KEY}" \
        --model "gemini-2.5-pro" \
        --model-name "gemini" \
        --provider "google" \
        --delay 0.5
else
    echo "[Step 3] SKIP: GOOGLE_API_KEY not set"
fi

# Step 4: DeepSeek annotation
RESP_DS="${METADATA_DIR}/responses_deepseek.jsonl"
if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
    echo "[Step 4] Running DeepSeek annotation..."
    python3 "${SCRIPT_DIR}/consensus_annotate.py" annotate \
        --prompts "${PROMPTS}" \
        --output "${RESP_DS}" \
        --api-base "https://api.deepseek.com/v1" \
        --api-key "${DEEPSEEK_API_KEY}" \
        --model "deepseek-chat" \
        --model-name "deepseek" \
        --provider "openai" \
        --delay 0.3
else
    echo "[Step 4] SKIP: DEEPSEEK_API_KEY not set"
fi

# Step 5: Consensus merge
echo ""
echo "[Step 5] Running consensus merge..."
RESP_FILES=""
for f in "${RESP_RULE}" "${RESP_CLAUDE}" "${RESP_GPT}" "${RESP_GEMINI}" "${RESP_DS}"; do
    if [ -f "$f" ]; then
        RESP_FILES="${RESP_FILES} $f"
    fi
done

# Also include Cursor-generated responses if available
for f in "${METADATA_DIR}"/responses_gpt_batch*.jsonl "${METADATA_DIR}"/responses_claude_batch*.jsonl; do
    if [ -f "$f" ]; then
        RESP_FILES="${RESP_FILES} $f"
    fi
done

CONSENSUS="${METADATA_DIR}/consensus_annotations.jsonl"
if [ -n "${RESP_FILES}" ]; then
    python3 "${SCRIPT_DIR}/consensus_annotate.py" consensus \
        --responses ${RESP_FILES} \
        --output "${CONSENSUS}"
else
    echo "ERROR: No response files found!"
    exit 1
fi

# Step 6: Apply to manifest
echo ""
echo "[Step 6] Applying consensus to manifest..."
ANNOTATED="${METADATA_DIR}/annotated_manifest.json"
python3 "${SCRIPT_DIR}/consensus_annotate.py" apply \
    --manifest "${METADATA_DIR}/graded_manifest.json" \
    --consensus "${CONSENSUS}" \
    --output "${ANNOTATED}"

echo ""
echo "=== Pipeline Complete ==="
echo "Consensus annotations: ${CONSENSUS}"
echo "Annotated manifest:    ${ANNOTATED}"
echo ""
echo "Next steps:"
echo "  1. Review entries with needs_human_review in ${ANNOTATED}"
echo "  2. Rebuild datasets with: python3 process_all_v2.py (using annotated manifest)"
echo "  3. Upload to HuggingFace Hub"
