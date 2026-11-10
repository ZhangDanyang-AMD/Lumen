#!/bin/bash
# Single source of every path this example uses. Source it, never hardcode.
#
# Defaults assume the container layout the README sets up: one host directory
# bind-mounted at /work. Override any variable before sourcing to fit a
# different layout, e.g.
#
#   WORK=/scratch/flydsl . env.sh
#
: "${WORK:=/work}"
: "${LUMEN_DIR:=$WORK/Lumen}"
: "${VEOMNI_DIR:=$WORK/VeOmni}"
: "${QWEN_IMAGE_DIR:=$WORK/models/Qwen-Image}"
: "${DATA_DIR:=$WORK/data/qwen_image_smoke}"
# Wan2.1-T2V-1.3B and the Tom-and-Jerry clips, for the video (T>1) comparison.
# Only needed by run_wan_10steps.sh; the image example does not touch them.
: "${WAN_DIR:=$WORK/models-extra/Wan2.1-T2V-1.3B}"
: "${WAN_DATA_DIR:=$WORK/data/tom-and-jerry-parquet}"
: "${OUT_DIR:=$WORK/outputs}"
: "${LOG_DIR:=$WORK/logs}"
: "${HF_HOME:=$WORK/hf-cache}"

# flydsl 0.3.2 installed to its own directory rather than over the one in the
# image; aiter's FlyDSL convolution needs it. See README section 2.5.
: "${FLYDSL_VERSION:=0.3.2}"
: "${FLYDSL_SIDECAR:=$WORK/pyenv/flydsl-$FLYDSL_VERSION}"
# Optional aiter checkout placed ahead of the image's aiter, e.g. an aiter main
# whose FlyDSL convolution supports gfx1250 (README section 3.11). Empty: the
# image's aiter.
: "${AITER_DIR:=}"

: "${EXAMPLE_DIR:=$LUMEN_DIR/examples/perf-veomni}"

# VeOmni commit this example is pinned to. Later commits may move the condition
# model or the DiT trainer hook points that train_dit_lumen.py wraps.
: "${VEOMNI_COMMIT:=573848a00fcd7329c2411346c6f4a983e9f67e3f}"

export WORK LUMEN_DIR VEOMNI_DIR QWEN_IMAGE_DIR DATA_DIR OUT_DIR LOG_DIR HF_HOME
export WAN_DIR WAN_DATA_DIR
export FLYDSL_SIDECAR FLYDSL_VERSION AITER_DIR EXAMPLE_DIR VEOMNI_COMMIT

# Put the sidecar ahead of the image's flydsl, AITER_DIR (if set) ahead of the
# image's aiter, and the cloned Lumen ahead of any Lumen already in the image.
export LUMEN_PYTHONPATH="$FLYDSL_SIDECAR:${AITER_DIR:+$AITER_DIR:}$LUMEN_DIR"
