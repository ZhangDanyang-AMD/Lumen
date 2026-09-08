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
: "${OUT_DIR:=$WORK/outputs}"
: "${LOG_DIR:=$WORK/logs}"
: "${HF_HOME:=$WORK/hf-cache}"

# flydsl 0.3.2 installed to its own directory rather than over the one in the
# image. See README "Why a sidecar flydsl".
: "${FLYDSL_SIDECAR:=$WORK/pyenv/flydsl-0.3.2}"
: "${FLYDSL_VERSION:=0.3.2}"

: "${EXAMPLE_DIR:=$LUMEN_DIR/examples/qwen-image-flydsl-conv}"

# VeOmni commit this example is pinned to. Later commits may move the condition
# model or the DiT trainer hook points that train_dit_lumen.py wraps.
: "${VEOMNI_COMMIT:=573848a00fcd7329c2411346c6f4a983e9f67e3f}"

export WORK LUMEN_DIR VEOMNI_DIR QWEN_IMAGE_DIR DATA_DIR OUT_DIR LOG_DIR HF_HOME
export FLYDSL_SIDECAR FLYDSL_VERSION EXAMPLE_DIR VEOMNI_COMMIT

# Put the sidecar ahead of the image's flydsl, and the cloned Lumen ahead of any
# Lumen already installed in the image.
export LUMEN_PYTHONPATH="$FLYDSL_SIDECAR:$LUMEN_DIR"
