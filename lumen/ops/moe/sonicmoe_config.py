###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Overlay gfx950 SonicMoE tile configs that AITER does not ship.

AITER resolves ``configs/<arch>/triton/moe/sonicmoe_bf16/DEFAULT.json`` inside
its own package. gfx950 has no such file, so the loader uses the gfx942
table. The Qwen3 tiles below are the ones selected by
``SONIC_MOE_USE_QWEN3_TUNED_GEMM=1`` (the docker default): they live in
``_specialized`` and are ignored when that variable is off.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_OVERLAY_PATH = Path(__file__).with_name("sonicmoe_gfx950.json")
_INSTALLED = False


def sonicmoe_config_overlay() -> dict:
    with _OVERLAY_PATH.open() as config_file:
        return json.load(config_file)


def apply_sonicmoe_config_overlay(base: dict, overlay: dict) -> dict:
    """Copy ``base`` and replace the specialized tiles named in ``overlay``."""
    merged = copy.deepcopy(base)
    for section, body in overlay.items():
        if not isinstance(body, dict):
            continue
        specialized = body.get("_specialized")
        if not specialized:
            continue
        destination = merged.setdefault(section, {})
        destination.setdefault("_specialized", {}).update(specialized)
    return merged


def install_sonicmoe_gfx950_config() -> None:
    """Prefer Lumen's gfx950 tiles when AITER has no gfx950 SonicMoE config."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        import aiter.ops.triton.utils.sonicmoe_config_utils as config_utils
        from aiter.ops.triton.utils._triton.arch_info import get_arch
        from aiter.ops.triton.utils.config_utils import resolve_config_dir
    except ImportError:
        return
    try:
        arch = get_arch()
    except Exception:
        logger.debug("SonicMoE gfx950 config overlay skipped: arch unavailable")
        return
    _INSTALLED = True
    if arch != "gfx950":
        return
    native = os.path.join(
        resolve_config_dir(
            "moe",
            "SONICMOE-BF16",
            backend="triton",
            arch="gfx950",
            cu_num=None,
        ),
        "DEFAULT.json",
    )
    if os.path.isfile(native):
        return

    original = config_utils.load_sonicmoe_configs
    overlay = sonicmoe_config_overlay()
    cached: dict = {}

    def load_sonicmoe_configs():
        if "config" not in cached:
            cached["config"] = apply_sonicmoe_config_overlay(original(), overlay)
            logger.info("Using Lumen gfx950 SonicMoE config %s", _OVERLAY_PATH)
        return cached["config"]

    config_utils.load_sonicmoe_configs = load_sonicmoe_configs
