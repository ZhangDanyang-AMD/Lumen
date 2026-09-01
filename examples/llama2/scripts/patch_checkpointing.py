#!/usr/bin/env python3
"""Deprecated — use ``python3 -m lumen.patches <megatron-root> --tag lora``."""

from __future__ import annotations

import sys

from _patch_lora_registry_shim import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
