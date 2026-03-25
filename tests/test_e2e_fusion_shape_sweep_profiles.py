###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import importlib.util
import sys
from pathlib import Path

PROFILE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "e2e_fusion_profiles.py"


def _load_profile_module():
    spec = importlib.util.spec_from_file_location("e2e_fusion_profiles", PROFILE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shape_sweep_uses_env_chunk_override():
    module = _load_profile_module()

    profiles = module.get_e2e_fusion_shape_sweep(env={"LUMEN_E2E_NUM_CHUNKS": "2"})

    assert profiles
    assert all(profile.num_chunks == 2 for profile in profiles)


def test_shape_sweep_preserves_profile_chunk_default_without_env_override():
    module = _load_profile_module()

    profiles = module.get_e2e_fusion_shape_sweep(env={})
    expected_chunks = [profile.num_chunks for profile in module._SHAPE_SWEEP_PROFILES]

    assert profiles
    assert [profile.num_chunks for profile in profiles] == expected_chunks
