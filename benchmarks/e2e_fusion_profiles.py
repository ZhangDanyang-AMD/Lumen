###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

DEFAULT_E2E_FUSION_PROFILE = "default"


@dataclass(frozen=True)
class E2EFusionProfile:
    name: str
    batch: int
    seq: int
    hidden: int
    ffn: int
    num_chunks: int

    @property
    def tokens(self) -> int:
        return self.batch * self.seq


_BASE_PROFILES = {
    DEFAULT_E2E_FUSION_PROFILE: E2EFusionProfile(
        name=DEFAULT_E2E_FUSION_PROFILE,
        batch=2,
        seq=2048,
        hidden=4096,
        ffn=14336,
        num_chunks=4,
    ),
    # Double token count to make communication more visible while keeping
    # the GEMM width unchanged.
    "backend_gap": E2EFusionProfile(
        name="backend_gap",
        batch=4,
        seq=2048,
        hidden=4096,
        ffn=14336,
        num_chunks=4,
    ),
    # Increase both token count and FFN width so the pipeline has more
    # compute to overlap and a better chance to amortize chunk overhead.
    "pipeline_gain": E2EFusionProfile(
        name="pipeline_gain",
        batch=4,
        seq=2048,
        hidden=4096,
        ffn=28672,
        num_chunks=4,
    ),
}


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if value is None:
        return default
    return int(value)


def _require_positive(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def get_e2e_fusion_profile(
    name: str | None = None,
    env: Mapping[str, str] | None = None,
) -> E2EFusionProfile:
    source_env = os.environ if env is None else env
    profile_name = name or source_env.get("LUMEN_E2E_PROFILE", DEFAULT_E2E_FUSION_PROFILE)

    if profile_name not in _BASE_PROFILES:
        valid = ", ".join(sorted(_BASE_PROFILES))
        raise ValueError(f"Unknown E2E fusion profile '{profile_name}'. Valid profiles: {valid}")

    base = _BASE_PROFILES[profile_name]
    return E2EFusionProfile(
        name=base.name,
        batch=_require_positive("LUMEN_E2E_BATCH", _env_int(source_env, "LUMEN_E2E_BATCH", base.batch)),
        seq=_require_positive("LUMEN_E2E_SEQ", _env_int(source_env, "LUMEN_E2E_SEQ", base.seq)),
        hidden=_require_positive("LUMEN_E2E_HIDDEN", _env_int(source_env, "LUMEN_E2E_HIDDEN", base.hidden)),
        ffn=_require_positive("LUMEN_E2E_FFN", _env_int(source_env, "LUMEN_E2E_FFN", base.ffn)),
        num_chunks=_require_positive(
            "LUMEN_E2E_NUM_CHUNKS",
            _env_int(source_env, "LUMEN_E2E_NUM_CHUNKS", base.num_chunks),
        ),
    )


def format_e2e_fusion_profile(profile: E2EFusionProfile) -> str:
    return (
        f"profile={profile.name}, B={profile.batch}, S={profile.seq}, "
        f"H={profile.hidden}, FFN={profile.ffn}, chunks={profile.num_chunks}"
    )
