###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

from benchmarks.e2e_fusion_profiles import (
    fp8_param_e2e_input_shape,
    fp8_param_e2e_weight_shape,
    fp8_param_pipeline_layer_shapes,
    get_e2e_fusion_profile,
    wgrad_delay_dims,
)


def test_wgrad_delay_dims_follow_default_profile():
    profile = get_e2e_fusion_profile()

    assert wgrad_delay_dims(profile) == (4096, 4096, 14336)


def test_wgrad_delay_dims_follow_backend_gap_profile():
    profile = get_e2e_fusion_profile("backend_gap")

    assert wgrad_delay_dims(profile) == (8192, 4096, 14336)


def test_fp8_param_e2e_shapes_follow_pipeline_gain_profile():
    profile = get_e2e_fusion_profile("pipeline_gain")

    assert fp8_param_e2e_weight_shape(profile) == (28672, 4096)
    assert fp8_param_e2e_input_shape(profile) == (4, 2048, 4096)


def test_fp8_param_pipeline_shapes_follow_profile():
    profile = get_e2e_fusion_profile("pipeline_gain")

    assert fp8_param_pipeline_layer_shapes(profile) == [
        (28672, 4096),
        (28672, 4096),
        (4096, 28672),
        (4096, 4096),
    ]
