###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import pytest
import torch
import torch.nn as nn


class TestFP8ParamWiring:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_enable_fp8_params_registers_weights(self):
        from lumen.models.megatron import _find_scaling_manager
        from lumen.quantize import disable, enable
        from lumen.quantize.config import QuantConfig

        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 64),
        ).cuda()

        config = QuantConfig()
        enable(model, config=config)

        sm = _find_scaling_manager(model)
        assert sm is not None

        sm.enable_fp8_params(model)
        assert sm.num_fp8_params > 0

        disable(model)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_optimizer_hook_marks_stale(self):
        from lumen.models.megatron import _find_scaling_manager
        from lumen.quantize import disable, enable
        from lumen.quantize.config import QuantConfig

        model = nn.Sequential(nn.Linear(64, 64)).cuda()
        config = QuantConfig()
        enable(model, config=config)

        sm = _find_scaling_manager(model)
        sm.enable_fp8_params(model)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        sm.register_fp8_optimizer_hook(optimizer)

        x = torch.randn(4, 64, device="cuda")
        out = model(x)
        out.sum().backward()
        optimizer.step()

        assert sm._fp8_param_stale, "FP8 params should be stale after optimizer.step()"

        disable(model)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_lazy_requant_after_stale(self):
        """After marking stale and calling quantize, cache is refreshed."""
        from lumen.models.megatron import _find_scaling_manager
        from lumen.quantize import disable, enable
        from lumen.quantize.config import QuantConfig

        model = nn.Sequential(nn.Linear(64, 64)).cuda()
        config = QuantConfig()
        enable(model, config=config)

        sm = _find_scaling_manager(model)
        sm.enable_fp8_params(model)

        sm.quantize_fp8_params()
        assert not sm._fp8_param_stale

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        sm.register_fp8_optimizer_hook(optimizer)
        x = torch.randn(4, 64, device="cuda")
        model(x).sum().backward()
        optimizer.step()
        assert sm._fp8_param_stale

        sm.quantize_fp8_params()
        assert not sm._fp8_param_stale

        disable(model)
