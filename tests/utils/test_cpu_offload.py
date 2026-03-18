###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import torch
import torch.nn as nn


class TestCPUOffloadManager:

    def test_construction(self):
        from lumen.utils.cpu_offload import CPUOffloadManager

        mgr = CPUOffloadManager(enabled=True)
        assert mgr.enabled
        assert mgr.pin_memory

    def test_register_hooks(self):
        from lumen.utils.cpu_offload import CPUOffloadManager

        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        mgr = CPUOffloadManager(enabled=True)
        count = mgr.register_hooks(model)
        assert count >= 2  # at least the two linear layers

    def test_register_hooks_disabled(self):
        from lumen.utils.cpu_offload import CPUOffloadManager

        model = nn.Sequential(nn.Linear(8, 16))
        mgr = CPUOffloadManager(enabled=False)
        count = mgr.register_hooks(model)
        assert count == 0

    def test_remove_hooks(self):
        from lumen.utils.cpu_offload import CPUOffloadManager

        model = nn.Sequential(nn.Linear(8, 16))
        mgr = CPUOffloadManager(enabled=True)
        mgr.register_hooks(model)
        mgr.remove_hooks()
        assert len(mgr._hooks) == 0

    def test_memory_saved_initial(self):
        from lumen.utils.cpu_offload import CPUOffloadManager

        mgr = CPUOffloadManager()
        assert mgr.memory_saved_bytes() == 0


class TestCPUOffloadContext:

    def test_context_manager(self):
        from lumen.utils.cpu_offload import lumen_cpu_offload_context

        model = nn.Sequential(nn.Linear(8, 4))
        x = torch.randn(2, 8)

        with lumen_cpu_offload_context(model, enabled=True):
            out = model(x)
            assert out.shape == (2, 4)

    def test_disabled_context(self):
        from lumen.utils.cpu_offload import lumen_cpu_offload_context

        model = nn.Sequential(nn.Linear(8, 4))
        x = torch.randn(2, 8)

        with lumen_cpu_offload_context(model, enabled=False):
            out = model(x)
            assert out.shape == (2, 4)

    def test_hooks_cleaned_up(self):
        from lumen.utils.cpu_offload import lumen_cpu_offload_context

        model = nn.Sequential(nn.Linear(8, 4))

        with lumen_cpu_offload_context(model, enabled=True) as mgr:
            hook_count = len(mgr._hooks)
            assert hook_count > 0

        # After context, hooks should be removed
        assert len(mgr._hooks) == 0
