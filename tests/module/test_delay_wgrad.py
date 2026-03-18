###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import torch


class TestDeferredWgrad:

    def test_defer_and_execute(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        dw = _DeferredWgrad()
        weight = torch.nn.Parameter(torch.randn(4, 4))
        weight.grad = torch.zeros(4, 4)
        grad_val = torch.ones(4, 4)

        dw.defer(weight, lambda: grad_val)
        assert dw.has_pending

        dw.execute()
        assert not dw.has_pending
        torch.testing.assert_close(weight.grad, grad_val)

    def test_defer_into_main_grad(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        dw = _DeferredWgrad()
        weight = torch.nn.Parameter(torch.randn(4, 4))
        weight.main_grad = torch.zeros(4, 4)
        grad_val = torch.ones(4, 4) * 2

        dw.defer(weight, lambda: grad_val)
        dw.execute()
        torch.testing.assert_close(weight.main_grad, grad_val)

    def test_no_pending_execute_is_noop(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        dw = _DeferredWgrad()
        dw.execute()  # should not raise
        assert not dw.has_pending

    def test_accumulates_on_existing_grad(self):
        from lumen.modules.parallel_linear import _DeferredWgrad

        dw = _DeferredWgrad()
        weight = torch.nn.Parameter(torch.randn(4, 4))
        weight.grad = torch.ones(4, 4)

        dw.defer(weight, lambda: torch.ones(4, 4) * 3)
        dw.execute()
        torch.testing.assert_close(weight.grad, torch.ones(4, 4) * 4)
