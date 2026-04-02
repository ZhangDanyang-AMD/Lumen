###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import pytest
import torch

from lumen.quantize.comm_tensor import FP8CommTensor


class TestFP8CommTensor:
    def _make(self, rows=16, cols=64):
        data = torch.randn(rows, cols, dtype=torch.bfloat16)
        return FP8CommTensor(data, fp8_dtype=torch.float8_e4m3fnuz)

    def test_creation(self):
        t = self._make()
        assert t.shape == (16, 64)
        assert t.dtype == torch.bfloat16

    def test_view_preserves_subclass(self):
        t = self._make()
        v = t.view(4, 4, 64)
        assert isinstance(v, FP8CommTensor)

    def test_clone_preserves_subclass(self):
        t = self._make()
        c = t.clone()
        assert isinstance(c, FP8CommTensor)

    def test_unknown_op_unwraps(self):
        t = self._make()
        result = t + 1.0
        assert not isinstance(result, FP8CommTensor)
        assert result.dtype == torch.bfloat16

    def test_flatten_unflatten_roundtrip(self):
        t = self._make()
        names, metadata = t.__tensor_flatten__()
        inner = {"_data": t._data}
        t2 = FP8CommTensor.__tensor_unflatten__(inner, metadata, t.shape, t.stride())
        assert isinstance(t2, FP8CommTensor)
        assert torch.equal(t._data, t2._data)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
    def test_pre_post_all_gather_roundtrip(self):
        data = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
        t = FP8CommTensor(data, fp8_dtype=torch.float8_e4m3fnuz)
        tensors, meta = FP8CommTensor.fsdp_pre_all_gather(t)
        result = FP8CommTensor.fsdp_post_all_gather(tensors, meta, torch.bfloat16)
        torch.testing.assert_close(result, data, atol=0.1, rtol=0.05)
