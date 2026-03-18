###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import argparse
from unittest.mock import patch

import torch.nn as nn


class TestFSDP2CLIArgs:

    def test_fsdp_version_default(self):
        from lumen.models.fsdp import add_common_fsdp_args

        parser = argparse.ArgumentParser()
        add_common_fsdp_args(parser)
        args = parser.parse_args([])
        assert args.fsdp_version == 1

    def test_fsdp_version_2(self):
        from lumen.models.fsdp import add_common_fsdp_args

        parser = argparse.ArgumentParser()
        add_common_fsdp_args(parser)
        args = parser.parse_args(["--fsdp-version", "2"])
        assert args.fsdp_version == 2


class TestApplyFSDP2:

    def test_function_exists(self):
        from lumen.models.fsdp import apply_fsdp2

        assert callable(apply_fsdp2)

    def test_returns_model(self):
        from lumen.models.fsdp import apply_fsdp2

        model = nn.Linear(8, 4)
        args = argparse.Namespace(linear_fp8=False)

        # Mock fully_shard since it requires distributed setup
        with patch("torch.distributed.fsdp.fully_shard", side_effect=lambda m, **kw: m):
            result = apply_fsdp2(model, args)
            assert result is model
