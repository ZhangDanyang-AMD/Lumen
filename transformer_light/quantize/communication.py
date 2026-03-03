###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import torch
import torch.distributed as dist


class QuantAllGather:
    """Quantized all-gather: quantize -> send -> receive -> dequantize."""

    def __init__(self, group, use_mori=False):
        self.group = group
        self.use_mori = use_mori  # True for inter-node RDMA

    def __call__(self, tensor, scale):
        if self.use_mori:
            import mori.shmem
            mori.shmem.allgather_fp8(tensor, scale, self.group)
        else:
            world_size = dist.get_world_size(self.group)
            output_list = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(output_list, tensor, group=self.group)
        return output_list, scale


# Backward-compat alias
FP8AllGather = QuantAllGather
