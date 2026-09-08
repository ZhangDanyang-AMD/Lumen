###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""FlyDSL convolution kernels.

Import the entry point lazily from ``conv3d_implicit`` rather than re-exporting
it here: importing this package must not require ``flydsl`` to be installed,
since ``lumen.ops.conv`` probes for it before deciding to use it.
"""
