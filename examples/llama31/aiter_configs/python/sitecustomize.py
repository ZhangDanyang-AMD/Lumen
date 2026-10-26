"""Compatibility shims for the gfx1250 AITER image's FlyDSL runtime."""

try:
    import flydsl.runtime.device as _flydsl_device

    if not hasattr(_flydsl_device, "check_smem_capacity"):

        def _check_smem_capacity(size, arch):
            limit = 128 * 1024 if str(arch).startswith("gfx1250") else 64 * 1024
            if int(size) > limit:
                raise ValueError(
                    f"shared-memory requirement {size} exceeds {limit} on {arch}"
                )

        _flydsl_device.check_smem_capacity = _check_smem_capacity
except ImportError:
    pass
