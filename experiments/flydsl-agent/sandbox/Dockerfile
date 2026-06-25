ARG BASE_IMAGE=rocm/vllm-dev:nightly_main_20260603
FROM ${BASE_IMAGE}

# FlyDSL + PyTorch + ROCm are already in base image.
# Add sandbox verification and benchmark scripts.
COPY verify.py /workspace/verify.py
COPY benchmark.py /workspace/benchmark.py

ENV PYTHONDONTWRITEBYTECODE=1
ENV FLYDSL_RUNTIME_ENABLE_CACHE=0
WORKDIR /workspace
