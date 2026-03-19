ARG BASE_IMAGE=rocm/7.0:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0_rc1
FROM ${BASE_IMAGE}

WORKDIR /workspace

# System build tools + mori build dependencies:
#   cmake/ninja    — mori CMake build
#   libopenmpi-dev — MPI (mori bootstrap)
#   libibverbs-dev, rdma-core — RDMA verbs + mlx5 provider (mori transport)
#   libpci-dev     — PCI topology detection
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ninja-build cmake \
        libopenmpi-dev libibverbs-dev rdma-core libpci-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy project source
COPY . /workspace/Lumen

# AITER (editable from third_party)
RUN cd /workspace/Lumen/third_party/aiter && pip install -e .

# mori — SDMA communication library (editable from third_party)
# Must init mori's own submodules (spdlog, msgpack-c) before cmake build.
RUN cd /workspace/Lumen/third_party/mori && \
    git submodule update --init --recursive && \
    pip install setuptools-scm && \
    pip install -e .

# torchao (quantization reference)
RUN pip install torchao>=0.8

# Megatron-LM-AMD (for Megatron backend)
ARG MEGATRON_COMMIT=8dd45f5a51378ec1ee7937dee3c20d8626df4763
RUN git clone --recursive https://github.com/ROCm/Megatron-LM.git megatron_lm
RUN pip uninstall -y megatron-core
RUN cd megatron_lm && git checkout ${MEGATRON_COMMIT} \
    && pip install -e .  -U --force-reinstall --no-deps \
    && cd megatron/core/datasets && make

ENV PYTHONPATH="/workspace/megatron_lm:${PYTHONPATH:-}"

# Lumen + test dependencies
RUN cd /workspace/Lumen && pip install -e ".[dev]"
RUN pip install -r /workspace/Lumen/requirements.txt

# Environment
ENV PYTHONPATH="/workspace/Lumen:${PYTHONPATH}"
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV MORI_ENABLE_SDMA=1

WORKDIR /workspace/Lumen

CMD ["pytest", "tests/ops/", "-v", "--tb=short"]
