ARG BASE_IMAGE=rocm/7.0:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0_rc1
FROM ${BASE_IMAGE}

WORKDIR /workspace

# System build tools + mori build dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ninja-build cmake \
        libopenmpi-dev libibverbs-dev rdma-core libpci-dev && \
    rm -rf /var/lib/apt/lists/*

COPY . /workspace/Lumen

# AITER — AMD-optimised kernels (CK attention backend, FP8 quant, hipBLASLt).
RUN cd /workspace/Lumen/third_party/aiter && \
    PREBUILD_KERNELS=1 pip install -e .

# mori — SDMA communication library (editable from third_party).
RUN cd /workspace/Lumen/third_party/mori && \
    git submodule update --init --recursive && \
    pip install setuptools-scm && \
    pip install -e .

# torchao (quantization reference).
RUN pip install torchao>=0.8

# Megatron-LM-AMD (for Megatron backend).
ARG MEGATRON_COMMIT=8dd45f5a51378ec1ee7937dee3c20d8626df4763
RUN git clone --recursive https://github.com/ROCm/Megatron-LM.git megatron_lm
RUN pip uninstall -y megatron-core
RUN cd megatron_lm && git checkout ${MEGATRON_COMMIT} \
    && pip install -e .  -U --force-reinstall --no-deps \
    && cd megatron/core/datasets && make

ENV PYTHONPATH="/workspace/megatron_lm:"
ENV PYTHONPATH="/workspace/Lumen/third_party/mori:${PYTHONPATH:-}"
ENV PYTHONPATH="/workspace/Lumen/third_party/aiter:${PYTHONPATH:-}"

# Lumen + example dependencies.
RUN cd /workspace/Lumen && pip install -e ".[dev]"
RUN pip install -r /workspace/Lumen/requirements.txt
RUN pip install -r /workspace/Lumen/examples/llama2/requirements.txt

ENV PYTHONPATH="/workspace/Lumen:${PYTHONPATH}"
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Launcher/config decides whether --use-sdma is active at runtime.
ENV MORI_ENABLE_SDMA=0
RUN python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')" \
        > /etc/ld.so.conf.d/torch.conf && ldconfig

# AMD MI GPU performance tuning (from perf_env.sh).
ENV NCCL_MIN_P2P_NCHANNELS=32
ENV NCCL_MIN_CTAS=32
ENV NCCL_NCHANNELS_PER_NET_PEER=32
ENV NCCL_NVLS_ENABLE=0
ENV TORCH_NCCL_AVOID_RECORD_STREAMS=1
ENV USE_HIPBLASLT=1
ENV TORCH_BLAS_PREFER_HIPBLASLT=1
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV PYTORCH_TUNABLEOP_ENABLED=1
ENV PYTORCH_TUNABLEOP_FILENAME=tunableop_results.csv
ENV OMP_NUM_THREADS=1

ENV TRANSFORMERS_OFFLINE=0
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /workspace/Lumen/examples/llama2
