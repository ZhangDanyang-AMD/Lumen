ARG BASE_IMAGE=rocm/7.0:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0_rc1
FROM ${BASE_IMAGE}

WORKDIR /workspace

# System build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ninja-build cmake \
        libopenmpi-dev libibverbs-dev rdma-core libpci-dev && \
    rm -rf /var/lib/apt/lists/*

COPY . /workspace/Lumen

# AITER — AMD-optimised kernels (Lumen patches operators with AITER backends).
RUN cd /workspace/Lumen/third_party/aiter && \
    PREBUILD_KERNELS=1 pip install -e .

# torchao (quantization reference).
RUN pip install torchao>=0.8

# Lumen (editable install).
RUN cd /workspace/Lumen && pip install -e ".[dev]"
RUN pip install -r /workspace/Lumen/requirements.txt

# CPT-specific dependencies: HuggingFace Transformers + PEFT for Qwen2 + LoRA.
# Pin transformers<4.50 to avoid torchao internal API breakage with the
# torchao version shipped in the ROCm base image.
RUN pip install \
    "transformers>=4.44,<4.50" \
    "peft>=0.12" \
    "datasets>=2.16" \
    "accelerate>=0.30" \
    "huggingface_hub>=0.20" \
    sentencepiece \
    protobuf

ENV PYTHONPATH="/workspace/Lumen/third_party/aiter:${PYTHONPATH:-}"
ENV PYTHONPATH="/workspace/Lumen:${PYTHONPATH}"
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# AMD MI GPU performance tuning.
ENV NCCL_MIN_P2P_NCHANNELS=32
ENV NCCL_MIN_CTAS=32
ENV NCCL_NCHANNELS_PER_NET_PEER=32
ENV NCCL_NVLS_ENABLE=0
ENV TORCH_NCCL_AVOID_RECORD_STREAMS=1
ENV USE_HIPBLASLT=1
ENV TORCH_BLAS_PREFER_HIPBLASLT=1
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV OMP_NUM_THREADS=1

ENV TRANSFORMERS_OFFLINE=0
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /workspace/Lumen/experiments/flydsl-agent/cpt
