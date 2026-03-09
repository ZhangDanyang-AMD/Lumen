ARG BASE_IMAGE=rocm/7.0:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0_rc1
FROM ${BASE_IMAGE}

WORKDIR /workspace

RUN pip install pybind11
RUN pip install ninja
RUN pip install packaging
RUN /usr/bin/python3 -m pip install pyYAML

# ---------------------------------------------------------------------------
# Install library dependencies
# ---------------------------------------------------------------------------
WORKDIR /workspace/deps

# Megatron-LM-AMD (for Megatron backend)
ARG MEGATRON_COMMIT=8dd45f5a51378ec1ee7937dee3c20d8626df4763
RUN git clone --recursive https://github.com/ROCm/Megatron-LM.git megatron_lm
RUN pip uninstall -y megatron-core
RUN cd megatron_lm && git checkout ${MEGATRON_COMMIT} \
    && pip install -e . && cd megatron/core/datasets && make

ENV PYTHONPATH="/workspace/deps/megatron_lm:${PYTHONPATH:-}"

# AITER — AMD-optimised kernels (CK attention backend, FP8 quant, hipBLASLt)
RUN git clone --recursive https://github.com/ROCm/aiter.git aiter
RUN cd aiter && pip install -e . --force-reinstall

ENV PYTHONPATH="/workspace/deps/megatron_lm:/workspace/deps/aiter:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
COPY examples/llama31/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Install Transformer Light
# ---------------------------------------------------------------------------
WORKDIR /workspace
COPY . /workspace/Transformer_Light
RUN cd /workspace/Transformer_Light && pip install -e ".[all]"

ENV PYTHONPATH="/workspace/Transformer_Light:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# AMD MI GPU performance tuning (from perf_env.sh)
# ---------------------------------------------------------------------------
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

# MLPerf / training defaults
ENV TRANSFORMERS_OFFLINE=0
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /workspace/Transformer_Light/examples/llama31
