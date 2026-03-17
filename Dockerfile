ARG BASE_IMAGE=rocm/7.0:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0_rc1
FROM ${BASE_IMAGE}

WORKDIR /workspace

# System build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Copy project source
COPY . /workspace/Lumen

# AITER (editable from third_party)
RUN cd /workspace/Lumen/third_party/aiter && pip install -e .

# torchao (quantization reference)
RUN pip install torchao>=0.8

# Lumen + test dependencies
RUN cd /workspace/Lumen && pip install -e ".[dev]"
RUN pip install -r /workspace/Lumen/requirements.txt

# Environment
ENV PYTHONPATH="/workspace/Lumen:${PYTHONPATH}"
ENV HIP_VISIBLE_DEVICES=0

WORKDIR /workspace/Lumen

CMD ["pytest", "tests/ops/", "-v", "--tb=short"]
