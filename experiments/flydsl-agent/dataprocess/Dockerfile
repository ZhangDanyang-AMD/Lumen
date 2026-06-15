FROM python:3.11-slim

LABEL description="Data processing pipeline for FlyDSL Code Agent training"
LABEL maintainer="danyzhan"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pyyaml

WORKDIR /workspace

COPY taxonomy.yaml /workspace/dataprocess/taxonomy.yaml
COPY generate_manifest.py /workspace/dataprocess/generate_manifest.py
COPY extract_cpt_data.py /workspace/dataprocess/extract_cpt_data.py
COPY generate_sft_data.py /workspace/dataprocess/generate_sft_data.py
COPY prepare_rl_specs.py /workspace/dataprocess/prepare_rl_specs.py
COPY validate_dataset.py /workspace/dataprocess/validate_dataset.py
COPY export_for_training.py /workspace/dataprocess/export_for_training.py
COPY run_pipeline.py /workspace/dataprocess/run_pipeline.py

RUN mkdir -p /workspace/dataprocess/output

ENTRYPOINT ["python", "/workspace/dataprocess/run_pipeline.py"]
CMD ["--repo", "/workspace/aiter", "--output-dir", "/workspace/dataprocess/output"]
