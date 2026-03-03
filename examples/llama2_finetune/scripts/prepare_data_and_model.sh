#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

git clone https://github.com/ROCm/flash-attention/ flash_attention \
    && cd flash_attention && git checkout b3c68b169824a58df339e4fcb0ad5e5a3e4d4327 \
    && git submodule update --init --recursive \
    && PYTORCH_ROCM_ARCH='gfx950' GPU_ARCHS="gfx950" MAX_JOBS=64 pip install --no-build-isolation .

cd /workspace/code/

python ${SCRIPT_DIR}/download_dataset.py --data_dir /data/gov_report  # download dataset
python ${SCRIPT_DIR}/download_model.py --model_dir /data/model  # download model checkpoint used for initialization; could take up to 30 minutes
python ${SCRIPT_DIR}/convert_dataset.py --data_dir /data/gov_report
python ${SCRIPT_DIR}/convert_model.py --input_name_or_path=/data/model --output_path=/data/model/llama2-70b.nemo
cd /data/model && find . -type f ! -name 'llama2-70b.nemo' -exec rm -f {} + && tar -xvf llama2-70b.nemo
cd /data/model && rm -rf llama2-70b.nemo
mv /data/gov_report /data/data
