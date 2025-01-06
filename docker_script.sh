#!/bin/bash

apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update
apt-get install -y
apt-get -y install cudnn9-cuda-12


echo "Step.1 - installing fastapi uvicorn onnxruntime-genai-cuda"
pip install fastapi uvicorn onnxruntime-genai-cuda

pip install -y huggingface_hub

echo "Diagnosis:"

conda env list

echo "pip list:"
pip list

nvcc --version
nvidia-smi

echo "find libcudnn"
find / -name libcudnn.so.*

git clone https://github.com/apsonawane/turnkeyml-cuda.git
cd turnkeyml-cuda
conda create -n tk-llm python=3.10
source /opt/conda/etc/profile.d/conda.sh
conda activate tk-llm
pip install -e .[llm-oga-cuda]

# export HF_HOME="/build/oga_models/hf_version/"
# echo "HF_HOME: $HF_HOME"

huggingface-cli download $MODEL_NAME --local-dir /build/oga_models/hf_version/

lemonade -i $MODEL_NAME --cache-dir "/build" oga-load --device cuda --dtype int4 accuracy-mmlu --tests management
