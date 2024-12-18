#!/bin/bash

#source activate base
#conda activate ptca

apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update
apt-get install -y
apt-get -y install cudnn9-cuda-12


echo "Step.1 - installing fastapi uvicorn onnxruntime-genai-cuda"
pip install fastapi uvicorn onnxruntime-genai-cuda

# echo "Step.2 - azure-cli"
# pip install --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge

# echo "Step.3 - azure-ai"
# pip install azure-ai-evaluation --upgrade

# echo "Step.4 - promptflow-azure"
# pip install promptflow-azure --upgrade

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
echo "Activating conda environment"
conda activate tk-llm
echo "Installing turnkeyllm"
pip install -e .[llm-oga-cuda]
echo "Installed turnkeyllm"

echo "Running lemonade command"
lemonade -i microsoft/Phi-3.5-mini-instruct --cache-dir "/root/" oga-load --device cuda --dtype int4 llm-prompt -p "Hello, my thoughts are"

# echo "Copying the model to ort_src"
ls -la "/root/oga_models/microsoft_phi-3.5-mini-4k-instruct/"

cp -r "/root/oga_models/microsoft_phi-3.5-mini-4k-instruct/" "/ort_src"

ls "/ort_src/"
