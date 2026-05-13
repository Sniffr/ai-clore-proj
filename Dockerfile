FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt update -y && DEBIAN_FRONTEND=noninteractive apt install -y \
    wget curl aria2 python3 python3-pip \
    git cmake build-essential libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp from source against CUDA 12.4.
# Backward compat guarantees this binary runs on any driver >= 550 (12.4, 12.8, 13.x, ...).
# sm_86 targets RTX 3090 and A5000 (Ampere). Add 80;89;90 for A100/RTX40xx/H100.
RUN git clone --depth=1 https://github.com/ggml-org/llama.cpp /llama.cpp \
    && cd /llama.cpp \
    && cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=ON \
    && cmake --build build -j$(nproc) --target llama-server \
    && mkdir -p /app \
    && find build -name "llama-server" -exec cp {} /app/ \; \
    && find build -name "*.so" -exec cp {} /app/ \; \
    && rm -rf /llama.cpp

ENV LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN pip install huggingface_hub[hf_transfer] --break-system-packages
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN mkdir -p /models

# Pre-download model during build
RUN aria2c -x 16 -s 16 -k 1M -d /models \
    -o Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
