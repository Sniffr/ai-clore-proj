FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN apt update -y && \
    DEBIAN_FRONTEND=noninteractive apt install -y wget curl aria2 python3 python3-pip && \
    pip install huggingface_hub[hf_transfer] --break-system-packages && \
    rm -rf /var/lib/apt/lists/*

ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN mkdir -p /models

# Pre-download model during build
RUN aria2c -x 16 -s 16 -k 1M -d /models \
    -o Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"