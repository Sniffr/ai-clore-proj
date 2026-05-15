FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt update -y && DEBIAN_FRONTEND=noninteractive apt install -y \
    wget curl aria2 python3 python3-pip \
    git cmake build-essential libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# libcuda.so.1 stub: needed at link time on the build machine (no GPU driver).
# The real libcuda.so.1 is injected at runtime by nvidia-container-runtime.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Build llama.cpp from source against CUDA 12.4.
# Backward compat guarantees this binary runs on any driver >= 550 (12.4, 12.8, 13.x, ...).
# Architectures: 70=V100 75=T4/RTX20 80=A100 86=RTX30/A5000 89=RTX40/L40 90=H100
RUN git clone --depth=1 https://github.com/ggml-org/llama.cpp /llama.cpp \
    && cd /llama.cpp \
    && cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
    && cmake --build build -j$(nproc) --target llama-server \
    && mkdir -p /app \
    && find build/bin -maxdepth 1 \( -name "llama-server" -o -name "*.so*" \) -exec cp {} /app/ \; \
    && rm -rf /llama.cpp

ENV LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN pip install huggingface_hub[hf_transfer] flask --break-system-packages
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN mkdir -p /models

COPY model_manager.py /app/model_manager.py

EXPOSE 5000dockerfile
