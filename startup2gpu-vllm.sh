#!/bin/sh
mkdir -p /models /var/log

# Dual GPU vLLM — tensor parallelism splits weights across both cards.
# Unlike llama.cpp, vLLM's tensor parallel IS supported for MoE.
REPO="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
SERVED_NAME="qwen3.6-35b"

export HF_TOKEN="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"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SNAPSHOT_DIR="/models/hub/models--$(echo $REPO | tr / -)/snapshots"
if [ ! -d "$SNAPSHOT_DIR" ] || [ -z "$(ls -A $SNAPSHOT_DIR 2>/dev/null)" ]; then
    echo "[$(date)] Downloading $REPO..." >> /var/log/startup.log
    huggingface-cli download "$REPO" \
        --token "$HF_TOKEN" \
        --cache-dir /models \
        >> /var/log/startup.log 2>&1
fi

cat > /root/vllm_config.json << EOF
{
  "repo_id": "$REPO",
  "served_model_name": "$SERVED_NAME",
  "max_model_len": 262144,
  "max_num_batched_tokens": 8192,
  "max_num_seqs": 4,
  "gpu_memory_utilization": 0.94,
  "kv_cache_dtype": "fp8",
  "cpu_offload_gb": 0,
  "enable_prefix_caching": true,
  "reasoning_parser": "qwen3",
  "tool_call_parser": "qwen3_coder",
  "enable_auto_tool_choice": true,
  "tensor_parallel_size": 2
}
EOF

cat > /root/onstart.sh << 'EOF'
#!/bin/sh
export HF_HOME=/models
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model manager on port 5000
python3 /app/model_manager_vllm.py >> /var/log/model-manager.log 2>&1 &

# Dual GPU via tensor parallelism. Each GPU holds half the weights.
# --tensor-parallel-size 2: required for multi-GPU.
# --max-model-len 262144: 256K context — Qwen3.6's native trained ceiling.
#   KV cache shards across both GPUs, so per-GPU footprint stays modest.
# --max-num-seqs 4: handle more concurrent users.
# vLLM TP works for MoE — unlike llama.cpp which falls back to pipeline parallel.
vllm serve cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --served-model-name qwen3.6-35b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.94 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --trust-remote-code >> /var/log/vllm-server.log 2>&1
EOF

chmod +x /root/onstart.sh
s6-svc -r /var/run/s6/services/onstart
