#!/bin/sh
mkdir -p /models /var/log

# Default model — AWQ-4bit Qwen3.6-35B-A3B, 4-bit, ~20 GB on disk, fits one 32 GB card.
REPO="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
SERVED_NAME="qwen3.6-35b"

export HF_TOKEN=""
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pre-warm the HF cache (vLLM will lazy-download otherwise but we want the manager
# to report progress). Skip if already cached.
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
  "max_model_len": 131072,
  "max_num_batched_tokens": 4096,
  "max_num_seqs": 2,
  "gpu_memory_utilization": 0.94,
  "kv_cache_dtype": "fp8",
  "cpu_offload_gb": 0,
  "enable_prefix_caching": true,
  "reasoning_parser": "qwen3",
  "tool_call_parser": "qwen3_coder",
  "enable_auto_tool_choice": true
}
EOF

cat > /root/onstart.sh << 'EOF'
#!/bin/sh
export HF_HOME=/models
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model manager on port 5000
python3 /app/model_manager_vllm.py >> /var/log/model-manager.log 2>&1 &

# Single GPU — AWQ 4-bit Qwen3.6-35B fits in ~20 GB, ~10 GB left for KV cache.
# --max-model-len 131072: 128K context. Qwen3.6 hybrid Mamba keeps KV tiny (~700 MiB/seq at fp8).
# --max-num-seqs 2: cap concurrent requests (controls KV cache slots).
# --gpu-memory-utilization 0.94: 32 GB Ada has enough headroom for tight util.
# --kv-cache-dtype fp8: halves KV memory at <1% quality cost.
# --enable-prefix-caching: dramatically speeds up shared-prefix prompts.
CUDA_VISIBLE_DEVICES=0 vllm serve cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --served-model-name qwen3.6-35b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 2 \
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
