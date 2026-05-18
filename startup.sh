#!/bin/sh
mkdir -p /models
MODEL="/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
URL="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
export HF_TOKEN=""

if [ ! -f "$MODEL" ]; then
    echo "[$(date)] Starting model download..." >> /var/log/startup.log

    echo "[$(date)] Trying aria2c..." >> /var/log/startup.log
    aria2c -x 16 -s 16 -k 1M -d /models -o Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
        --header="Authorization: Bearer $HF_TOKEN" \
        "$URL" >> /var/log/startup.log 2>&1

    if [ ! -f "$MODEL" ]; then
        echo "[$(date)] aria2c failed, trying wget..." >> /var/log/startup.log
        wget --header="Authorization: Bearer $HF_TOKEN" \
            -O "$MODEL" "$URL" >> /var/log/startup.log 2>&1
    fi

    if [ ! -f "$MODEL" ]; then
        echo "[$(date)] wget failed, trying curl..." >> /var/log/startup.log
        curl -L -H "Authorization: Bearer $HF_TOKEN" \
            -o "$MODEL" "$URL" >> /var/log/startup.log 2>&1
    fi
fi

FILESIZE=$(stat -c%s "$MODEL" 2>/dev/null || echo 0)
if [ "$FILESIZE" -lt 1000000000 ]; then
    echo "[$(date)] ERROR: All download methods failed ($FILESIZE bytes)" >> /var/log/startup.log
    rm -f "$MODEL"
    exit 1
fi

echo "[$(date)] Download complete ($FILESIZE bytes)" >> /var/log/startup.log

pip install flask --break-system-packages --ignore-installed -q >> /var/log/startup.log 2>&1

cat > /root/llama_config.json << 'EOF'
{
  "model": "/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
  "context": 131072,
  "temp": 0.6,
  "top_k": 20,
  "top_p": 0.95,
  "min_p": 0.05,
  "repeat_penalty": 1.0,
  "np": 1,
  "batch_size": 4096
}
EOF

cat > /root/onstart.sh << 'EOF'
#!/bin/sh
export LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH

# Model manager on port 5000
python3 /app/model_manager.py >> /var/log/model-manager.log 2>&1 &

# Single GPU — 131K context. Qwen3.6 hybrid Mamba keeps KV cache tiny (~1.4 GB)
# so this fits on 24 GB cards with ~1.5 GB headroom. Bump to 262144 on 32 GB+ cards.
# -np 1: eliminates auto 4-slot KV pool (35% speed penalty if omitted)
# -b 4096: faster prompt prefill vs default 2048
CUDA_VISIBLE_DEVICES=0 /app/llama-server \
  -m /models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 999 \
  -fa on \
  -c 131072 \
  -np 1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --no-mmap \
  --jinja \
  -b 4096 \
  --temp 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --min-p 0.05 \
  --repeat-penalty 1.0 >> /var/log/llama-server.log 2>&1
EOF

chmod +x /root/onstart.sh
s6-svc -r /var/run/s6/services/onstart
