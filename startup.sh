#!/bin/sh
mkdir -p /models
MODEL="/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
URL="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
export HF_TOKEN="hf_your_token_here"

if [ ! -f "$MODEL" ]; then
    echo "[$(date)] Starting model download..." >> /var/log/startup.log

    # Try 1: aria2c (multi-connection, fastest that actually works)
    echo "[$(date)] Trying aria2c..." >> /var/log/startup.log
    aria2c -x 16 -s 16 -k 1M -d /models -o Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
        --header="Authorization: Bearer $HF_TOKEN" \
        "$URL" >> /var/log/startup.log 2>&1

    # Try 2: wget
    if [ ! -f "$MODEL" ]; then
        echo "[$(date)] aria2c failed, trying wget..." >> /var/log/startup.log
        wget --header="Authorization: Bearer $HF_TOKEN" \
            -O "$MODEL" "$URL" >> /var/log/startup.log 2>&1
    fi

    # Try 3: curl
    if [ ! -f "$MODEL" ]; then
        echo "[$(date)] wget failed, trying curl..." >> /var/log/startup.log
        curl -L -H "Authorization: Bearer $HF_TOKEN" \
            -o "$MODEL" "$URL" >> /var/log/startup.log 2>&1
    fi
fi

# Verify download
FILESIZE=$(stat -c%s "$MODEL" 2>/dev/null || echo 0)
if [ "$FILESIZE" -lt 1000000000 ]; then
    echo "[$(date)] ERROR: All download methods failed ($FILESIZE bytes)" >> /var/log/startup.log
    rm -f "$MODEL"
    exit 1
fi

echo "[$(date)] Download complete ($FILESIZE bytes)" >> /var/log/startup.log

cat > /root/onstart.sh << 'EOF'
#!/bin/sh
export LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH
/app/llama-server \
  -m /models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 999 \
  -fa on \
  -c 65536 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --no-mmap \
  --jinja >> /var/log/llama-server.log 2>&1
EOF

chmod +x /root/onstart.sh
s6-svc -r /var/run/s6/services/onstart