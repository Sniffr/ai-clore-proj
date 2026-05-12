#!/bin/sh
export DEBIAN_FRONTEND="noninteractive"
export PATH="/usr/bin:/usr/local/bin:/bin:/usr/sbin:/sbin:$PATH"

apt update -y
apt install -y wget curl aria2

mkdir -p /models

# Download model (only if not already there)
if [ ! -f /models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf ]; then
    echo "[$(date)] Starting model download..." >> /var/log/startup.log
    aria2c -x 16 -s 16 -k 1M -d /models -o Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
        "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" \
        >> /var/log/startup.log 2>&1
fi

# Verify download
FILESIZE=$(stat -c%s /models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf 2>/dev/null || echo 0)
if [ "$FILESIZE" -lt 1000000000 ]; then
    echo "[$(date)] ERROR: Model too small ($FILESIZE bytes)" >> /var/log/startup.log
    exit 1
fi

# Create onstart.sh (runs every container start)
cat > /root/onstart.sh << 'EOF'
#!/bin/sh
export LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH
/app/llama-server \
  -m /models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 999 \
  -fa on \
  -c 8192 \
  --jinja >> /var/log/llama-server.log 2>&1
EOF

chmod +x /root/onstart.sh
s6-svc -r /var/run/s6/services/onstart