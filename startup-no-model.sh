#!/bin/sh
mkdir -p /models

pip install flask --break-system-packages --ignore-installed -q >> /var/log/startup.log 2>&1



cat > /root/onstart.sh << 'EOF'
#!/bin/sh
export LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH

# Model manager on port 5000
python3 /app/model_manager.py >> /var/log/model-manager.log 2>&1 &
EOF

chmod +x /root/onstart.sh
s6-svc -r /var/run/s6/services/onstart