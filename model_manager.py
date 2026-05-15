import os
import json
import subprocess
import threading
import time
import uuid
from pathlib import Path
from flask import Flask, jsonify, request

app = Flask(__name__)

MODELS_DIR = Path("/models")
CONFIG_FILE = Path("/root/llama_config.json")
TOKEN_FILE = Path("/root/.hf_token")
ONSTART = Path("/root/onstart.sh")

downloads = {}  # id -> {status, filename, repo_id, downloaded, total, error}
_lock = threading.Lock()

DEFAULT_CONFIG = {
    "model": None,
    "context": 262144,
    "temp": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.05,
    "repeat_penalty": 1.0,
    "np": 1,
    "batch_size": 4096,
}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_config():
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def load_token():
    return TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else None


def write_onstart(cfg):
    if not cfg.get("model"):
        return
    lines = [
        "#!/bin/sh",
        "export LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH",
        "# Start model manager",
        "python3 /app/model_manager.py >> /var/log/model-manager.log 2>&1 &",
        "/app/llama-server \\",
        f"  -m {cfg['model']} \\",
        "  --host 0.0.0.0 \\",
        "  --port 8080 \\",
        "  -ngl 999 \\",
        "  -fa on \\",
        f"  -c {cfg['context']} \\",
        f"  -np {cfg['np']} \\",
        "  --cache-type-k q8_0 \\",
        "  --cache-type-v q8_0 \\",
        "  --no-mmap \\",
        "  --jinja \\",
        f"  -b {cfg['batch_size']} \\",
        f"  --temp {cfg['temp']} \\",
        f"  --top-k {cfg['top_k']} \\",
        f"  --top-p {cfg['top_p']} \\",
        f"  --min-p {cfg['min_p']} \\",
        f"  --repeat-penalty {cfg['repeat_penalty']} >> /var/log/llama-server.log 2>&1",
    ]
    ONSTART.write_text("\n".join(lines) + "\n")
    ONSTART.chmod(0o755)


def restart_llama(cfg):
    write_onstart(cfg)
    subprocess.run(["pkill", "-9", "llama-server"], capture_output=True)
    # wait for VRAM to free
    for _ in range(15):
        time.sleep(2)
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and int(result.stdout.strip()) < 2000:
            break
    Path("/var/log/llama-server.log").write_text("")
    subprocess.run(["s6-svc", "-r", "/var/run/s6/services/onstart"], capture_output=True)


def active_model():
    result = subprocess.run(
        ["ps", "aux"], capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if "llama-server" in line and "-m " in line:
            parts = line.split("-m ")
            if len(parts) > 1:
                return parts[1].split()[0]
    return None


def get_hf_url(repo_id, filename):
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def get_expected_size(repo_id, filename, token=None):
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token or load_token())
        info = api.model_info(repo_id, files_metadata=True)
        for f in (info.siblings or []):
            if f.rfilename == filename:
                return f.size
    except Exception:
        pass
    return None


def _run_download(dl_id, repo_id, filename, token):
    dest = MODELS_DIR / filename
    url = get_hf_url(repo_id, filename)
    tok = token or load_token()

    expected = get_expected_size(repo_id, filename, tok)
    with _lock:
        downloads[dl_id]["total"] = expected

    aria2_cmd = [
        "aria2c", "-x", "16", "-s", "16", "-k", "1M",
        "-d", str(MODELS_DIR),
        "-o", filename,
        "--allow-overwrite=true",
    ]
    if tok:
        aria2_cmd += [f"--header=Authorization: Bearer {tok}"]
    aria2_cmd.append(url)

    proc = subprocess.Popen(aria2_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    with _lock:
        downloads[dl_id]["pid"] = proc.pid

    # poll file size for progress while aria2c runs
    while proc.poll() is None:
        try:
            downloaded = dest.stat().st_size if dest.exists() else 0
            with _lock:
                downloads[dl_id]["downloaded"] = downloaded
        except Exception:
            pass
        time.sleep(1)

    proc.wait()
    final_size = dest.stat().st_size if dest.exists() else 0

    with _lock:
        if proc.returncode == 0 and final_size > 1_000_000:
            downloads[dl_id]["status"] = "done"
            downloads[dl_id]["downloaded"] = final_size
            downloads[dl_id]["total"] = final_size
        else:
            downloads[dl_id]["status"] = "error"
            downloads[dl_id]["error"] = f"aria2c exit {proc.returncode}, size={final_size}"
            if dest.exists() and final_size < 1_000_000:
                dest.unlink(missing_ok=True)


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    cfg = load_config()
    return jsonify({
        "service": "llama model manager",
        "active_model": active_model(),
        "config": cfg,
        "endpoints": [
            "GET  /api/models",
            "GET  /api/status",
            "POST /api/download  {repo_id, filename, token?}",
            "GET  /api/downloads",
            "GET  /api/downloads/<id>",
            "DELETE /api/downloads/<id>",
            "POST /api/switch   {filename, context?, temp?, top_k?, top_p?, min_p?, repeat_penalty?, np?, batch_size?}",
            "POST /api/token    {token}",
            "GET  /api/token",
            "DELETE /api/models/<filename>",
        ]
    })


@app.get("/api/models")
def list_models():
    models = []
    for f in sorted(MODELS_DIR.glob("*.gguf")):
        models.append({
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "size_gb": round(f.stat().st_size / 1e9, 2),
            "active": f.name in (active_model() or ""),
        })
    return jsonify(models)


@app.get("/api/status")
def server_status():
    # llama health
    try:
        r = subprocess.run(
            ["curl", "-s", "http://localhost:8080/health"],
            capture_output=True, text=True, timeout=3
        )
        health = json.loads(r.stdout) if r.stdout else {"status": "unreachable"}
    except Exception:
        health = {"status": "unreachable"}

    # VRAM
    vram = {}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        used, free, total = [int(x.strip()) for x in r.stdout.strip().split(",")]
        vram = {"used_mib": used, "free_mib": free, "total_mib": total}
    except Exception:
        pass

    cfg = load_config()
    return jsonify({
        "health": health,
        "active_model": active_model(),
        "config": cfg,
        "vram": vram,
    })


@app.post("/api/download")
def start_download():
    body = request.get_json(force=True) or {}
    repo_id = body.get("repo_id", "").strip()
    filename = body.get("filename", "").strip()
    token = body.get("token", "").strip() or None

    if not repo_id or not filename:
        return jsonify({"error": "repo_id and filename are required"}), 400

    dest = MODELS_DIR / filename
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return jsonify({"error": f"{filename} already exists", "size_gb": round(dest.stat().st_size / 1e9, 2)}), 409

    dl_id = str(uuid.uuid4())[:8]
    with _lock:
        downloads[dl_id] = {
            "id": dl_id,
            "status": "downloading",
            "repo_id": repo_id,
            "filename": filename,
            "downloaded": 0,
            "total": None,
            "error": None,
            "pid": None,
        }

    t = threading.Thread(target=_run_download, args=(dl_id, repo_id, filename, token), daemon=True)
    t.start()
    return jsonify({"id": dl_id, "message": f"Download started for {filename}"}), 202


@app.get("/api/downloads")
def list_downloads():
    with _lock:
        result = list(downloads.values())
    for d in result:
        if d["total"] and d["downloaded"]:
            d["progress_pct"] = round(d["downloaded"] / d["total"] * 100, 1)
        else:
            d["progress_pct"] = None
    return jsonify(result)


@app.get("/api/downloads/<dl_id>")
def get_download(dl_id):
    with _lock:
        d = downloads.get(dl_id)
    if not d:
        return jsonify({"error": "not found"}), 404
    d = dict(d)
    if d["total"] and d["downloaded"]:
        d["progress_pct"] = round(d["downloaded"] / d["total"] * 100, 1)
    return jsonify(d)


@app.delete("/api/downloads/<dl_id>")
def cancel_download(dl_id):
    with _lock:
        d = downloads.get(dl_id)
    if not d:
        return jsonify({"error": "not found"}), 404
    pid = d.get("pid")
    if pid:
        subprocess.run(["kill", "-9", str(pid)], capture_output=True)
    # remove partial file
    dest = MODELS_DIR / d["filename"]
    for ext in ("", ".aria2"):
        p = Path(str(dest) + ext)
        if p.exists():
            p.unlink(missing_ok=True)
    with _lock:
        downloads[dl_id]["status"] = "cancelled"
    return jsonify({"message": "cancelled"})


@app.post("/api/switch")
def switch_model():
    body = request.get_json(force=True) or {}
    filename = body.get("filename", "").strip()
    if not filename:
        return jsonify({"error": "filename is required"}), 400

    dest = MODELS_DIR / filename
    if not dest.exists():
        return jsonify({"error": f"{filename} not found in /models"}), 404

    cfg = load_config()
    cfg["model"] = str(dest)

    # allow overriding server params at switch time
    for key in ("context", "temp", "top_k", "top_p", "min_p", "repeat_penalty", "np", "batch_size"):
        if key in body:
            cfg[key] = body[key]

    save_config(cfg)

    t = threading.Thread(target=restart_llama, args=(cfg,), daemon=True)
    t.start()

    return jsonify({"message": f"Switching to {filename}", "config": cfg})


@app.post("/api/token")
def set_token():
    body = request.get_json(force=True) or {}
    token = body.get("token", "").strip()
    if not token:
        return jsonify({"error": "token is required"}), 400
    TOKEN_FILE.write_text(token)
    TOKEN_FILE.chmod(0o600)
    return jsonify({"message": "Token saved"})


@app.get("/api/token")
def check_token():
    return jsonify({"token_set": TOKEN_FILE.exists()})


@app.delete("/api/token")
def delete_token():
    TOKEN_FILE.unlink(missing_ok=True)
    return jsonify({"message": "Token removed"})


@app.delete("/api/models/<path:filename>")
def delete_model(filename):
    dest = MODELS_DIR / filename
    if not dest.exists():
        return jsonify({"error": "not found"}), 404
    if filename in (active_model() or ""):
        return jsonify({"error": "cannot delete the currently active model"}), 409
    dest.unlink()
    return jsonify({"message": f"{filename} deleted"})


if __name__ == "__main__":
    # init config from current onstart.sh if no config exists yet
    if not CONFIG_FILE.exists():
        cfg = dict(DEFAULT_CONFIG)
        current = active_model()
        if current:
            cfg["model"] = current
        save_config(cfg)

    app.run(host="0.0.0.0", port=5000, threaded=True)
