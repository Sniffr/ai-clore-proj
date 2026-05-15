import os
import json
import subprocess
import threading
import time
import uuid
from pathlib import Path
from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)

MODELS_DIR = Path("/models")
CONFIG_FILE = Path("/root/llama_config.json")
TOKEN_FILE = Path("/root/.hf_token")
ONSTART = Path("/root/onstart.sh")
LLAMA_BIN = "/app/llama-server"

downloads = {}
_dl_lock = threading.Lock()

_servers_lock = threading.Lock()
primary_state = {
    "port": 8080, "gpu": None,
    "status": "unknown",
    "message": "", "model": None, "started_at": None,
}
extra_servers = {}

DEFAULT_CONFIG = {
    "model": None, "context": 131072, "temp": 0.6,
    "top_k": 20, "top_p": 0.95, "min_p": 0.05,
    "repeat_penalty": 1.0, "np": 1, "batch_size": 4096,
}

# ── config helpers ────────────────────────────────────────────────────────────

def load_config():
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return dict(DEFAULT_CONFIG)

def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

def load_token():
    return TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else None

def sanitize_repo_id(repo_id):
    return repo_id.split(":")[0].strip()

def get_hf_url(repo_id, filename):
    return f"https://huggingface.co/{sanitize_repo_id(repo_id)}/resolve/main/{filename}"

# ── GPU helpers ───────────────────────────────────────────────────────────────

def get_gpus():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        gpus = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 5:
                gpus.append({
                    "id": int(parts[0]), "name": parts[1],
                    "used_mib": int(parts[2]), "free_mib": int(parts[3]),
                    "total_mib": int(parts[4]),
                })
        return gpus
    except Exception:
        return []

def detect_model_type(name):
    n = name.lower()
    if "qwen3" in n:
        return "moe_mamba"  # Qwen3.x series: hybrid Transformer-Mamba MoE
    if any(x in n for x in ["moe", "mixture", "deepseek-moe"]):
        return "moe"
    if "mamba" in n or "ssm" in n:
        return "mamba"
    return "dense"

def estimate_kv_overhead_mib(model_type, context):
    """Estimate KV cache + compute buffer in MiB for q8_0 KV caching."""
    ctx_k = context / 1024
    if model_type in ("moe_mamba", "mamba"):
        # Qwen3.x: ~4/62 full-attention layers; measured ~2.8 GB at 262K, ~1.4 GB at 131K
        kv = max(400, int(ctx_k * 11))
    elif model_type == "moe":
        kv = max(600, int(ctx_k * 25))
    else:
        kv = max(800, int(ctx_k * 80))
    return kv + 800  # +800 MiB compute buffer

def total_free_vram_mib():
    return sum(g["free_mib"] for g in get_gpus())

def poll_health(port, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = subprocess.run(
                ["curl", "-sf", f"http://localhost:{port}/health"],
                capture_output=True, timeout=3
            )
            if r.returncode == 0:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

# ── VRAM fit + recommendation ─────────────────────────────────────────────────

def estimate_fit(model_path_or_size, context=None, overhead_mib=None):
    if isinstance(model_path_or_size, (str, Path)):
        p = Path(model_path_or_size)
        size_bytes = p.stat().st_size if p.exists() else 0
        model_name = p.name
    else:
        size_bytes = model_path_or_size
        model_name = ""

    model_mib = size_bytes // (1024 * 1024)
    ctx = context or 131072

    if overhead_mib is None:
        overhead_mib = estimate_kv_overhead_mib(detect_model_type(model_name), ctx)

    needed_mib = model_mib + overhead_mib
    gpus = get_gpus()
    total_free = sum(g["free_mib"] for g in gpus)

    gpu_analysis = []
    for g in gpus:
        gpu_analysis.append({
            "id": g["id"], "name": g["name"],
            "free_mib": g["free_mib"], "total_mib": g["total_mib"],
            "fits": g["free_mib"] >= needed_mib,
            "headroom_mib": g["free_mib"] - needed_mib,
        })

    return {
        "model_mib": model_mib,
        "model_gb": round(size_bytes / 1e9, 2),
        "overhead_mib": overhead_mib,
        "needed_mib": needed_mib,
        "fits_single_gpu": any(a["fits"] for a in gpu_analysis),
        "fits_multi_gpu": total_free >= needed_mib,
        "total_free_mib": total_free,
        "gpus": gpu_analysis,
        "recommended": (
            "single_gpu" if any(a["fits"] for a in gpu_analysis)
            else "multi_gpu" if total_free >= needed_mib
            else "does_not_fit"
        ),
    }

def _active_models():
    result = {}
    r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "llama-server" in line and "-m " in line:
            parts = line.split("-m ")
            if len(parts) > 1:
                model_path = parts[1].split()[0]
                filename = Path(model_path).name
                port = 8080
                if "--port " in line:
                    try:
                        port = int(line.split("--port ")[1].split()[0])
                    except Exception:
                        pass
                result[filename] = port
    return result

def recommend_deployment(model_path, context=None, assume_switch=False):
    """
    assume_switch=True: simulate post-restart state (primary VRAM freed).
    Use this when recommending for a model switch, not an extra launch.
    """
    p = Path(model_path)
    ctx = context or load_config().get("context", 131072)
    gpus = get_gpus()

    if assume_switch:
        # After pkill llama-server, GPUs will be nearly empty (~400 MiB idle baseline)
        for g in gpus:
            g["free_mib"] = g["total_mib"] - 400
            g["used_mib"] = 400

    fit = estimate_fit(p, context=ctx)
    # Re-run estimate_fit using adjusted GPU VRAM
    if assume_switch:
        model_name = p.name
        overhead = estimate_kv_overhead_mib(detect_model_type(model_name), ctx)
        model_mib = (p.stat().st_size if p.exists() else 0) // (1024 * 1024)
        needed_mib = model_mib + overhead
        total_free = sum(g["free_mib"] for g in gpus)
        gpu_analysis = [{"id": g["id"], "name": g["name"],
                          "free_mib": g["free_mib"], "total_mib": g["total_mib"],
                          "fits": g["free_mib"] >= needed_mib,
                          "headroom_mib": g["free_mib"] - needed_mib} for g in gpus]
        fit = {"model_mib": model_mib, "model_gb": round(p.stat().st_size/1e9, 2) if p.exists() else 0,
               "overhead_mib": overhead, "needed_mib": needed_mib,
               "fits_single_gpu": any(a["fits"] for a in gpu_analysis),
               "fits_multi_gpu": total_free >= needed_mib,
               "total_free_mib": total_free, "gpus": gpu_analysis,
               "recommended": ("single_gpu" if any(a["fits"] for a in gpu_analysis)
                                else "multi_gpu" if total_free >= needed_mib else "does_not_fit")}
    model_type = detect_model_type(p.name)

    # GPUs with >2 GB used are considered occupied
    occupied = {g["id"] for g in gpus if g["used_mib"] > 2000}

    # Best single-GPU candidates: prefer unoccupied, then by headroom
    candidates = sorted(
        [g for g in fit["gpus"] if g["fits"]],
        key=lambda g: (g["id"] in occupied, -g["headroom_mib"])
    )

    warnings = []

    if candidates:
        best = candidates[0]
        others = [g for g in gpus if g["id"] != best["id"]]
        for og in others:
            status = "occupied" if og["id"] in occupied else f"{og['free_mib']:,} MiB free"
            if og["id"] not in occupied:
                warnings.append(
                    f"GPU {og['id']} stays idle ({og['free_mib']:,} MiB free) — "
                    "use 'Launch extra model' to run a second model there"
                )
        if model_type in ("moe", "moe_mamba") and len(gpus) > 1:
            warnings.append(
                "MoE model: single GPU avoids pipeline parallelism overhead — lower latency"
            )
        return {
            "strategy": "single_gpu",
            "recommended_gpu": best["id"],
            "headroom_mib": best["headroom_mib"],
            "reasoning": (
                f"Fits on GPU {best['id']} with {best['headroom_mib']:,} MiB to spare. "
                "Single GPU is always faster than multi-GPU for models that fit."
            ),
            "warnings": warnings,
            "fit": fit,
        }
    elif fit["fits_multi_gpu"]:
        warnings.append("No single GPU has enough free VRAM — pipeline parallelism across all GPUs")
        if model_type in ("moe", "moe_mamba"):
            warnings.append("MoE + pipeline parallelism: expect ~15–30% latency overhead vs single GPU")
        return {
            "strategy": "multi_gpu",
            "recommended_gpu": None,
            "headroom_mib": fit["total_free_mib"] - fit["needed_mib"],
            "reasoning": (
                f"Too large for one GPU (needs {fit['needed_mib']:,} MiB; "
                f"max free on single GPU: {max(g['free_mib'] for g in gpus):,} MiB). "
                "Using all GPUs via pipeline parallelism."
            ),
            "warnings": warnings,
            "fit": fit,
        }
    else:
        return {
            "strategy": "does_not_fit",
            "recommended_gpu": None,
            "headroom_mib": fit["total_free_mib"] - fit["needed_mib"],
            "reasoning": (
                f"Insufficient VRAM: needs ~{fit['needed_mib']:,} MiB, "
                f"only {fit['total_free_mib']:,} MiB free across all GPUs."
            ),
            "warnings": ["Stop other running models to free VRAM first"],
            "fit": fit,
        }

def perf_tips(model_path, cfg):
    name = Path(model_path).name.lower() if model_path else ""
    gpus = get_gpus()
    tips = []
    model_type = detect_model_type(name)
    ctx = cfg.get("context", 131072)
    np_val = cfg.get("np", 1)
    batch = cfg.get("batch_size", 2048)
    total_vram = sum(g["total_mib"] for g in gpus)

    if model_type in ("moe_mamba", "mamba"):
        tips.append({"level": "info",
                     "text": "Hybrid Mamba/MoE: only a few layers use full-attention KV cache. "
                             "262K context costs ~2.8 GB VRAM vs ~80 GB for a dense model."})

    if np_val > 1:
        tips.append({"level": "warn",
                     "text": f"np={np_val}: creates {np_val}× KV slots, causing ~35% throughput "
                             "penalty for single-user use. Set np=1 unless you need concurrent users."})

    if batch < 4096:
        tips.append({"level": "warn",
                     "text": f"batch_size={batch}: increase to 4096 for faster prompt prefill."})

    if model_type in ("moe", "moe_mamba") and len(gpus) > 1 and cfg.get("gpu") is None:
        tips.append({"level": "warn",
                     "text": "MoE on multiple GPUs uses pipeline parallelism (tensor-split not "
                             "supported for MoE). Single GPU has lower latency if the model fits."})

    if model_type == "moe_mamba" and total_vram >= 58000 and ctx < 262144:
        tips.append({"level": "info",
                     "text": f"With {total_vram//1024} GB total VRAM and a Mamba hybrid you can "
                             f"run full 262K context. Currently at {ctx//1024}K."})

    tips.append({"level": "ok",
                 "text": "KV cache q8_0: <0.5% perplexity cost vs fp16, optimal size/quality tradeoff."})
    tips.append({"level": "ok",
                 "text": "-fa on (flash attention): ~40% VRAM reduction at long contexts, higher throughput."})
    return tips

# ── llama-server lifecycle ────────────────────────────────────────────────────

def build_llama_args(cfg, port=8080, gpu=None):
    args = [LLAMA_BIN,
            "-m", cfg["model"],
            "--host", "0.0.0.0",
            "--port", str(port),
            "-ngl", "999",
            "-fa", "on",
            "-c", str(cfg["context"]),
            "-np", str(cfg["np"]),
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--no-mmap",
            "--jinja",
            "-b", str(cfg["batch_size"]),
            "--temp", str(cfg["temp"]),
            "--top-k", str(cfg["top_k"]),
            "--top-p", str(cfg["top_p"]),
            "--min-p", str(cfg["min_p"]),
            "--repeat-penalty", str(cfg["repeat_penalty"])]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"/app:{env.get('LD_LIBRARY_PATH', '')}"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return args, env

def write_onstart(cfg, gpu=None):
    if not cfg.get("model"):
        return
    gpu_line = f'export CUDA_VISIBLE_DEVICES="{gpu}"' if gpu is not None else ""
    lines = [
        "#!/bin/sh",
        "export LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH",
        gpu_line,
        "python3 /app/model_manager.py >> /var/log/model-manager.log 2>&1 &",
        f"{LLAMA_BIN} \\",
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
    ONSTART.write_text("\n".join(l for l in lines if l) + "\n")
    ONSTART.chmod(0o755)

def _do_restart_primary(cfg, gpu=None):
    with _servers_lock:
        primary_state.update({"status": "stopping", "message": "Stopping current server...",
                               "model": cfg.get("model"), "gpu": gpu})
    subprocess.run(["pkill", "-9", "llama-server"], capture_output=True)

    with _servers_lock:
        primary_state.update({"status": "clearing_vram", "message": "Waiting for VRAM to free..."})
    for _ in range(30):
        time.sleep(2)
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        total_used = sum(int(x.strip()) for x in r.stdout.strip().splitlines() if x.strip().isdigit())
        if total_used < 3000:
            break

    write_onstart(cfg, gpu)
    Path("/var/log/llama-server.log").write_text("")

    with _servers_lock:
        primary_state.update({"status": "loading",
                               "message": f"Loading {Path(cfg['model']).name}..."})
    subprocess.run(["s6-svc", "-r", "/var/run/s6/services/onstart"], capture_output=True)

    if poll_health(8080, timeout=300):
        with _servers_lock:
            primary_state.update({"status": "running", "message": "", "started_at": time.time()})
    else:
        with _servers_lock:
            primary_state.update({"status": "error", "message": "Server failed to start — check logs"})

def restart_primary(cfg, gpu=None):
    save_config(cfg)
    threading.Thread(target=_do_restart_primary, args=(cfg, gpu), daemon=True).start()

def _do_start_extra(port, model_path, cfg, gpu):
    with _servers_lock:
        extra_servers[port].update({"status": "loading",
                                    "message": f"Loading {Path(model_path).name}..."})
    args, env = build_llama_args(cfg, port=port, gpu=gpu)
    log = open(f"/var/log/llama-server-{port}.log", "w")
    proc = subprocess.Popen(args, env=env, stdout=log, stderr=log)
    with _servers_lock:
        extra_servers[port]["pid"] = proc.pid
        extra_servers[port]["proc"] = proc

    if poll_health(port, timeout=300):
        with _servers_lock:
            extra_servers[port].update({"status": "running", "message": "", "started_at": time.time()})
    else:
        with _servers_lock:
            extra_servers[port].update({"status": "error", "message": "Failed to start — check logs"})
        proc.kill()

# ── download ──────────────────────────────────────────────────────────────────

def get_expected_size(repo_id, filename, token=None):
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token or load_token())
        info = api.model_info(sanitize_repo_id(repo_id), files_metadata=True)
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
    with _dl_lock:
        downloads[dl_id]["total"] = expected

    aria2_cmd = ["aria2c", "-x", "16", "-s", "16", "-k", "1M",
                 "-d", str(MODELS_DIR), "-o", filename, "--allow-overwrite=true"]
    if tok:
        aria2_cmd += [f"--header=Authorization: Bearer {tok}"]
    aria2_cmd.append(url)

    log_path = Path(f"/var/log/download-{dl_id}.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(aria2_cmd, stdout=log_file, stderr=log_file)

    with _dl_lock:
        downloads[dl_id]["pid"] = proc.pid
        downloads[dl_id]["log"] = str(log_path)

    while proc.poll() is None:
        try:
            downloaded = dest.stat().st_size if dest.exists() else 0
            with _dl_lock:
                downloads[dl_id]["downloaded"] = downloaded
        except Exception:
            pass
        time.sleep(1)

    log_file.close()
    proc.wait()
    final_size = dest.stat().st_size if dest.exists() else 0

    with _dl_lock:
        if proc.returncode == 0 and final_size > 1_000_000:
            downloads[dl_id].update({"status": "done", "downloaded": final_size, "total": final_size})
        else:
            err_detail = f"exit {proc.returncode}, {final_size} bytes"
            try:
                lines = [l.strip() for l in log_path.read_text().splitlines()
                         if l.strip() and "Redirecting" not in l]
                if lines:
                    err_detail = lines[-1][:120]
            except Exception:
                pass
            downloads[dl_id]["status"] = "error"
            downloads[dl_id]["error"] = err_detail
            if dest.exists() and final_size < 1_000_000:
                dest.unlink(missing_ok=True)

# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/api")
def api_index():
    return jsonify({"service": "llama model manager", "ui": "GET /"})


@app.get("/api/status")
def server_status():
    try:
        r = subprocess.run(["curl", "-sf", "http://localhost:8080/health"],
                           capture_output=True, text=True, timeout=3)
        health = json.loads(r.stdout) if r.stdout else {"status": "unreachable"}
        if r.returncode == 0 and primary_state["status"] == "unknown":
            with _servers_lock:
                primary_state["status"] = "running"
    except Exception:
        health = {"status": "unreachable"}

    with _servers_lock:
        state = dict(primary_state)
        extras = {p: {k: v for k, v in s.items() if k != "proc"} for p, s in extra_servers.items()}

    gpus = get_gpus()
    vram = {}
    if gpus:
        # Annotate each GPU with what's running on it
        for g in gpus:
            g["assigned_model"] = None
            g["assigned_port"] = None
            p_gpu = state.get("gpu")
            p_model = state.get("model")
            if p_model and state["status"] in ("running", "loading", "clearing_vram"):
                if p_gpu is None or int(p_gpu) == g["id"]:
                    g["assigned_model"] = Path(p_model).name
                    g["assigned_port"] = 8080
            for port, s in extras.items():
                if s.get("gpu") is not None and int(s["gpu"]) == g["id"]:
                    if s["status"] in ("running", "loading"):
                        g["assigned_model"] = Path(s["model"]).name if s.get("model") else None
                        g["assigned_port"] = port

        vram = {
            "gpus": gpus,
            "total_used_mib": sum(g["used_mib"] for g in gpus),
            "total_free_mib": sum(g["free_mib"] for g in gpus),
            "total_mib": sum(g["total_mib"] for g in gpus),
        }

    cfg = load_config()
    return jsonify({"health": health, "server_state": state, "config": cfg, "vram": vram})


@app.get("/api/server/state")
def get_server_state():
    with _servers_lock:
        return jsonify(dict(primary_state))


@app.get("/api/gpus")
def gpu_info():
    return jsonify(get_gpus())


@app.get("/api/recommend")
def get_recommendation():
    filename = request.args.get("filename", "").strip()
    context = request.args.get("context", type=int)
    # assume_switch=1: compute recommendation assuming current model VRAM is freed first
    assume_switch = request.args.get("assume_switch", "1") == "1"
    if not filename:
        return jsonify({"error": "filename required"}), 400
    dest = MODELS_DIR / filename
    if not dest.exists():
        return jsonify({"error": f"{filename} not found"}), 404
    rec = recommend_deployment(dest, context=context, assume_switch=assume_switch)
    cfg = load_config()
    if context:
        cfg["context"] = context
    rec["tips"] = perf_tips(str(dest), cfg)
    return jsonify(rec)


@app.get("/api/models")
def list_models():
    active = _active_models()
    models = []
    for f in sorted(MODELS_DIR.glob("*.gguf")):
        fit = estimate_fit(f)
        models.append({
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "size_gb": round(f.stat().st_size / 1e9, 2),
            "active_on": active.get(f.name),
            "fit": fit,
        })
    return jsonify(models)


@app.get("/api/vram/check")
def vram_check():
    filename = request.args.get("filename", "").strip()
    context = request.args.get("context", type=int)
    repo_id = request.args.get("repo_id", "").strip()
    size_bytes = 0
    if filename:
        dest = MODELS_DIR / filename
        if dest.exists():
            size_bytes = dest.stat().st_size
        elif repo_id:
            size_bytes = get_expected_size(repo_id, filename) or 0
    if not size_bytes:
        return jsonify({"error": "File not found locally; provide repo_id to estimate from HF metadata"}), 404
    return jsonify(estimate_fit(size_bytes, context=context))


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
    for key in ("context", "temp", "top_k", "top_p", "min_p", "repeat_penalty", "np", "batch_size"):
        if key in body:
            cfg[key] = body[key]

    # Auto-select GPU unless caller explicitly passed one (always assume primary VRAM freed)
    if "gpu" in body:
        raw = body["gpu"]
        gpu = int(raw) if raw is not None and str(raw).strip() != "" else None
        if gpu is None:
            rec = recommend_deployment(dest, context=cfg.get("context"), assume_switch=True)
            gpu = rec.get("recommended_gpu")
    else:
        rec = recommend_deployment(dest, context=cfg.get("context"), assume_switch=True)
        gpu = rec.get("recommended_gpu")

    restart_primary(cfg, gpu=gpu)
    return jsonify({"message": f"Switching to {filename} on GPU {gpu if gpu is not None else 'all'}",
                    "config": cfg, "gpu": gpu})


@app.post("/api/server/restart")
def server_restart():
    body = request.get_json(force=True) or {}
    cfg = load_config()
    for key in ("context", "temp", "top_k", "top_p", "min_p", "repeat_penalty", "np", "batch_size"):
        if key in body:
            cfg[key] = body[key]
    if not cfg.get("model"):
        return jsonify({"error": "no model configured"}), 400
    gpu = body.get("gpu")
    restart_primary(cfg, gpu=int(gpu) if gpu is not None and str(gpu).strip() != "" else None)
    return jsonify({"message": "Restarting...", "config": cfg})


@app.get("/api/server/logs")
def server_logs():
    n = int(request.args.get("lines", 50))
    port = int(request.args.get("port", 8080))
    log = f"/var/log/llama-server-{port}.log" if port != 8080 else "/var/log/llama-server.log"
    try:
        r = subprocess.run(["tail", f"-{n}", log], capture_output=True, text=True)
        return jsonify({"lines": r.stdout.splitlines()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── multi-server ──────────────────────────────────────────────────────────────

@app.get("/api/servers")
def list_servers():
    with _servers_lock:
        extras = {p: {k: v for k, v in s.items() if k != "proc"} for p, s in extra_servers.items()}
        primary = dict(primary_state)
    return jsonify({"primary": primary, "extra": extras})


@app.post("/api/servers")
def launch_extra_server():
    body = request.get_json(force=True) or {}
    filename = body.get("filename", "").strip()
    port = int(body.get("port", 8081))
    gpu = body.get("gpu")

    if not filename:
        return jsonify({"error": "filename required"}), 400
    dest = MODELS_DIR / filename
    if not dest.exists():
        return jsonify({"error": f"{filename} not found"}), 404

    with _servers_lock:
        if port in extra_servers:
            return jsonify({"error": f"Port {port} already in use"}), 409

    # Auto-select a free GPU if not specified
    if gpu is None or str(gpu).strip() == "":
        fit = estimate_fit(dest)
        with _servers_lock:
            used_gpus = {int(s["gpu"]) for s in extra_servers.values() if s.get("gpu") is not None}
        p_gpu = primary_state.get("gpu")
        if p_gpu is not None:
            used_gpus.add(int(p_gpu))
        gpu = None
        for ga in fit["gpus"]:
            if ga["fits"] and ga["id"] not in used_gpus:
                gpu = ga["id"]
                break
        if gpu is None:
            return jsonify({"error": "No free GPU found with sufficient VRAM for this model"}), 409
    else:
        gpu = int(gpu)

    fit = estimate_fit(dest)
    gpu_fit = next((g for g in fit["gpus"] if g["id"] == gpu), None)
    if gpu_fit and not gpu_fit["fits"]:
        return jsonify({
            "error": f"Model needs {fit['needed_mib']:,} MiB but GPU {gpu} only has {gpu_fit['free_mib']:,} MiB free",
            "fit": fit
        }), 409

    base_cfg = load_config()
    cfg = {**base_cfg, "model": str(dest)}
    for key in ("context", "temp", "top_k", "top_p", "min_p", "repeat_penalty", "np", "batch_size"):
        if key in body:
            cfg[key] = body[key]

    with _servers_lock:
        extra_servers[port] = {
            "port": port, "gpu": gpu, "model": str(dest),
            "pid": None, "proc": None,
            "status": "loading", "message": f"Loading {filename}...",
            "config": cfg, "started_at": None,
        }

    threading.Thread(target=_do_start_extra, args=(port, dest, cfg, gpu), daemon=True).start()
    return jsonify({"message": f"Launching {filename} on GPU {gpu} port {port}", "port": port, "gpu": gpu}), 202


@app.delete("/api/servers/<int:port>")
def stop_extra_server(port):
    with _servers_lock:
        s = extra_servers.get(port)
    if not s:
        return jsonify({"error": "not found"}), 404
    proc = s.get("proc")
    if proc:
        proc.kill()
    with _servers_lock:
        del extra_servers[port]
    return jsonify({"message": f"Server on port {port} stopped"})


# ── download routes ───────────────────────────────────────────────────────────

@app.get("/api/repo/files")
def repo_files():
    repo_id = sanitize_repo_id(request.args.get("repo_id", ""))
    if not repo_id:
        return jsonify({"error": "repo_id required"}), 400
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=load_token())
        info = api.model_info(repo_id, files_metadata=True)
        files = [{"filename": f.rfilename,
                  "size_bytes": f.size,
                  "size_gb": round(f.size / 1e9, 2) if f.size else None}
                 for f in (info.siblings or []) if f.rfilename.endswith(".gguf")]
        return jsonify({"repo_id": repo_id, "files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 404


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
        return jsonify({"error": f"{filename} already exists",
                        "size_gb": round(dest.stat().st_size / 1e9, 2)}), 409
    dl_id = str(uuid.uuid4())[:8]
    with _dl_lock:
        downloads[dl_id] = {"id": dl_id, "status": "downloading",
                             "repo_id": repo_id, "filename": filename,
                             "downloaded": 0, "total": None, "error": None, "pid": None}
    threading.Thread(target=_run_download, args=(dl_id, repo_id, filename, token), daemon=True).start()
    return jsonify({"id": dl_id, "message": f"Download started: {filename}"}), 202


@app.get("/api/downloads")
def list_downloads():
    with _dl_lock:
        result = list(downloads.values())
    for d in result:
        if d["total"] and d["downloaded"]:
            d["progress_pct"] = round(d["downloaded"] / d["total"] * 100, 1)
        else:
            d["progress_pct"] = None
    return jsonify(result)


@app.get("/api/downloads/<dl_id>")
def get_download(dl_id):
    with _dl_lock:
        d = downloads.get(dl_id)
    if not d:
        return jsonify({"error": "not found"}), 404
    d = dict(d)
    if d["total"] and d["downloaded"]:
        d["progress_pct"] = round(d["downloaded"] / d["total"] * 100, 1)
    return jsonify(d)


@app.delete("/api/downloads/<dl_id>")
def cancel_download(dl_id):
    with _dl_lock:
        d = downloads.get(dl_id)
    if not d:
        return jsonify({"error": "not found"}), 404
    pid = d.get("pid")
    if pid:
        subprocess.run(["kill", "-9", str(pid)], capture_output=True)
    dest = MODELS_DIR / d["filename"]
    for ext in ("", ".aria2"):
        Path(str(dest) + ext).unlink(missing_ok=True)
    with _dl_lock:
        downloads[dl_id]["status"] = "cancelled"
    return jsonify({"message": "cancelled"})


@app.post("/api/token")
def set_token():
    body = request.get_json(force=True) or {}
    token = body.get("token", "").strip()
    if not token:
        return jsonify({"error": "token required"}), 400
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
    active = _active_models()
    if filename in active:
        return jsonify({"error": "cannot delete an active model"}), 409
    dest.unlink()
    return jsonify({"message": f"{filename} deleted"})


# ── UI ────────────────────────────────────────────────────────────────────────

UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LLM Manager</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:monospace;background:#111;color:#ccc;font-size:13px;line-height:1.6}
#app{max-width:1080px;margin:0 auto;padding:16px}
header{display:flex;align-items:center;gap:10px;padding:10px 0 16px;border-bottom:1px solid #222;margin-bottom:20px}
header h1{font-size:14px;color:#fff;font-weight:normal;letter-spacing:2px}
#hdr-state{font-size:11px;padding:2px 8px;border-radius:10px;border:1px solid #333;background:#1a1a1a}
#refresh-ts{margin-left:auto;font-size:11px;color:#444}
section{margin-bottom:26px}
h2{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#555;margin-bottom:8px;padding-bottom:5px;border-bottom:1px solid #1a1a1a}
.card{background:#161616;border:1px solid #222;border-radius:4px;padding:14px}
.row{display:flex;gap:20px;flex-wrap:wrap}
.kv{display:flex;flex-direction:column;gap:1px;min-width:100px}
.kv label{font-size:10px;color:#555;text-transform:uppercase}
.kv span{color:#ddd}
.gpu-row{min-width:280px}
.vbar{height:5px;background:#1e1e1e;border-radius:3px;margin-top:5px;overflow:hidden;min-width:180px}
.vbar-fill{height:100%;border-radius:3px;transition:width .5s}
.vbar-fill.ok{background:#1976d2}
.vbar-fill.warn{background:#f57c00}
.vbar-fill.full{background:#c62828}
table{width:100%;border-collapse:collapse}
th{text-align:left;font-size:10px;color:#444;text-transform:uppercase;padding:5px 8px;border-bottom:1px solid #1e1e1e}
td{padding:6px 8px;border-bottom:1px solid #191919;vertical-align:middle}
tr:last-child td{border-bottom:none}
tr:hover td{background:#181818}
button{font-family:monospace;font-size:12px;padding:3px 9px;border:1px solid #2a2a2a;border-radius:3px;background:#1a1a1a;color:#bbb;cursor:pointer}
button:hover{background:#222;border-color:#444}
button.primary{border-color:#1565c0;color:#90caf9;background:#0a1828}
button.primary:hover{background:#0f2040}
button.danger{border-color:#5c1a1a;color:#ef9a9a;background:#180d0d}
button.danger:hover{background:#220f0f}
button.success{border-color:#1b5e20;color:#a5d6a7;background:#0a1f0a}
button:disabled{opacity:.35;cursor:not-allowed}
input,select{font-family:monospace;font-size:12px;padding:4px 7px;background:#141414;border:1px solid #262626;border-radius:3px;color:#ddd;width:100%}
input:focus,select:focus{outline:none;border-color:#1565c0}
.frow{display:flex;gap:7px;align-items:flex-end;flex-wrap:wrap}
.fg{display:flex;flex-direction:column;gap:3px}
.fg label{font-size:10px;color:#555}
.fg.grow{flex:1;min-width:130px}
.msg{font-size:11px;padding:3px 8px;border-radius:3px;margin-top:5px;display:none}
.msg.ok{color:#81c784;background:#091509;border:1px solid #1b3d1b}
.msg.err{color:#e57373;background:#180a0a;border:1px solid #3d1b1b}
.msg.info{color:#90caf9;background:#080f1a;border:1px solid #162840}
.pb{height:3px;background:#1e1e1e;border-radius:2px;overflow:hidden;margin-top:3px}
.pb-fill{height:100%;background:#1976d2;border-radius:2px;transition:width .3s}
.fit-ok{color:#66bb6a;font-size:11px}
.fit-warn{color:#ffa726;font-size:11px}
.fit-no{color:#ef5350;font-size:11px}
.cfg-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:7px}
.state-badge{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px}
.state-running{background:#0d2a0d;color:#66bb6a;border:1px solid #1b5e20}
.state-loading{background:#0a1828;color:#90caf9;border:1px solid #1565c0;animation:pulse 1.2s ease-in-out infinite}
.state-stopping,.state-clearing_vram{background:#1a1400;color:#ffcc02;border:1px solid #5a4a00;animation:pulse 1.2s ease-in-out infinite}
.state-error{background:#2a0d0d;color:#ef9a9a;border:1px solid #7f1d1d}
.state-unknown{background:#1a1a1a;color:#666;border:1px solid #333}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
pre{background:#0d0d0d;border:1px solid #1e1e1e;border-radius:4px;padding:10px;font-size:11px;color:#8bc34a;white-space:pre-wrap;word-break:break-all;max-height:300px;overflow-y:auto}
.rec-card{padding:10px;border-radius:4px;margin-bottom:12px;font-size:11px}
.rec-single{background:#0d2a0d;border:1px solid #1b5e20}
.rec-multi{background:#1a1400;border:1px solid #5a4a00}
.rec-nope{background:#2a0d0d;border:1px solid #7f1d1d}
.tip{padding:3px 0;border-top:1px solid #1e1e1e;font-size:11px}
.tip-ok{color:#66bb6a}
.tip-warn{color:#ffa726}
.tip-info{color:#90caf9}
details summary{cursor:pointer;font-size:11px;color:#555;user-select:none;padding:4px 0}
details summary:hover{color:#888}
</style>
</head>
<body>
<div id="app">

<header>
  <h1>LLM MANAGER</h1>
  <span id="hdr-state" class="state-badge state-unknown">checking...</span>
  <span id="refresh-ts"></span>
</header>

<section>
  <h2>Status</h2>
  <div id="status-card" class="card">Loading...</div>
</section>

<section>
  <h2>Running servers</h2>
  <div id="servers-msg" class="msg"></div>
  <table>
    <thead><tr><th>Port</th><th>Model</th><th>GPU</th><th>State</th><th></th></tr></thead>
    <tbody id="servers-body"><tr><td colspan="5" style="color:#444">Loading...</td></tr></tbody>
  </table>
</section>

<section>
  <h2>Models on disk</h2>
  <div id="models-msg" class="msg"></div>
  <table>
    <thead><tr><th>Filename</th><th>Size</th><th>Best GPU fit</th><th>Active</th><th></th></tr></thead>
    <tbody id="models-body"><tr><td colspan="5" style="color:#444">Loading...</td></tr></tbody>
  </table>
</section>

<!-- SWITCH PANEL -->
<section id="switch-section" style="display:none">
  <h2>Switch primary — <span id="switch-target-label" style="color:#90caf9"></span></h2>
  <div class="card">
    <div id="rec-card" style="display:none" class="rec-card">
      <div id="rec-title" style="font-weight:bold;margin-bottom:4px"></div>
      <div id="rec-reasoning" style="color:#aaa;margin-bottom:6px"></div>
      <div id="rec-warnings"></div>
    </div>
    <div class="frow" style="margin-bottom:10px">
      <div class="fg">
        <label>GPU <span id="gpu-hint" style="color:#444;font-size:10px"></span></label>
        <input id="sw-gpu" type="text" placeholder="auto" style="width:100px">
      </div>
    </div>
    <div class="cfg-grid" id="switch-cfg-grid"></div>
    <details style="margin-top:10px">
      <summary>Performance tips</summary>
      <div id="perf-tips" style="margin-top:6px"></div>
    </details>
    <div style="margin-top:12px;display:flex;gap:7px;align-items:center">
      <button class="primary" onclick="doSwitch()">Switch &amp; restart</button>
      <button onclick="document.getElementById('switch-section').style.display='none'">Cancel</button>
      <span id="switch-msg" class="msg"></span>
    </div>
  </div>
</section>

<!-- LAUNCH EXTRA PANEL -->
<section id="launch-section" style="display:none">
  <h2>Launch extra model — <span id="launch-target-label" style="color:#90caf9"></span></h2>
  <div class="card">
    <div class="frow" style="margin-bottom:10px">
      <div class="fg">
        <label>GPU <span style="color:#444;font-size:10px">(blank = auto)</span></label>
        <input id="launch-gpu" type="text" placeholder="auto" style="width:80px">
      </div>
      <div class="fg">
        <label>Port</label>
        <input id="launch-port" type="number" value="8081" style="width:80px">
      </div>
    </div>
    <div class="cfg-grid" id="launch-cfg-grid"></div>
    <div style="margin-top:12px;display:flex;gap:7px;align-items:center">
      <button class="success" onclick="doLaunch()">Launch</button>
      <button onclick="document.getElementById('launch-section').style.display='none'">Cancel</button>
      <span id="launch-msg" class="msg"></span>
    </div>
  </div>
</section>

<section>
  <h2>Download model</h2>
  <div class="card">
    <div class="frow">
      <div class="fg grow">
        <label>HuggingFace repo ID <span style="color:#444">(owner/name)</span></label>
        <input id="dl-repo" type="text" placeholder="unsloth/Qwen3.6-35B-A3B-GGUF">
      </div>
      <div class="fg" style="align-self:flex-end">
        <button onclick="browseRepo()">Browse files</button>
      </div>
      <div class="fg grow">
        <label>Filename</label>
        <input id="dl-file" type="text" placeholder="model.gguf" list="repo-files-list">
        <datalist id="repo-files-list"></datalist>
      </div>
      <div class="fg grow">
        <label>HF Token <span style="color:#444">(optional)</span></label>
        <input id="dl-token" type="password" placeholder="hf_...">
      </div>
      <div class="fg" style="align-self:flex-end">
        <button class="primary" onclick="startDownload()">Download</button>
      </div>
    </div>
    <div id="repo-browse" style="display:none;margin-top:10px">
      <table>
        <thead><tr><th>Filename</th><th>Size</th><th>GPU fit</th><th></th></tr></thead>
        <tbody id="repo-files-body"></tbody>
      </table>
    </div>
    <div id="dl-msg" class="msg"></div>
  </div>
</section>

<section>
  <h2>Downloads</h2>
  <div id="dl-empty" style="color:#444;font-size:12px">No downloads.</div>
  <table id="dl-table" style="display:none">
    <thead><tr><th>File</th><th>Progress</th><th>Status</th><th></th></tr></thead>
    <tbody id="dl-body"></tbody>
  </table>
</section>

<section>
  <h2>Primary server config &amp; restart</h2>
  <div class="card">
    <div class="cfg-grid" id="cfg-grid"></div>
    <div style="margin-top:10px;display:flex;gap:7px;align-items:center">
      <button class="primary" onclick="restartServer()">Apply &amp; restart</button>
      <span id="cfg-msg" class="msg"></span>
    </div>
  </div>
</section>

<section>
  <h2>HuggingFace token</h2>
  <div class="card">
    <div class="frow">
      <div class="fg grow">
        <label id="token-label">checking...</label>
        <input id="token-input" type="password" placeholder="hf_...">
      </div>
      <div class="fg" style="align-self:flex-end"><button class="primary" onclick="saveToken()">Save</button></div>
      <div class="fg" style="align-self:flex-end"><button class="danger" onclick="deleteToken()">Remove</button></div>
    </div>
    <div id="token-msg" class="msg"></div>
  </div>
</section>

<section>
  <h2>Logs &nbsp;
    <select id="log-port" onchange="fetchLogs()" style="width:90px;padding:1px 4px">
      <option value="8080">port 8080</option>
    </select>
    &nbsp;last
    <select id="log-lines" onchange="fetchLogs()" style="width:55px;padding:1px 4px">
      <option>30</option><option selected>60</option><option>120</option>
    </select> lines
  </h2>
  <pre id="log-out">Loading...</pre>
</section>

</div>
<script>
const CFG_FIELDS = [
  {key:'context',        label:'Context',        step:1024,  type:'number'},
  {key:'temp',           label:'Temperature',    step:0.05,  type:'number'},
  {key:'top_k',          label:'Top-K',          step:1,     type:'number'},
  {key:'top_p',          label:'Top-P',          step:0.05,  type:'number'},
  {key:'min_p',          label:'Min-P',          step:0.01,  type:'number'},
  {key:'repeat_penalty', label:'Repeat penalty', step:0.05,  type:'number'},
  {key:'np',             label:'Parallel slots', step:1,     type:'number'},
  {key:'batch_size',     label:'Batch size',     step:512,   type:'number'},
];

let currentCfg = {};
let switchTarget = null;
let launchTarget = null;
let fastPoll = false;
let _pollTimer = null;

function msg(id, text, type='info') {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text; el.className = 'msg ' + type; el.style.display = '';
  if (type === 'ok') setTimeout(()=>el.style.display='none', 4000);
}
function fmtSize(b){ return b>1e9?(b/1e9).toFixed(1)+' GB':b>1e6?(b/1e6).toFixed(0)+' MB':'0'; }
function fmtMib(m){ return m>=1024?(m/1024).toFixed(1)+' GB':m+' MiB'; }
function stateBadge(s, msgStr) {
  const labels = {running:'running',loading:'loading...',
    stopping:'stopping...',clearing_vram:'freeing VRAM...',error:'error',unknown:'unknown'};
  const label = labels[s] || s;
  return `<span class="state-badge state-${s}">${label}${msgStr&&s!=='running'?' — '+msgStr:''}</span>`;
}

// ── status ─────────────────────────────────────────────────────────────────
async function fetchStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    currentCfg = d.config || {};
    const ss = d.server_state || {};
    const st = ss.status || 'unknown';
    const v = d.vram || {};

    const hdr = document.getElementById('hdr-state');
    hdr.textContent = {running:'running',loading:'loading...',
      stopping:'restarting...',clearing_vram:'restarting...',error:'error',unknown:'unknown'}[st] || st;
    hdr.className = 'state-badge state-' + st;

    fastPoll = (st !== 'running' && st !== 'unknown' && st !== 'error');

    const gpuRows = (v.gpus || []).map(g => {
      const pct = Math.round(g.used_mib / g.total_mib * 100);
      const cls = pct > 90 ? 'full' : pct > 70 ? 'warn' : 'ok';
      const assign = g.assigned_model
        ? `<span style="color:#66bb6a;font-size:10px"> ← ${g.assigned_model} <span style="color:#555">port ${g.assigned_port}</span></span>`
        : `<span style="color:#333;font-size:10px"> ← idle</span>`;
      return `<div class="kv gpu-row">
        <label>GPU ${g.id} — ${g.name}</label>
        <span>${g.used_mib.toLocaleString()} / ${g.total_mib.toLocaleString()} MiB (${g.free_mib.toLocaleString()} free)${assign}</span>
        <div class="vbar"><div class="vbar-fill ${cls}" style="width:${pct}%"></div></div>
      </div>`;
    }).join('');

    document.getElementById('status-card').innerHTML = `
      <div class="row">
        <div class="kv"><label>Server</label>${stateBadge(st, ss.message)}</div>
        <div class="kv"><label>Active model</label><span>${ss.model ? ss.model.split('/').pop() : '—'}</span></div>
        <div class="kv"><label>Context</label><span>${(currentCfg.context||0).toLocaleString()} tok</span></div>
        <div class="kv"><label>Temperature</label><span>${currentCfg.temp ?? '—'}</span></div>
        ${gpuRows}
      </div>`;

    buildCfgGrid('cfg-grid', currentCfg, 'cfg-');
    document.getElementById('refresh-ts').textContent = new Date().toLocaleTimeString();
  } catch(e) {}
}

// ── servers table ─────────────────────────────────────────────────────────
async function fetchServers() {
  try {
    const r = await fetch('/api/servers');
    const d = await r.json();
    const rows = [];
    const p = d.primary || {};
    rows.push(`<tr>
      <td>8080 <span style="color:#555;font-size:10px">primary</span></td>
      <td>${p.model ? p.model.split('/').pop() : '—'}</td>
      <td>${p.gpu !== null && p.gpu !== undefined ? 'GPU '+p.gpu : 'all'}</td>
      <td>${stateBadge(p.status||'unknown', p.message)}</td>
      <td></td>
    </tr>`);
    for (const [port, s] of Object.entries(d.extra || {})) {
      rows.push(`<tr>
        <td>${port}</td>
        <td>${s.model ? s.model.split('/').pop() : '—'}</td>
        <td>GPU ${s.gpu}</td>
        <td>${stateBadge(s.status, s.message)}</td>
        <td><button class="danger" onclick="stopExtra(${port})">Stop</button></td>
      </tr>`);
      const sel = document.getElementById('log-port');
      if (!sel.querySelector(`option[value="${port}"]`)) {
        const o = document.createElement('option');
        o.value = port; o.textContent = 'port '+port; sel.appendChild(o);
      }
    }
    document.getElementById('servers-body').innerHTML = rows.join('');
  } catch(e) {}
}

async function stopExtra(port) {
  if (!confirm('Stop server on port '+port+'?')) return;
  const r = await fetch('/api/servers/'+port, {method:'DELETE'});
  const d = await r.json();
  msg('servers-msg', d.message||d.error, r.ok?'ok':'err');
  fetchServers();
}

// ── models table ──────────────────────────────────────────────────────────
async function fetchModels() {
  try {
    const r = await fetch('/api/models');
    const models = await r.json();
    if (!models.length) {
      document.getElementById('models-body').innerHTML =
        '<tr><td colspan="5" style="color:#444">No models in /models</td></tr>';
      return;
    }
    document.getElementById('models-body').innerHTML = models.map(m => {
      const fit = m.fit || {};
      // Find best single-GPU candidate (most headroom among fitting GPUs)
      const bestGpu = (fit.gpus||[]).filter(g=>g.fits)
                        .sort((a,b)=>b.headroom_mib-a.headroom_mib)[0];
      let fitLabel;
      if (bestGpu) {
        fitLabel = `<span class="fit-ok">✓ GPU ${bestGpu.id} &nbsp;<span style="color:#555">`
                 + `${fmtMib(bestGpu.headroom_mib)} free</span></span>`;
      } else if (fit.fits_multi_gpu) {
        fitLabel = `<span class="fit-warn">⚠ multi-GPU only</span>`;
      } else {
        fitLabel = `<span class="fit-no">✗ won't fit</span>`;
      }

      // Free GPU for launching a second instance (fits AND not occupied by this model)
      const freeGpu = (fit.gpus||[]).find(g => g.fits && (!m.active_on || m.active_on !== 8080));

      return `<tr>
        <td>${m.filename}</td>
        <td style="color:#666">${m.size_gb} GB</td>
        <td>${fitLabel}</td>
        <td>${m.active_on ? '<span style="color:#66bb6a">port '+m.active_on+'</span>' : ''}</td>
        <td style="text-align:right;white-space:nowrap">
          ${!m.active_on ? `<button onclick="openSwitch('${m.filename}')">Switch primary</button> ` : ''}
          ${freeGpu && !m.active_on ? `<button class="success" onclick="openLaunch('${m.filename}')">+ extra on GPU ${freeGpu.id}</button> ` : ''}
          ${m.active_on ? `<button class="success" onclick="openLaunch('${m.filename}')">+ extra instance</button> ` : ''}
          ${!m.active_on ? `<button class="danger" onclick="deleteModel('${m.filename}')">Delete</button>` : ''}
        </td>
      </tr>`;
    }).join('');
  } catch(e) {}
}

// ── switch ────────────────────────────────────────────────────────────────
async function openSwitch(filename) {
  switchTarget = filename;
  document.getElementById('switch-target-label').textContent = filename;
  document.getElementById('switch-section').style.display = '';
  document.getElementById('switch-section').scrollIntoView({behavior:'smooth',block:'nearest'});
  buildCfgGrid('switch-cfg-grid', currentCfg, 'sw-');
  document.getElementById('switch-msg').style.display = 'none';
  document.getElementById('rec-card').style.display = 'none';
  document.getElementById('perf-tips').innerHTML = '';
  document.getElementById('sw-gpu').value = '';
  document.getElementById('gpu-hint').textContent = 'fetching recommendation...';

  try {
    const ctx = currentCfg.context || 131072;
    const r = await fetch(`/api/recommend?filename=${encodeURIComponent(filename)}&context=${ctx}`);
    if (r.ok) showRecommendation(await r.json());
  } catch(e) {
    document.getElementById('gpu-hint').textContent = '';
  }
}

function showRecommendation(rec) {
  const card = document.getElementById('rec-card');
  const strats = {
    single_gpu: {cls:'rec-single', icon:'✓', label:'Single GPU recommended'},
    multi_gpu:  {cls:'rec-multi',  icon:'⚠', label:'Multi-GPU required'},
    does_not_fit:{cls:'rec-nope', icon:'✗', label:'Insufficient VRAM'},
  };
  const s = strats[rec.strategy] || strats.does_not_fit;
  card.className = 'rec-card ' + s.cls;
  document.getElementById('rec-title').textContent = s.icon + ' ' + s.label;
  document.getElementById('rec-reasoning').textContent = rec.reasoning;
  document.getElementById('rec-warnings').innerHTML = (rec.warnings||[])
    .map(w=>`<div style="color:#aaa;margin-top:2px">→ ${w}</div>`).join('');

  if (rec.recommended_gpu !== null && rec.recommended_gpu !== undefined) {
    document.getElementById('sw-gpu').value = rec.recommended_gpu;
    document.getElementById('gpu-hint').textContent = `(recommended: GPU ${rec.recommended_gpu})`;
  } else {
    document.getElementById('sw-gpu').value = '';
    document.getElementById('gpu-hint').textContent = rec.strategy === 'multi_gpu' ? '(all GPUs)' : '';
  }

  const tipColors = {ok:'#66bb6a', warn:'#ffa726', info:'#90caf9'};
  const tipIcons = {ok:'✓', warn:'⚠', info:'ℹ'};
  document.getElementById('perf-tips').innerHTML = (rec.tips||[]).map(t =>
    `<div class="tip tip-${t.level}">${tipIcons[t.level]||'·'} ${t.text}</div>`
  ).join('');

  card.style.display = '';
}

async function doSwitch() {
  if (!switchTarget) return;
  const gpuVal = document.getElementById('sw-gpu').value.trim();
  const body = {filename: switchTarget, ...readCfgGrid('sw-')};
  if (gpuVal !== '') body.gpu = parseInt(gpuVal);
  const btn = event.target; btn.disabled = true;
  try {
    const r = await fetch('/api/switch', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const d = await r.json();
    if (r.ok) {
      msg('switch-msg', `Switching to GPU ${d.gpu !== null ? d.gpu : 'all'} — restarting...`, 'info');
      fastPoll = true;
      document.getElementById('switch-section').style.display = 'none';
      scheduleRefresh();
    } else {
      msg('switch-msg', d.error||'Error', 'err');
    }
  } catch(e) { msg('switch-msg', e.message, 'err'); }
  btn.disabled = false;
}

// ── launch extra ──────────────────────────────────────────────────────────
function openLaunch(filename) {
  launchTarget = filename;
  document.getElementById('launch-target-label').textContent = filename;
  document.getElementById('launch-section').style.display = '';
  document.getElementById('launch-section').scrollIntoView({behavior:'smooth',block:'nearest'});
  buildCfgGrid('launch-cfg-grid', currentCfg, 'lnc-');
  document.getElementById('launch-msg').style.display = 'none';
  document.getElementById('launch-gpu').value = '';
}

async function doLaunch() {
  if (!launchTarget) return;
  const gpuVal = document.getElementById('launch-gpu').value.trim();
  const port = parseInt(document.getElementById('launch-port').value);
  const body = {filename: launchTarget, port, ...readCfgGrid('lnc-')};
  if (gpuVal !== '') body.gpu = parseInt(gpuVal);
  const btn = event.target; btn.disabled = true;
  try {
    const r = await fetch('/api/servers', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const d = await r.json();
    if (r.ok) {
      msg('launch-msg', `Launching on GPU ${d.gpu} port ${d.port}...`, 'info');
      document.getElementById('launch-section').style.display = 'none';
    } else {
      msg('launch-msg', d.error||'Error', 'err');
    }
  } catch(e) { msg('launch-msg', e.message, 'err'); }
  btn.disabled = false;
}

async function deleteModel(filename) {
  if (!confirm('Delete '+filename+'?')) return;
  const r = await fetch('/api/models/'+encodeURIComponent(filename), {method:'DELETE'});
  const d = await r.json();
  msg('models-msg', d.message||d.error, r.ok?'ok':'err');
  fetchModels();
}

// ── cfg grid ──────────────────────────────────────────────────────────────
function buildCfgGrid(gridId, cfg, prefix) {
  const el = document.getElementById(gridId);
  if (!el) return;
  el.innerHTML = CFG_FIELDS.map(f=>`
    <div class="fg">
      <label>${f.label}</label>
      <input type="${f.type}" step="${f.step}" id="${prefix}${f.key}" value="${cfg[f.key]??''}">
    </div>`).join('');
}
function readCfgGrid(prefix) {
  const out = {};
  CFG_FIELDS.forEach(f => {
    const el = document.getElementById(prefix+f.key);
    if (el && el.value !== '') out[f.key] = Number(el.value);
  });
  return out;
}

// ── download ──────────────────────────────────────────────────────────────
async function browseRepo() {
  const repo = document.getElementById('dl-repo').value.trim().split(':')[0];
  if (!repo) { msg('dl-msg','Enter repo ID first','err'); return; }
  msg('dl-msg','Fetching file list...','info');
  try {
    const r = await fetch('/api/repo/files?repo_id='+encodeURIComponent(repo));
    const d = await r.json();
    if (!r.ok) { msg('dl-msg', d.error||'Not found', 'err'); return; }
    const gpus = await (await fetch('/api/gpus')).json();
    const gpu0free = gpus[0]?.free_mib || 0;
    document.getElementById('repo-files-list').innerHTML =
      d.files.map(f=>`<option value="${f.filename}">`).join('');
    document.getElementById('repo-files-body').innerHTML = d.files.map(f => {
      const needMib = f.size_bytes ? Math.round(f.size_bytes/1024/1024) + 1400 : null;
      const fits = needMib ? needMib <= gpu0free : null;
      const fitLabel = fits === null ? '?' : fits
        ? `<span class="fit-ok">✓ GPU 0</span>`
        : `<span class="fit-no">✗ GPU 0 full</span>`;
      return `<tr>
        <td>${f.filename}</td>
        <td style="color:#666">${f.size_gb ?? '?'} GB</td>
        <td>${fitLabel}</td>
        <td><button onclick="
          document.getElementById('dl-file').value='${f.filename}';
          document.getElementById('repo-browse').style.display='none';
          document.getElementById('dl-msg').style.display='none'
        ">Select</button></td>
      </tr>`;
    }).join('');
    document.getElementById('repo-browse').style.display = '';
    document.getElementById('dl-msg').style.display = 'none';
  } catch(e) { msg('dl-msg', e.message, 'err'); }
}

async function startDownload() {
  const repo = document.getElementById('dl-repo').value.trim();
  const file = document.getElementById('dl-file').value.trim();
  const token = document.getElementById('dl-token').value.trim();
  if (!repo || !file) { msg('dl-msg','Repo ID and filename required','err'); return; }
  const body = {repo_id: repo.split(':')[0], filename: file};
  if (token) body.token = token;
  const btn = event.target; btn.disabled = true;
  try {
    const r = await fetch('/api/download', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const d = await r.json();
    msg('dl-msg', r.ok ? 'Download started: '+file : (d.error||'Error'), r.ok?'ok':'err');
    if (r.ok) document.getElementById('dl-file').value = '';
  } catch(e) { msg('dl-msg', e.message, 'err'); }
  btn.disabled = false;
}

async function fetchDownloads() {
  try {
    const r = await fetch('/api/downloads');
    const list = await r.json();
    document.getElementById('dl-empty').style.display = list.length ? 'none' : '';
    document.getElementById('dl-table').style.display = list.length ? '' : 'none';
    document.getElementById('dl-body').innerHTML = list.map(d => {
      const pct = d.progress_pct ?? 0;
      const col = d.status==='done'?'#66bb6a':d.status==='error'?'#ef5350':d.status==='cancelled'?'#666':'#90caf9';
      return `<tr>
        <td>${d.filename}</td>
        <td style="min-width:140px">${fmtSize(d.downloaded)} / ${fmtSize(d.total)}
          <div class="pb"><div class="pb-fill" style="width:${pct}%"></div></div></td>
        <td><span style="color:${col}">${d.status}${d.status==='downloading'?' '+pct+'%':''}</span>
          ${d.error?`<br><span style="color:#666;font-size:10px">${d.error}</span>`:''}</td>
        <td>${d.status==='downloading'?`<button class="danger" onclick="cancelDl('${d.id}')">Cancel</button>`:''}</td>
      </tr>`;
    }).join('');
  } catch(e) {}
}

async function cancelDl(id) {
  await fetch('/api/downloads/'+id, {method:'DELETE'});
  fetchDownloads();
}

async function restartServer() {
  const body = readCfgGrid('cfg-');
  const btn = event.target; btn.disabled = true;
  try {
    const r = await fetch('/api/server/restart', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const d = await r.json();
    msg('cfg-msg', r.ok ? 'Restarting...' : (d.error||'Error'), r.ok?'info':'err');
    if (r.ok) { fastPoll = true; scheduleRefresh(); }
  } catch(e) { msg('cfg-msg', e.message, 'err'); }
  btn.disabled = false;
}

async function fetchToken() {
  const r = await fetch('/api/token');
  const d = await r.json();
  document.getElementById('token-label').textContent = d.token_set ? 'Saved token (set)' : 'No token saved';
}
async function saveToken() {
  const token = document.getElementById('token-input').value.trim();
  if (!token) { msg('token-msg','Enter a token','err'); return; }
  const r = await fetch('/api/token', {method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify({token})});
  const d = await r.json();
  msg('token-msg', d.message||d.error, r.ok?'ok':'err');
  document.getElementById('token-input').value='';
  fetchToken();
}
async function deleteToken() {
  await fetch('/api/token', {method:'DELETE'});
  msg('token-msg','Token removed','ok'); fetchToken();
}

async function fetchLogs() {
  const n = document.getElementById('log-lines').value;
  const p = document.getElementById('log-port').value;
  try {
    const r = await fetch(`/api/server/logs?lines=${n}&port=${p}`);
    const d = await r.json();
    const pre = document.getElementById('log-out');
    pre.textContent = d.lines?.join('\n') || '(empty)';
    pre.scrollTop = pre.scrollHeight;
  } catch(e) {}
}

// ── poll loop (dynamic interval) ──────────────────────────────────────────
async function refresh() {
  await Promise.all([fetchStatus(), fetchServers(), fetchModels(),
                     fetchDownloads(), fetchToken(), fetchLogs()]);
}

function scheduleRefresh() {
  clearTimeout(_pollTimer);
  _pollTimer = setTimeout(async () => {
    await refresh();
    scheduleRefresh();
  }, fastPoll ? 2000 : 5000);
}

refresh().then(scheduleRefresh);
</script>
</body>
</html>"""


@app.get("/")
def ui():
    return render_template_string(UI_HTML)


if __name__ == "__main__":
    if not CONFIG_FILE.exists():
        save_config(dict(DEFAULT_CONFIG))
    app.run(host="0.0.0.0", port=5000, threaded=True)
