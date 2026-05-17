import os
import json
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)

MODELS_DIR = Path("/models")
HF_HUB_DIR = MODELS_DIR / "hub"
CONFIG_FILE = Path("/root/vllm_config.json")
TOKEN_FILE = Path("/root/.hf_token")
ONSTART = Path("/root/onstart.sh")
PRIMARY_PORT = 8000

downloads = {}
_dl_lock = threading.Lock()

_servers_lock = threading.Lock()
primary_state = {
    "port": PRIMARY_PORT, "gpus": None,
    "status": "unknown",
    "message": "", "repo_id": None, "started_at": None,
}
extra_servers = {}

DEFAULT_CONFIG = {
    "repo_id": None,
    "served_model_name": None,
    "max_model_len": 20480,
    "max_num_batched_tokens": 4096,
    "max_num_seqs": 2,
    "gpu_memory_utilization": 0.92,
    "kv_cache_dtype": "fp8",
    "cpu_offload_gb": 0,
    "enable_prefix_caching": True,
    "enable_auto_tool_choice": True,
    "reasoning_parser": "qwen3",
    "tool_call_parser": "qwen3_coder",
    "tensor_parallel_size": 1,
}

# ── config helpers ────────────────────────────────────────────────────────────

def load_config():
    if CONFIG_FILE.exists():
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(json.loads(CONFIG_FILE.read_text()))
        return cfg
    return dict(DEFAULT_CONFIG)

def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

def load_token():
    return TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else None

def sanitize_repo_id(repo_id):
    return repo_id.split(":")[0].strip()

def repo_to_cache_dir(repo_id):
    """HF cache layout: hub/models--<org>--<name>/snapshots/<rev>/..."""
    return HF_HUB_DIR / f"models--{sanitize_repo_id(repo_id).replace('/', '--')}"

def repo_size_bytes(repo_id):
    """Sum of all snapshot files for a cached repo."""
    d = repo_to_cache_dir(repo_id)
    if not d.exists():
        return 0
    total = 0
    snap = d / "snapshots"
    if snap.exists():
        for p in snap.rglob("*"):
            if p.is_file() and not p.is_symlink():
                try:
                    total += p.stat().st_size
                except Exception:
                    pass
            elif p.is_symlink():
                try:
                    total += p.resolve().stat().st_size
                except Exception:
                    pass
    return total

def list_cached_repos():
    """Return [{repo_id, size_bytes, size_gb}] for repos in the HF cache."""
    if not HF_HUB_DIR.exists():
        return []
    out = []
    for d in HF_HUB_DIR.iterdir():
        if not d.is_dir() or not d.name.startswith("models--"):
            continue
        repo_id = d.name[len("models--"):].replace("--", "/", 1)
        # Replace remaining "--" with "-" if any (rare; HF org/name shouldn't have it)
        # Heuristic: only first "--" is the org/name split
        size = repo_size_bytes(repo_id)
        if size == 0:
            continue
        out.append({
            "repo_id": repo_id,
            "size_bytes": size,
            "size_gb": round(size / 1e9, 2),
        })
    return sorted(out, key=lambda r: r["repo_id"])

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

def detect_quant(repo_id):
    n = repo_id.lower()
    if "awq" in n: return "awq"
    if "gptq" in n: return "gptq"
    if "fp8" in n: return "fp8"
    if "int4" in n or "4bit" in n or "w4" in n: return "int4"
    if "int8" in n or "8bit" in n: return "int8"
    return "fp16"

def detect_arch(repo_id):
    n = repo_id.lower()
    if "qwen3" in n:
        return "moe_mamba"  # Qwen3.x hybrid Transformer-Mamba MoE
    if "mixtral" in n or "moe" in n:
        return "moe"
    if "mamba" in n:
        return "mamba"
    return "dense"

# Estimate model weight VRAM footprint given on-disk size + quant.
def estimate_weights_mib(repo_id):
    size = repo_size_bytes(repo_id)
    if size == 0:
        return 0
    mib = size // (1024 * 1024)
    # AWQ/GPTQ on disk ≈ in-VRAM weights. fp16 same. fp8 same. No multiplier needed.
    return mib

def estimate_kv_overhead_mib(arch, max_model_len, kv_cache_dtype, max_num_seqs):
    """KV cache scales with context × concurrent sequences. fp8 halves it."""
    ctx_k = max_model_len / 1024
    if arch in ("moe_mamba", "mamba"):
        per_seq = max(200, int(ctx_k * 5.5))  # Qwen3.x: ~half of fp16 dense
    elif arch == "moe":
        per_seq = max(300, int(ctx_k * 12))
    else:
        per_seq = max(400, int(ctx_k * 40))
    if kv_cache_dtype == "fp8":
        per_seq = per_seq // 2
    return per_seq * max(1, max_num_seqs) + 1500  # +1.5 GB activation buffer

def poll_health(port, timeout=600):
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

def estimate_fit(repo_id, cfg=None):
    cfg = cfg or load_config()
    arch = detect_arch(repo_id)
    weights_mib = estimate_weights_mib(repo_id)
    overhead_mib = estimate_kv_overhead_mib(
        arch,
        cfg.get("max_model_len", 20480),
        cfg.get("kv_cache_dtype", "fp8"),
        cfg.get("max_num_seqs", 2),
    )
    needed_mib = weights_mib + overhead_mib

    gpus = get_gpus()
    total_free = sum(g["free_mib"] for g in gpus)
    util = cfg.get("gpu_memory_utilization", 0.92)

    gpu_analysis = []
    for g in gpus:
        usable = int(g["total_mib"] * util) - max(0, g["used_mib"])
        gpu_analysis.append({
            "id": g["id"], "name": g["name"],
            "free_mib": g["free_mib"], "total_mib": g["total_mib"],
            "usable_mib": usable,
            "fits": usable >= needed_mib,
            "headroom_mib": usable - needed_mib,
        })

    # Tensor parallelism (TP) splits weights across N GPUs.
    # KV cache also splits. Needed per-GPU = needed / N.
    tp_options = []
    n_gpus = len(gpus)
    for tp in (1, 2, 4, 8):
        if tp > n_gpus:
            break
        if n_gpus % tp != 0:
            continue
        per_gpu_needed = needed_mib // tp
        min_usable = min((int(g["total_mib"] * util) for g in gpus[:tp]), default=0)
        tp_options.append({
            "tp": tp,
            "per_gpu_needed_mib": per_gpu_needed,
            "fits": min_usable >= per_gpu_needed,
        })

    return {
        "arch": arch,
        "quant": detect_quant(repo_id),
        "weights_mib": weights_mib,
        "weights_gb": round(weights_mib / 1024, 2),
        "overhead_mib": overhead_mib,
        "needed_mib": needed_mib,
        "total_free_mib": total_free,
        "gpus": gpu_analysis,
        "tp_options": tp_options,
        "fits_single_gpu": any(a["fits"] for a in gpu_analysis),
        "recommended_tp": (
            1 if any(a["fits"] for a in gpu_analysis)
            else next((o["tp"] for o in tp_options if o["fits"]), None)
        ),
    }

def _active_repos():
    """Map repo_id → port for currently-running vllm serve processes."""
    result = {}
    r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "vllm" not in line or "serve" not in line:
            continue
        # Find the positional repo argument (token after "serve")
        parts = line.split()
        try:
            i = parts.index("serve")
            repo_id = parts[i + 1]
        except (ValueError, IndexError):
            continue
        port = PRIMARY_PORT
        if "--port" in parts:
            try:
                port = int(parts[parts.index("--port") + 1])
            except Exception:
                pass
        result[repo_id] = port
    return result

def recommend_deployment(repo_id, cfg=None, assume_switch=False):
    cfg = cfg or load_config()
    gpus = get_gpus()
    if assume_switch:
        # After we pkill vllm, GPUs will be ~empty.
        for g in gpus:
            g["free_mib"] = g["total_mib"] - 400
            g["used_mib"] = 400

    arch = detect_arch(repo_id)
    weights_mib = estimate_weights_mib(repo_id)
    overhead_mib = estimate_kv_overhead_mib(
        arch,
        cfg.get("max_model_len", 20480),
        cfg.get("kv_cache_dtype", "fp8"),
        cfg.get("max_num_seqs", 2),
    )
    needed_mib = weights_mib + overhead_mib
    util = cfg.get("gpu_memory_utilization", 0.92)

    # Build GPU analysis using adjusted free VRAM
    gpu_analysis = []
    for g in gpus:
        usable = int(g["total_mib"] * util) - max(0, g["used_mib"])
        gpu_analysis.append({
            "id": g["id"], "name": g["name"],
            "free_mib": g["free_mib"], "total_mib": g["total_mib"],
            "usable_mib": usable,
            "fits": usable >= needed_mib,
            "headroom_mib": usable - needed_mib,
        })
    total_free = sum(g["free_mib"] for g in gpus)

    occupied = {g["id"] for g in gpus if g["used_mib"] > 2000}
    candidates = sorted(
        [g for g in gpu_analysis if g["fits"]],
        key=lambda g: (g["id"] in occupied, -g["headroom_mib"])
    )

    fit = {
        "arch": arch, "quant": detect_quant(repo_id),
        "weights_mib": weights_mib, "weights_gb": round(weights_mib / 1024, 2),
        "overhead_mib": overhead_mib, "needed_mib": needed_mib,
        "total_free_mib": total_free, "gpus": gpu_analysis,
    }

    warnings = []
    n_gpus = len(gpus)

    if candidates:
        best = candidates[0]
        others = [g for g in gpus if g["id"] != best["id"]]
        for og in others:
            if og["id"] not in occupied:
                warnings.append(
                    f"GPU {og['id']} stays idle ({og['free_mib']:,} MiB free) — "
                    "use 'Launch extra model' to run a second model there"
                )
        return {
            "strategy": "single_gpu",
            "tensor_parallel_size": 1,
            "recommended_gpus": [best["id"]],
            "headroom_mib": best["headroom_mib"],
            "reasoning": (
                f"Fits on GPU {best['id']} with {best['headroom_mib']:,} MiB to spare "
                f"(at {int(util*100)}% util). Single GPU avoids tensor-parallel overhead."
            ),
            "warnings": warnings,
            "fit": fit,
        }

    # Try tensor parallelism across all available GPUs
    for tp in (2, 4, 8):
        if tp > n_gpus or n_gpus % tp != 0:
            continue
        per_gpu_needed = needed_mib // tp
        # Verify each candidate GPU can hold its shard
        chosen = sorted(gpus, key=lambda g: -(g["total_mib"] * util - g["used_mib"]))[:tp]
        min_usable = min(int(g["total_mib"] * util) - g["used_mib"] for g in chosen)
        if min_usable >= per_gpu_needed:
            gids = [g["id"] for g in chosen]
            warnings.append(
                f"No single GPU fits the model — splitting weights across "
                f"{tp} GPUs via tensor parallelism."
            )
            warnings.append(
                "vLLM tensor parallelism works for MoE (unlike llama.cpp). "
                "Expect ~5–15% latency overhead vs single GPU."
            )
            return {
                "strategy": "tensor_parallel",
                "tensor_parallel_size": tp,
                "recommended_gpus": gids,
                "headroom_mib": min_usable - per_gpu_needed,
                "reasoning": (
                    f"Model needs {needed_mib:,} MiB total. Tensor-parallel split "
                    f"across {tp} GPUs ({per_gpu_needed:,} MiB each), {min_usable - per_gpu_needed:,} "
                    "MiB headroom per GPU."
                ),
                "warnings": warnings,
                "fit": fit,
            }

    return {
        "strategy": "does_not_fit",
        "tensor_parallel_size": None,
        "recommended_gpus": [],
        "headroom_mib": total_free - needed_mib,
        "reasoning": (
            f"Insufficient VRAM: needs ~{needed_mib:,} MiB, "
            f"only {total_free:,} MiB free across all GPUs. "
            f"Consider --cpu-offload-gb, a smaller quant (AWQ/GPTQ), or fewer max_num_seqs."
        ),
        "warnings": ["Lower max_model_len or max_num_seqs to reduce KV cache.",
                     "Stop other running models to free VRAM first."],
        "fit": fit,
    }

def perf_tips(repo_id, cfg):
    arch = detect_arch(repo_id) if repo_id else "dense"
    quant = detect_quant(repo_id) if repo_id else "fp16"
    gpus = get_gpus()
    tips = []
    ctx = cfg.get("max_model_len", 20480)
    seqs = cfg.get("max_num_seqs", 2)
    util = cfg.get("gpu_memory_utilization", 0.92)
    kv = cfg.get("kv_cache_dtype", "fp8")
    tp = cfg.get("tensor_parallel_size", 1)

    if arch == "moe_mamba":
        tips.append({"level": "info",
                     "text": "Qwen3.x hybrid Mamba-MoE: tiny KV cache compared to dense models. "
                             "Long contexts (65K+) are cheap. Bump --max-model-len if you have VRAM."})
    if arch in ("moe", "moe_mamba") and tp > 1:
        tips.append({"level": "ok",
                     "text": "vLLM tensor parallelism works for MoE (unlike llama.cpp pipeline-only fallback)."})

    if quant in ("awq", "gptq", "int4"):
        tips.append({"level": "ok",
                     "text": f"{quant.upper()} quantization: ~4× memory reduction vs fp16, "
                             "~1–2% perplexity cost. Best size/quality tradeoff for 30B+."})

    if kv == "fp8":
        tips.append({"level": "ok",
                     "text": "--kv-cache-dtype fp8: halves KV memory, <1% quality cost. "
                             "Enables longer context or more concurrent sequences."})
    else:
        tips.append({"level": "warn",
                     "text": "--kv-cache-dtype is fp16: switch to fp8 to halve KV memory."})

    if cfg.get("enable_prefix_caching"):
        tips.append({"level": "ok",
                     "text": "--enable-prefix-caching: massive speedup when many requests share a system prompt."})
    else:
        tips.append({"level": "warn",
                     "text": "--enable-prefix-caching is off: enable for huge speedup on repeated prefixes."})

    if util < 0.88:
        tips.append({"level": "warn",
                     "text": f"--gpu-memory-utilization={util}: low utilization leaves VRAM unused. "
                             "0.92–0.95 is typical."})
    elif util > 0.96:
        tips.append({"level": "warn",
                     "text": f"--gpu-memory-utilization={util}: very high — risk of CUDA OOM on long prompts. "
                             "0.92–0.94 is safer."})

    if seqs == 1:
        tips.append({"level": "info",
                     "text": "max_num_seqs=1: optimal for single-user latency. "
                             "Bump to 4–8 if serving multiple clients (uses more KV)."})

    if len(gpus) > 1 and tp == 1:
        tips.append({"level": "info",
                     "text": f"{len(gpus)} GPUs available, tp=1: the other GPU is idle. "
                             "Launch a second model there, or bump tp for higher throughput."})

    if cfg.get("cpu_offload_gb", 0) > 0:
        tips.append({"level": "warn",
                     "text": f"--cpu-offload-gb={cfg['cpu_offload_gb']}: model partially on CPU. "
                             "Major latency hit. Only use if VRAM truly insufficient."})

    return tips

# ── vLLM lifecycle ────────────────────────────────────────────────────────────

def build_vllm_args(cfg, port=PRIMARY_PORT, gpus=None, tp=None):
    repo = cfg["repo_id"]
    tp = tp or cfg.get("tensor_parallel_size", 1)
    args = ["vllm", "serve", repo,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--tensor-parallel-size", str(tp),
            "--max-model-len", str(cfg.get("max_model_len", 20480)),
            "--max-num-batched-tokens", str(cfg.get("max_num_batched_tokens", 4096)),
            "--max-num-seqs", str(cfg.get("max_num_seqs", 2)),
            "--gpu-memory-utilization", str(cfg.get("gpu_memory_utilization", 0.92)),
            "--kv-cache-dtype", cfg.get("kv_cache_dtype", "fp8"),
            "--trust-remote-code",
            "--limit-mm-per-prompt", '{"image":0,"video":0}']

    if cfg.get("served_model_name"):
        args += ["--served-model-name", cfg["served_model_name"]]
    if cfg.get("enable_prefix_caching"):
        args.append("--enable-prefix-caching")
    if cfg.get("enable_auto_tool_choice"):
        args.append("--enable-auto-tool-choice")
    if cfg.get("reasoning_parser"):
        args += ["--reasoning-parser", cfg["reasoning_parser"]]
    if cfg.get("tool_call_parser"):
        args += ["--tool-call-parser", cfg["tool_call_parser"]]
    if cfg.get("cpu_offload_gb", 0) > 0:
        args += ["--cpu-offload-gb", str(cfg["cpu_offload_gb"])]

    env = dict(os.environ)
    env.setdefault("HF_HOME", str(MODELS_DIR))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    return args, env

def write_onstart(cfg, gpus=None, tp=None):
    if not cfg.get("repo_id"):
        return
    tp = tp or cfg.get("tensor_parallel_size", 1)
    gpu_line = f'export CUDA_VISIBLE_DEVICES="{",".join(str(g) for g in gpus)}"' if gpus else ""
    flags = [
        f"  --served-model-name {cfg.get('served_model_name', 'model')} \\",
        "  --host 0.0.0.0 \\",
        f"  --port {PRIMARY_PORT} \\",
        f"  --tensor-parallel-size {tp} \\",
        f"  --max-model-len {cfg.get('max_model_len', 20480)} \\",
        f"  --max-num-batched-tokens {cfg.get('max_num_batched_tokens', 4096)} \\",
        f"  --max-num-seqs {cfg.get('max_num_seqs', 2)} \\",
        f"  --gpu-memory-utilization {cfg.get('gpu_memory_utilization', 0.92)} \\",
        f"  --kv-cache-dtype {cfg.get('kv_cache_dtype', 'fp8')} \\",
    ]
    if cfg.get("enable_prefix_caching"):
        flags.append("  --enable-prefix-caching \\")
    if cfg.get("enable_auto_tool_choice"):
        flags.append("  --enable-auto-tool-choice \\")
    if cfg.get("reasoning_parser"):
        flags.append(f"  --reasoning-parser {cfg['reasoning_parser']} \\")
    if cfg.get("tool_call_parser"):
        flags.append(f"  --tool-call-parser {cfg['tool_call_parser']} \\")
    if cfg.get("cpu_offload_gb", 0) > 0:
        flags.append(f"  --cpu-offload-gb {cfg['cpu_offload_gb']} \\")
    flags.append("  --limit-mm-per-prompt '{\"image\":0,\"video\":0}' \\")
    flags.append("  --trust-remote-code >> /var/log/vllm-server.log 2>&1")

    lines = [
        "#!/bin/sh",
        "export HF_HOME=/models",
        "export HF_HUB_ENABLE_HF_TRANSFER=1",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        gpu_line,
        "python3 /app/model_manager_vllm.py >> /var/log/model-manager.log 2>&1 &",
        f"vllm serve {cfg['repo_id']} \\",
    ] + flags
    ONSTART.write_text("\n".join(l for l in lines if l) + "\n")
    ONSTART.chmod(0o755)

def _do_restart_primary(cfg, gpus, tp):
    with _servers_lock:
        primary_state.update({"status": "stopping", "message": "Stopping current vLLM server...",
                               "repo_id": cfg.get("repo_id"),
                               "gpus": list(gpus) if gpus else None, "tp": tp})
    # vllm forks workers — kill the whole tree
    subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True)

    with _servers_lock:
        primary_state.update({"status": "clearing_vram", "message": "Waiting for VRAM to free..."})
    for _ in range(45):
        time.sleep(2)
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        total_used = sum(int(x.strip()) for x in r.stdout.strip().splitlines() if x.strip().isdigit())
        if total_used < 3000:
            break

    write_onstart(cfg, gpus, tp)
    Path("/var/log/vllm-server.log").write_text("")

    with _servers_lock:
        primary_state.update({"status": "loading",
                               "message": f"Loading {cfg['repo_id']}... (cold start can take 2–4 min)"})
    # Try s6 first (Clore container layout); fall back to direct spawn.
    s6_path = "/var/run/s6/services/onstart"
    if Path(s6_path).exists():
        subprocess.run(["s6-svc", "-r", s6_path], capture_output=True)
    else:
        log = open("/var/log/vllm-server.log", "ab")
        args, env = build_vllm_args(cfg, port=PRIMARY_PORT, gpus=gpus, tp=tp)
        subprocess.Popen(args, env=env, stdout=log, stderr=log)

    if poll_health(PRIMARY_PORT, timeout=600):
        with _servers_lock:
            primary_state.update({"status": "running", "message": "", "started_at": time.time()})
    else:
        with _servers_lock:
            primary_state.update({"status": "error", "message": "Server failed to start — check logs"})

def restart_primary(cfg, gpus=None, tp=None):
    save_config(cfg)
    threading.Thread(target=_do_restart_primary, args=(cfg, gpus, tp), daemon=True).start()

def _do_start_extra(port, repo_id, cfg, gpus):
    with _servers_lock:
        extra_servers[port].update({"status": "loading",
                                    "message": f"Loading {repo_id}..."})
    args, env = build_vllm_args(cfg, port=port, gpus=gpus, tp=len(gpus) if gpus else 1)
    log = open(f"/var/log/vllm-server-{port}.log", "w")
    proc = subprocess.Popen(args, env=env, stdout=log, stderr=log)
    with _servers_lock:
        extra_servers[port]["pid"] = proc.pid
        extra_servers[port]["proc"] = proc

    if poll_health(port, timeout=600):
        with _servers_lock:
            extra_servers[port].update({"status": "running", "message": "", "started_at": time.time()})
    else:
        with _servers_lock:
            extra_servers[port].update({"status": "error", "message": "Failed to start — check logs"})
        proc.kill()

# ── download (HF snapshot) ────────────────────────────────────────────────────

def get_repo_size_remote(repo_id, token=None):
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token or load_token())
        info = api.model_info(sanitize_repo_id(repo_id), files_metadata=True)
        return sum((f.size or 0) for f in (info.siblings or []))
    except Exception:
        return None

def _run_download(dl_id, repo_id, token):
    tok = token or load_token()
    expected = get_repo_size_remote(repo_id, tok)
    with _dl_lock:
        downloads[dl_id]["total"] = expected

    env = dict(os.environ)
    env["HF_HOME"] = str(MODELS_DIR)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    if tok:
        env["HF_TOKEN"] = tok

    cmd = ["huggingface-cli", "download", sanitize_repo_id(repo_id),
           "--cache-dir", str(MODELS_DIR)]

    log_path = Path(f"/var/log/download-{dl_id}.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env)

    with _dl_lock:
        downloads[dl_id]["pid"] = proc.pid
        downloads[dl_id]["log"] = str(log_path)

    while proc.poll() is None:
        try:
            downloaded = repo_size_bytes(repo_id)
            with _dl_lock:
                downloads[dl_id]["downloaded"] = downloaded
        except Exception:
            pass
        time.sleep(2)

    log_file.close()
    proc.wait()
    final_size = repo_size_bytes(repo_id)

    with _dl_lock:
        if proc.returncode == 0 and final_size > 1_000_000:
            downloads[dl_id].update({"status": "done", "downloaded": final_size,
                                     "total": expected or final_size})
        else:
            err_detail = f"exit {proc.returncode}, {final_size} bytes"
            try:
                lines = [l.strip() for l in log_path.read_text().splitlines() if l.strip()]
                if lines:
                    err_detail = lines[-1][:160]
            except Exception:
                pass
            downloads[dl_id]["status"] = "error"
            downloads[dl_id]["error"] = err_detail

# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/api")
def api_index():
    return jsonify({"service": "vllm model manager", "ui": "GET /"})


@app.get("/api/status")
def server_status():
    try:
        r = subprocess.run(["curl", "-sf", f"http://localhost:{PRIMARY_PORT}/health"],
                           capture_output=True, text=True, timeout=3)
        health = {"status": "ok"} if r.returncode == 0 else {"status": "unreachable"}
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
        for g in gpus:
            g["assigned_model"] = None
            g["assigned_port"] = None
            p_gpus = state.get("gpus") or []
            p_repo = state.get("repo_id")
            if p_repo and state["status"] in ("running", "loading", "clearing_vram"):
                if not p_gpus or g["id"] in p_gpus:
                    g["assigned_model"] = p_repo
                    g["assigned_port"] = PRIMARY_PORT
            for port, s in extras.items():
                s_gpus = s.get("gpus") or []
                if g["id"] in s_gpus and s["status"] in ("running", "loading"):
                    g["assigned_model"] = s.get("repo_id")
                    g["assigned_port"] = port

        vram = {
            "gpus": gpus,
            "total_used_mib": sum(g["used_mib"] for g in gpus),
            "total_free_mib": sum(g["free_mib"] for g in gpus),
            "total_mib": sum(g["total_mib"] for g in gpus),
        }

    cfg = load_config()
    return jsonify({"health": health, "server_state": state, "config": cfg,
                    "vram": vram, "extras": extras})


@app.get("/api/server/state")
def get_server_state():
    with _servers_lock:
        return jsonify(dict(primary_state))


@app.get("/api/gpus")
def gpu_info():
    return jsonify(get_gpus())


@app.get("/api/recommend")
def get_recommendation():
    repo_id = request.args.get("repo_id", "").strip()
    if not repo_id:
        return jsonify({"error": "repo_id required"}), 400
    assume_switch = request.args.get("assume_switch", "1") == "1"
    cfg = load_config()
    # Allow overriding key params via query string for what-if exploration
    for key in ("max_model_len", "max_num_seqs", "gpu_memory_utilization", "kv_cache_dtype"):
        v = request.args.get(key)
        if v is not None:
            try:
                cfg[key] = type(DEFAULT_CONFIG[key])(v) if DEFAULT_CONFIG[key] is not None else v
            except Exception:
                pass
    rec = recommend_deployment(repo_id, cfg=cfg, assume_switch=assume_switch)
    rec["tips"] = perf_tips(repo_id, cfg)
    return jsonify(rec)


@app.get("/api/models")
def list_models():
    active = _active_repos()
    cfg = load_config()
    out = []
    for r in list_cached_repos():
        fit = estimate_fit(r["repo_id"], cfg=cfg)
        out.append({
            "repo_id": r["repo_id"],
            "size_bytes": r["size_bytes"],
            "size_gb": r["size_gb"],
            "active_on": active.get(r["repo_id"]),
            "fit": fit,
        })
    return jsonify(out)


@app.post("/api/switch")
def switch_model():
    body = request.get_json(force=True) or {}
    repo_id = body.get("repo_id", "").strip()
    if not repo_id:
        return jsonify({"error": "repo_id is required"}), 400
    if repo_size_bytes(repo_id) == 0:
        return jsonify({"error": f"{repo_id} not cached; download it first"}), 404

    cfg = load_config()
    cfg["repo_id"] = repo_id
    if body.get("served_model_name"):
        cfg["served_model_name"] = body["served_model_name"]
    for key in ("max_model_len", "max_num_batched_tokens", "max_num_seqs",
                "gpu_memory_utilization", "kv_cache_dtype", "cpu_offload_gb",
                "enable_prefix_caching", "enable_auto_tool_choice",
                "reasoning_parser", "tool_call_parser"):
        if key in body:
            cfg[key] = body[key]

    rec = recommend_deployment(repo_id, cfg=cfg, assume_switch=True)
    if rec["strategy"] == "does_not_fit":
        return jsonify({"error": rec["reasoning"], "recommendation": rec}), 409

    # Caller can override
    if "gpus" in body and body["gpus"]:
        gpus = [int(x) for x in body["gpus"]]
        tp = body.get("tensor_parallel_size") or len(gpus)
    else:
        gpus = rec["recommended_gpus"]
        tp = rec["tensor_parallel_size"]

    cfg["tensor_parallel_size"] = tp
    restart_primary(cfg, gpus=gpus, tp=tp)
    return jsonify({"message": f"Switching to {repo_id} on GPU(s) {gpus} (tp={tp})",
                    "config": cfg, "gpus": gpus, "tensor_parallel_size": tp,
                    "recommendation": rec})


@app.post("/api/server/restart")
def server_restart():
    body = request.get_json(force=True) or {}
    cfg = load_config()
    for key in ("max_model_len", "max_num_batched_tokens", "max_num_seqs",
                "gpu_memory_utilization", "kv_cache_dtype", "cpu_offload_gb",
                "enable_prefix_caching", "enable_auto_tool_choice",
                "reasoning_parser", "tool_call_parser"):
        if key in body:
            cfg[key] = body[key]
    if not cfg.get("repo_id"):
        return jsonify({"error": "no model configured"}), 400
    gpus = body.get("gpus")
    if gpus:
        gpus = [int(x) for x in gpus]
    tp = body.get("tensor_parallel_size") or cfg.get("tensor_parallel_size", 1)
    cfg["tensor_parallel_size"] = tp
    restart_primary(cfg, gpus=gpus, tp=tp)
    return jsonify({"message": "Restarting...", "config": cfg})


@app.get("/api/server/logs")
def server_logs():
    n = int(request.args.get("lines", 80))
    port = int(request.args.get("port", PRIMARY_PORT))
    log = f"/var/log/vllm-server-{port}.log" if port != PRIMARY_PORT else "/var/log/vllm-server.log"
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
    repo_id = body.get("repo_id", "").strip()
    port = int(body.get("port", 8001))
    gpus = body.get("gpus")

    if not repo_id:
        return jsonify({"error": "repo_id required"}), 400
    if repo_size_bytes(repo_id) == 0:
        return jsonify({"error": f"{repo_id} not cached; download it first"}), 404
    with _servers_lock:
        if port in extra_servers or port == PRIMARY_PORT:
            return jsonify({"error": f"Port {port} already in use"}), 409

    cfg = load_config()
    cfg_extra = {**cfg, "repo_id": repo_id}
    for key in ("max_model_len", "max_num_batched_tokens", "max_num_seqs",
                "gpu_memory_utilization", "kv_cache_dtype"):
        if key in body:
            cfg_extra[key] = body[key]

    # Auto-select a free GPU if none specified
    if not gpus:
        fit = estimate_fit(repo_id, cfg=cfg_extra)
        with _servers_lock:
            used_gpus = set()
            for s in extra_servers.values():
                used_gpus.update(s.get("gpus") or [])
        for g in (primary_state.get("gpus") or []):
            used_gpus.add(int(g))
        chosen = None
        for ga in fit["gpus"]:
            if ga["fits"] and ga["id"] not in used_gpus:
                chosen = ga["id"]
                break
        if chosen is None:
            return jsonify({"error": "No free GPU with sufficient VRAM for this model"}), 409
        gpus = [chosen]
    else:
        gpus = [int(x) for x in gpus]

    cfg_extra["tensor_parallel_size"] = len(gpus)

    with _servers_lock:
        extra_servers[port] = {
            "port": port, "gpus": gpus, "repo_id": repo_id,
            "pid": None, "proc": None,
            "status": "loading", "message": f"Loading {repo_id}...",
            "config": cfg_extra, "started_at": None,
        }

    threading.Thread(target=_do_start_extra, args=(port, repo_id, cfg_extra, gpus),
                     daemon=True).start()
    return jsonify({"message": f"Launching {repo_id} on GPU {gpus} port {port}",
                    "port": port, "gpus": gpus}), 202


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

@app.get("/api/repo/info")
def repo_info():
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
                 for f in (info.siblings or [])]
        total = sum((f.size or 0) for f in (info.siblings or []))
        return jsonify({"repo_id": repo_id, "files": files,
                        "total_size_bytes": total,
                        "total_size_gb": round(total / 1e9, 2),
                        "cached": repo_size_bytes(repo_id) > 0})
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.post("/api/download")
def start_download():
    body = request.get_json(force=True) or {}
    repo_id = body.get("repo_id", "").strip()
    token = body.get("token", "").strip() or None
    if not repo_id:
        return jsonify({"error": "repo_id is required"}), 400
    if repo_size_bytes(repo_id) > 0:
        return jsonify({"error": f"{repo_id} already cached",
                        "size_gb": round(repo_size_bytes(repo_id) / 1e9, 2)}), 409
    dl_id = str(uuid.uuid4())[:8]
    with _dl_lock:
        downloads[dl_id] = {"id": dl_id, "status": "downloading",
                             "repo_id": repo_id, "downloaded": 0,
                             "total": None, "error": None, "pid": None}
    threading.Thread(target=_run_download, args=(dl_id, repo_id, token), daemon=True).start()
    return jsonify({"id": dl_id, "message": f"Download started: {repo_id}"}), 202


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


@app.delete("/api/models/<path:repo_id>")
def delete_model(repo_id):
    active = _active_repos()
    if repo_id in active:
        return jsonify({"error": "cannot delete an active model"}), 409
    d = repo_to_cache_dir(repo_id)
    if not d.exists():
        return jsonify({"error": "not found"}), 404
    shutil.rmtree(d)
    return jsonify({"message": f"{repo_id} deleted"})


# ── UI ────────────────────────────────────────────────────────────────────────

UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>vLLM Manager</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:monospace;background:#111;color:#ccc;font-size:13px;line-height:1.6}
#app{max-width:1100px;margin:0 auto;padding:16px}
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
.bar{height:18px;background:#0a0a0a;border:1px solid #222;border-radius:3px;position:relative;overflow:hidden;margin-top:4px}
.bar > div{height:100%;background:linear-gradient(90deg,#3a6,#5d8);transition:width .4s}
.bar > span{position:absolute;top:0;left:6px;font-size:11px;color:#fff;line-height:18px;text-shadow:0 0 3px #000}
button{font-family:inherit;font-size:12px;background:#222;color:#ccc;border:1px solid #333;padding:5px 10px;border-radius:3px;cursor:pointer}
button:hover{background:#2a2a2a;color:#fff}
button:disabled{opacity:0.4;cursor:not-allowed}
button.primary{background:#2c5a3a;border-color:#3a7a4a;color:#fff}
button.primary:hover{background:#3a7a4a}
button.danger{background:#5a2c2c;border-color:#7a3a3a}
button.danger:hover{background:#7a3a3a;color:#fff}
input,select{font-family:inherit;font-size:12px;background:#0a0a0a;color:#ccc;border:1px solid #222;padding:4px 8px;border-radius:3px}
input:focus,select:focus{outline:none;border-color:#3a6}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:6px 8px;border-bottom:1px solid #222;color:#555;font-weight:normal;text-transform:uppercase;font-size:10px;letter-spacing:1px}
td{padding:6px 8px;border-bottom:1px solid #1a1a1a;vertical-align:middle}
tr.active td{background:#152015}
.status-running{color:#5d8}
.status-loading{color:#fc6}
.status-stopping,.status-clearing_vram{color:#fc6}
.status-error{color:#f66}
.status-unknown{color:#666}
.rec-ok{border-left:3px solid #5d8;padding-left:10px}
.rec-warn{border-left:3px solid #fc6;padding-left:10px}
.rec-bad{border-left:3px solid #f66;padding-left:10px}
.tip{padding:5px 8px;margin:3px 0;font-size:11px;border-radius:2px}
.tip-ok{background:#152015;color:#9c9}
.tip-warn{background:#2a2310;color:#fc9}
.tip-info{background:#152030;color:#9bd}
details{margin-top:6px}
summary{cursor:pointer;color:#666;font-size:11px}
summary:hover{color:#aaa}
.muted{color:#555;font-size:11px}
.assigned{color:#9c9;font-size:11px;margin-left:8px}
.pill{display:inline-block;background:#0a1a0a;border:1px solid #2a4a2a;padding:1px 6px;border-radius:8px;font-size:10px;color:#9c9}
.pill.warn{background:#1a1505;border-color:#4a3a1a;color:#fc9}
.gpu-tag{display:inline-block;background:#152030;border:1px solid #2a4060;padding:0 5px;border-radius:8px;font-size:10px;color:#9bd;margin-right:3px}
form.inline{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.field{display:flex;flex-direction:column;gap:2px}
.field label{font-size:10px;color:#555;text-transform:uppercase}
.actions{display:flex;gap:6px;align-items:center;justify-content:flex-end}
.config-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.config-grid .field{margin-bottom:0}
</style>
</head>
<body>
<div id="app">
  <header>
    <h1>vLLM MANAGER</h1>
    <span id="hdr-state" class="status-unknown">—</span>
    <span id="refresh-ts"></span>
  </header>

  <section>
    <h2>Primary server (port 8000)</h2>
    <div class="card" id="primary-card">loading...</div>
  </section>

  <section>
    <h2>GPUs</h2>
    <div class="card" id="gpu-card">loading...</div>
  </section>

  <section>
    <h2>Cached models</h2>
    <div class="card" id="models-card">loading...</div>
  </section>

  <section id="switch-section" style="display:none">
    <h2>Switch primary <button onclick="closeSwitch()" style="float:right">×</button></h2>
    <div class="card" id="switch-card"></div>
  </section>

  <section>
    <h2>Running servers</h2>
    <div class="card" id="servers-card">loading...</div>
  </section>

  <section>
    <h2>Download model</h2>
    <div class="card">
      <form class="inline" onsubmit="startDownload(event)">
        <div class="field" style="flex:1;min-width:280px">
          <label>HF repo (e.g. cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit)</label>
          <input id="dl-repo" type="text" required style="width:100%">
        </div>
        <button class="primary" type="submit">Start download</button>
        <button type="button" onclick="checkRepo()">Inspect repo</button>
      </form>
      <div id="dl-info" class="muted" style="margin-top:8px"></div>
      <div id="downloads-list" style="margin-top:10px"></div>
    </div>
  </section>

  <section>
    <h2>HF token</h2>
    <div class="card">
      <form class="inline" onsubmit="saveToken(event)">
        <input id="tok" type="password" placeholder="hf_..." style="flex:1;min-width:240px">
        <button class="primary" type="submit">Save</button>
        <button type="button" onclick="clearToken()">Clear</button>
        <span id="tok-status" class="muted"></span>
      </form>
    </div>
  </section>

  <section>
    <h2>vLLM server logs <button onclick="loadLogs()" style="float:right">Reload</button></h2>
    <div class="card"><pre id="logs" style="font-size:11px;color:#888;max-height:240px;overflow:auto;white-space:pre-wrap">click reload</pre></div>
  </section>
</div>

<script>
let fastPoll = false;
let pendingSwitchRepo = null;

function fmtGB(b){ return (b/1e9).toFixed(2)+' GB'; }
function fmtMiB(m){ return m.toLocaleString()+' MiB'; }
function el(html){ const t=document.createElement('template'); t.innerHTML=html.trim(); return t.content.firstChild; }

async function api(path, opts={}){
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json()).error || r.statusText);
  return r.json();
}

function renderPrimary(state, cfg, health){
  const s = state.status || 'unknown';
  const statusEl = document.getElementById('hdr-state');
  statusEl.className = 'status-'+s;
  statusEl.textContent = s.toUpperCase();
  const gpus = state.gpus ? state.gpus.join(',') : '—';
  const tp = state.tp || cfg.tensor_parallel_size || 1;
  document.getElementById('primary-card').innerHTML = `
    <div class="row">
      <div class="kv"><label>Repo</label><span>${state.repo_id || '—'}</span></div>
      <div class="kv"><label>Served name</label><span>${cfg.served_model_name || '—'}</span></div>
      <div class="kv"><label>GPU(s)</label><span>${gpus} (tp=${tp})</span></div>
      <div class="kv"><label>Context</label><span>${(cfg.max_model_len||0).toLocaleString()}</span></div>
      <div class="kv"><label>Max seqs</label><span>${cfg.max_num_seqs}</span></div>
      <div class="kv"><label>KV dtype</label><span>${cfg.kv_cache_dtype}</span></div>
      <div class="kv"><label>GPU util</label><span>${cfg.gpu_memory_utilization}</span></div>
      <div class="kv"><label>Health</label><span class="status-${health.status==='ok'?'running':'error'}">${health.status}</span></div>
    </div>
    ${state.message ? `<div class="muted" style="margin-top:8px">${state.message}</div>` : ''}
    <div style="margin-top:10px"><button onclick="restart()">Restart</button></div>
  `;
}

function renderGPUs(vram){
  if (!vram || !vram.gpus){ document.getElementById('gpu-card').innerHTML = '<span class="muted">no GPUs</span>'; return; }
  document.getElementById('gpu-card').innerHTML = vram.gpus.map(g => {
    const pct = (g.used_mib / g.total_mib * 100).toFixed(0);
    const assigned = g.assigned_model
      ? `<span class="assigned">← ${g.assigned_model} (port ${g.assigned_port})</span>`
      : `<span class="assigned" style="color:#555">← idle</span>`;
    return `
      <div style="margin-bottom:8px">
        <div><strong>GPU ${g.id}</strong> ${g.name} · ${fmtMiB(g.total_mib)} ${assigned}</div>
        <div class="bar"><div style="width:${pct}%"></div><span>${fmtMiB(g.used_mib)} used / ${fmtMiB(g.free_mib)} free</span></div>
      </div>`;
  }).join('');
}

async function renderModels(){
  let models;
  try { models = await api('/api/models'); }
  catch(e){ document.getElementById('models-card').innerHTML = '<span class="muted">'+e.message+'</span>'; return; }

  if (!models.length){
    document.getElementById('models-card').innerHTML = '<span class="muted">no models cached — use the download form below</span>';
    return;
  }
  const rows = models.map(m => {
    const fit = m.fit;
    const fitBadge = fit.fits_single_gpu
      ? `<span class="pill">fits 1 GPU</span>`
      : (fit.recommended_tp ? `<span class="pill">tp=${fit.recommended_tp}</span>` : `<span class="pill warn">no fit</span>`);
    const activeCls = m.active_on ? 'active' : '';
    const activeBadge = m.active_on ? `<span class="pill">port ${m.active_on}</span>` : '';
    return `<tr class="${activeCls}">
      <td>${m.repo_id} ${activeBadge}</td>
      <td>${m.size_gb} GB</td>
      <td>${fit.quant} · ${fit.arch}</td>
      <td>${fitBadge}</td>
      <td class="actions">
        <button onclick="openSwitch('${m.repo_id}')">Switch primary</button>
        <button onclick="launchExtra('${m.repo_id}')">+ extra</button>
        <button class="danger" onclick="delModel('${m.repo_id}')">Delete</button>
      </td>
    </tr>`;
  }).join('');
  document.getElementById('models-card').innerHTML = `
    <table>
      <thead><tr><th>Repo</th><th>Size</th><th>Type</th><th>Fit</th><th></th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function openSwitch(repo){
  pendingSwitchRepo = repo;
  document.getElementById('switch-section').style.display = 'block';
  document.getElementById('switch-card').innerHTML = 'computing recommendation...';
  const rec = await api('/api/recommend?repo_id='+encodeURIComponent(repo)+'&assume_switch=1');
  const cls = rec.strategy === 'does_not_fit' ? 'rec-bad' : (rec.strategy === 'tensor_parallel' ? 'rec-warn' : 'rec-ok');
  const gpusStr = (rec.recommended_gpus || []).join(',');
  const tips = (rec.tips || []).map(t => `<div class="tip tip-${t.level}">${t.text}</div>`).join('');
  const warns = (rec.warnings || []).map(w => `<div class="muted">⚠ ${w}</div>`).join('');
  const cfg = await api('/api/status').then(s=>s.config);

  document.getElementById('switch-card').innerHTML = `
    <div class="${cls}" style="margin-bottom:14px">
      <div><strong>${rec.strategy.replace('_',' ')}</strong> — ${rec.reasoning}</div>
      <div class="muted" style="margin-top:4px">Weights ${rec.fit.weights_gb} GB · overhead ~${(rec.fit.overhead_mib/1024).toFixed(1)} GB · headroom ${rec.headroom_mib>0?'+':''}${rec.headroom_mib.toLocaleString()} MiB</div>
      ${warns}
    </div>
    <form class="inline" onsubmit="doSwitch(event)">
      <div class="field"><label>Repo</label><input value="${repo}" readonly style="width:300px;color:#888"></div>
      <div class="field"><label>GPUs (csv)</label><input id="sw-gpus" value="${gpusStr}" style="width:80px"></div>
      <div class="field"><label>tp size</label><input id="sw-tp" type="number" value="${rec.tensor_parallel_size||1}" min="1" max="8" style="width:60px"></div>
      <div class="field"><label>max_model_len</label><input id="sw-ctx" type="number" value="${cfg.max_model_len}" style="width:90px"></div>
      <div class="field"><label>max_num_seqs</label><input id="sw-seqs" type="number" value="${cfg.max_num_seqs}" style="width:60px"></div>
      <div class="field"><label>gpu_util</label><input id="sw-util" type="number" step="0.01" value="${cfg.gpu_memory_utilization}" style="width:60px"></div>
      <div class="field"><label>kv dtype</label>
        <select id="sw-kv"><option value="fp8"${cfg.kv_cache_dtype==='fp8'?' selected':''}>fp8</option><option value="auto"${cfg.kv_cache_dtype==='auto'?' selected':''}>auto/fp16</option></select>
      </div>
      <button class="primary" type="submit" ${rec.strategy==='does_not_fit'?'disabled':''}>Switch</button>
    </form>
    <details style="margin-top:10px">
      <summary>Performance tips (${(rec.tips||[]).length})</summary>
      <div style="margin-top:6px">${tips}</div>
    </details>
  `;
}

function closeSwitch(){
  document.getElementById('switch-section').style.display = 'none';
  pendingSwitchRepo = null;
}

async function doSwitch(e){
  e.preventDefault();
  const gpusRaw = document.getElementById('sw-gpus').value.trim();
  const gpus = gpusRaw ? gpusRaw.split(',').map(x=>parseInt(x.trim())) : [];
  const body = {
    repo_id: pendingSwitchRepo,
    gpus: gpus,
    tensor_parallel_size: parseInt(document.getElementById('sw-tp').value),
    max_model_len: parseInt(document.getElementById('sw-ctx').value),
    max_num_seqs: parseInt(document.getElementById('sw-seqs').value),
    gpu_memory_utilization: parseFloat(document.getElementById('sw-util').value),
    kv_cache_dtype: document.getElementById('sw-kv').value,
  };
  try {
    await api('/api/switch', {method:'POST', headers:{'Content-Type':'application/json'},
                              body: JSON.stringify(body)});
    fastPoll = true;
    closeSwitch();
    refresh();
  } catch(e){ alert(e.message); }
}

async function launchExtra(repo){
  const portRaw = prompt('Port for extra server', '8001');
  if (!portRaw) return;
  const port = parseInt(portRaw);
  try {
    await api('/api/servers', {method:'POST', headers:{'Content-Type':'application/json'},
                                body: JSON.stringify({repo_id: repo, port: port})});
    fastPoll = true;
    refresh();
  } catch(e){ alert(e.message); }
}

async function stopExtra(port){
  if (!confirm('Stop server on port '+port+'?')) return;
  await api('/api/servers/'+port, {method:'DELETE'});
  refresh();
}

async function delModel(repo){
  if (!confirm('Delete '+repo+' from cache?')) return;
  try {
    await api('/api/models/'+encodeURIComponent(repo), {method:'DELETE'});
    refresh();
  } catch(e){ alert(e.message); }
}

async function restart(){
  if (!confirm('Restart primary vLLM server?')) return;
  await api('/api/server/restart', {method:'POST', headers:{'Content-Type':'application/json'},
                                     body: '{}'});
  fastPoll = true;
  refresh();
}

async function renderServers(extras){
  const items = Object.entries(extras || {});
  if (!items.length){
    document.getElementById('servers-card').innerHTML = '<span class="muted">no extra servers running</span>';
    return;
  }
  document.getElementById('servers-card').innerHTML = items.map(([port,s]) => `
    <div style="display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid #1a1a1a">
      <span class="status-${s.status}">●</span>
      <strong>port ${port}</strong>
      <span>${s.repo_id || '—'}</span>
      <span class="gpu-tag">GPU ${(s.gpus||[]).join(',')}</span>
      <span class="muted">${s.message || s.status}</span>
      <button class="danger" style="margin-left:auto" onclick="stopExtra(${port})">Stop</button>
    </div>
  `).join('');
}

async function checkRepo(){
  const repo = document.getElementById('dl-repo').value.trim();
  if (!repo) return;
  document.getElementById('dl-info').textContent = 'fetching repo info...';
  try {
    const info = await api('/api/repo/info?repo_id='+encodeURIComponent(repo));
    document.getElementById('dl-info').innerHTML =
      `${info.total_size_gb} GB total · ${info.files.length} files · ${info.cached ? '<span style="color:#5d8">already cached</span>' : '<span style="color:#fc6">not cached</span>'}`;
  } catch(e){ document.getElementById('dl-info').textContent = e.message; }
}

async function startDownload(e){
  e.preventDefault();
  const repo = document.getElementById('dl-repo').value.trim();
  try {
    await api('/api/download', {method:'POST', headers:{'Content-Type':'application/json'},
                                 body: JSON.stringify({repo_id: repo})});
    document.getElementById('dl-repo').value = '';
    refresh();
  } catch(e){ alert(e.message); }
}

async function renderDownloads(){
  let dls = [];
  try { dls = await api('/api/downloads'); } catch(e){}
  if (!dls.length){ document.getElementById('downloads-list').innerHTML=''; return; }
  document.getElementById('downloads-list').innerHTML = dls.map(d => {
    const pct = d.progress_pct;
    const bar = pct != null
      ? `<div class="bar" style="width:200px"><div style="width:${pct}%"></div><span>${pct}%</span></div>`
      : `<span class="muted">${(d.downloaded/1e9).toFixed(2)} GB</span>`;
    return `<div style="display:flex;align-items:center;gap:10px;padding:4px 0">
      <span class="status-${d.status==='done'?'running':d.status==='error'?'error':'loading'}">●</span>
      <span>${d.repo_id}</span>
      ${bar}
      <span class="muted">${d.status}${d.error?': '+d.error:''}</span>
    </div>`;
  }).join('');
}

async function saveToken(e){
  e.preventDefault();
  const tok = document.getElementById('tok').value.trim();
  if (!tok) return;
  await api('/api/token', {method:'POST', headers:{'Content-Type':'application/json'},
                            body: JSON.stringify({token: tok})});
  document.getElementById('tok').value = '';
  document.getElementById('tok-status').textContent = 'saved';
  setTimeout(()=>document.getElementById('tok-status').textContent='', 2000);
}

async function clearToken(){
  await api('/api/token', {method:'DELETE'});
  document.getElementById('tok-status').textContent = 'cleared';
  setTimeout(()=>document.getElementById('tok-status').textContent='', 2000);
}

async function loadLogs(){
  try {
    const r = await api('/api/server/logs?lines=120');
    document.getElementById('logs').textContent = r.lines.join('\n');
  } catch(e){ document.getElementById('logs').textContent = e.message; }
}

async function refresh(){
  try {
    const s = await api('/api/status');
    renderPrimary(s.server_state, s.config, s.health);
    renderGPUs(s.vram);
    renderServers(s.extras);
    fastPoll = ['loading','stopping','clearing_vram'].includes(s.server_state.status);
    document.getElementById('refresh-ts').textContent = new Date().toLocaleTimeString();
  } catch(e){
    document.getElementById('hdr-state').textContent = 'ERR: '+e.message;
  }
  await renderModels();
  await renderDownloads();
  try {
    const t = await api('/api/token');
    document.getElementById('tok-status').textContent = t.token_set ? 'token set' : '';
  } catch(e){}
}

function scheduleRefresh(){
  setTimeout(async ()=>{ await refresh(); scheduleRefresh(); }, fastPoll ? 2000 : 5000);
}

refresh().then(scheduleRefresh);
</script>
</body>
</html>
"""


@app.get("/")
def ui():
    return render_template_string(UI_HTML)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
