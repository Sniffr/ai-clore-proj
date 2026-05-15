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

UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LLM Manager</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:monospace;background:#111;color:#ccc;font-size:13px;line-height:1.5}
#app{max-width:960px;margin:0 auto;padding:16px}
header{display:flex;align-items:center;gap:12px;padding:10px 0 18px;border-bottom:1px solid #2a2a2a;margin-bottom:20px}
header h1{font-size:15px;color:#fff;font-weight:normal;letter-spacing:1px}
#health-dot{width:9px;height:9px;border-radius:50%;background:#444;flex-shrink:0}
#health-dot.ok{background:#4caf50}
#health-dot.err{background:#f44336}
#refresh-ts{margin-left:auto;font-size:11px;color:#555}
section{margin-bottom:28px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:1px;color:#666;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #1e1e1e}
.card{background:#181818;border:1px solid #252525;border-radius:4px;padding:14px}
.row{display:flex;gap:24px;flex-wrap:wrap}
.kv{display:flex;flex-direction:column;gap:2px}
.kv span:first-child{font-size:11px;color:#555}
.kv span:last-child{color:#e0e0e0}
.vram-bar{height:6px;background:#222;border-radius:3px;margin-top:6px;overflow:hidden}
.vram-bar-fill{height:100%;background:#1976d2;border-radius:3px;transition:width .4s}
table{width:100%;border-collapse:collapse}
th{text-align:left;font-size:11px;color:#555;text-transform:uppercase;padding:6px 8px;border-bottom:1px solid #1e1e1e}
td{padding:7px 8px;border-bottom:1px solid #1a1a1a;vertical-align:middle}
tr:last-child td{border-bottom:none}
tr:hover td{background:#1a1a1a}
.active-badge{color:#4caf50;font-size:11px}
.tag{font-size:10px;padding:1px 5px;border-radius:2px;background:#1e1e1e;color:#888;border:1px solid #2a2a2a}
btn,button,.btn{font-family:monospace;font-size:12px;padding:4px 10px;border:1px solid #333;border-radius:3px;background:#1e1e1e;color:#ccc;cursor:pointer;white-space:nowrap}
button:hover,.btn:hover{background:#252525;border-color:#444}
button.primary,.btn.primary{border-color:#1565c0;color:#90caf9;background:#0d1a2e}
button.primary:hover{background:#112240}
button.danger{border-color:#5c1a1a;color:#ef9a9a;background:#1a0d0d}
button.danger:hover{background:#220f0f}
button:disabled{opacity:.4;cursor:not-allowed}
input,select{font-family:monospace;font-size:12px;padding:5px 8px;background:#151515;border:1px solid #2a2a2a;border-radius:3px;color:#ddd;width:100%}
input:focus,select:focus{outline:none;border-color:#1565c0}
.form-row{display:flex;gap:8px;align-items:flex-end;flex-wrap:wrap}
.form-group{display:flex;flex-direction:column;gap:4px}
.form-group label{font-size:11px;color:#555}
.form-group.grow{flex:1;min-width:140px}
.msg{font-size:11px;padding:4px 8px;border-radius:3px;margin-top:6px}
.msg.ok{color:#81c784;background:#0a1f0a;border:1px solid #1b3d1b}
.msg.err{color:#e57373;background:#1f0a0a;border:1px solid #3d1b1b}
.msg.info{color:#90caf9;background:#0a1020;border:1px solid #1b2d4a}
.prog-bar{height:4px;background:#1e1e1e;border-radius:2px;overflow:hidden;margin-top:4px}
.prog-bar-fill{height:100%;background:#1976d2;border-radius:2px;transition:width .3s}
pre#log-out{background:#0d0d0d;border:1px solid #1e1e1e;border-radius:4px;padding:12px;font-size:11px;color:#8bc34a;overflow-x:auto;white-space:pre-wrap;word-break:break-all;max-height:340px;overflow-y:auto}
.cfg-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px}
.sep{width:1px;background:#222;align-self:stretch;margin:0 4px}
</style>
</head>
<body>
<div id="app">

<header>
  <div id="health-dot"></div>
  <h1>LLM MANAGER</h1>
  <span id="refresh-ts"></span>
</header>

<!-- STATUS -->
<section>
  <h2>Status</h2>
  <div class="card" id="status-card">Loading...</div>
</section>

<!-- MODELS -->
<section>
  <h2>Models on disk</h2>
  <div id="models-msg"></div>
  <table>
    <thead><tr><th>Filename</th><th>Size</th><th>Active</th><th></th></tr></thead>
    <tbody id="models-body"><tr><td colspan="4" style="color:#555">Loading...</td></tr></tbody>
  </table>
</section>

<!-- SWITCH CONFIG -->
<section id="switch-section" style="display:none">
  <h2>Switch model — server config</h2>
  <div class="card">
    <div class="cfg-grid" id="switch-cfg-grid"></div>
    <div style="margin-top:12px;display:flex;gap:8px;align-items:center">
      <button class="primary" onclick="doSwitch()">Switch &amp; restart</button>
      <button onclick="document.getElementById('switch-section').style.display='none'">Cancel</button>
      <span id="switch-msg" class="msg" style="display:none"></span>
    </div>
  </div>
</section>

<!-- DOWNLOAD -->
<section>
  <h2>Download model</h2>
  <div class="card">
    <div class="form-row">
      <div class="form-group grow">
        <label>HuggingFace repo ID</label>
        <input id="dl-repo" type="text" placeholder="unsloth/Qwen3.6-35B-A3B-GGUF">
      </div>
      <div class="form-group grow">
        <label>Filename</label>
        <input id="dl-file" type="text" placeholder="Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf">
      </div>
      <div class="form-group" style="min-width:200px">
        <label>HF Token (optional — overrides saved)</label>
        <input id="dl-token" type="password" placeholder="hf_...">
      </div>
      <div class="form-group">
        <label>&nbsp;</label>
        <button class="primary" onclick="startDownload()">Download</button>
      </div>
    </div>
    <div id="dl-msg" class="msg" style="display:none"></div>
  </div>
</section>

<!-- ACTIVE DOWNLOADS -->
<section>
  <h2>Downloads</h2>
  <div id="downloads-empty" style="color:#555;font-size:12px">No downloads yet.</div>
  <table id="downloads-table" style="display:none">
    <thead><tr><th>File</th><th>Repo</th><th>Progress</th><th>Status</th><th></th></tr></thead>
    <tbody id="downloads-body"></tbody>
  </table>
</section>

<!-- SERVER CONFIG -->
<section>
  <h2>Server config &amp; restart</h2>
  <div class="card">
    <div class="cfg-grid" id="cfg-grid"></div>
    <div style="margin-top:12px;display:flex;gap:8px;align-items:center">
      <button class="primary" onclick="restartServer()">Apply &amp; restart</button>
      <span id="cfg-msg" class="msg" style="display:none"></span>
    </div>
  </div>
</section>

<!-- TOKEN -->
<section>
  <h2>HuggingFace token</h2>
  <div class="card">
    <div class="form-row">
      <div class="form-group grow">
        <label id="token-label">Status: checking...</label>
        <input id="token-input" type="password" placeholder="hf_...">
      </div>
      <div class="form-group">
        <label>&nbsp;</label>
        <button class="primary" onclick="saveToken()">Save token</button>
      </div>
      <div class="form-group">
        <label>&nbsp;</label>
        <button class="danger" onclick="deleteToken()">Remove</button>
      </div>
    </div>
    <div id="token-msg" class="msg" style="display:none"></div>
  </div>
</section>

<!-- LOGS -->
<section>
  <h2>Server log <span style="color:#555;font-size:11px">(last <select id="log-lines" onchange="fetchLogs()" style="width:60px;padding:2px 4px"><option>30</option><option>60</option><option selected>100</option><option>200</option></select> lines)</span></h2>
  <pre id="log-out">Loading...</pre>
</section>

</div>

<script>
const CFG_FIELDS = [
  {key:'context',  label:'Context',        type:'number', step:1024},
  {key:'temp',     label:'Temperature',    type:'number', step:0.05},
  {key:'top_k',    label:'Top-K',          type:'number', step:1},
  {key:'top_p',    label:'Top-P',          type:'number', step:0.05},
  {key:'min_p',    label:'Min-P',          type:'number', step:0.01},
  {key:'repeat_penalty', label:'Repeat penalty', type:'number', step:0.05},
  {key:'np',       label:'Parallel slots', type:'number', step:1},
  {key:'batch_size',label:'Batch size',    type:'number', step:512},
];

let currentCfg = {};
let switchTarget = null;

// ── helpers ──────────────────────────────────────────────────────────────────
function msg(id, text, type='info') {
  const el = document.getElementById(id);
  el.textContent = text; el.className = 'msg ' + type; el.style.display = '';
  if (type === 'ok') setTimeout(() => el.style.display='none', 4000);
}
function fmtSize(b) {
  if (!b) return '—';
  return b > 1e9 ? (b/1e9).toFixed(1)+' GB' : (b/1e6).toFixed(0)+' MB';
}
function fmtPct(d, t) {
  return t ? Math.min(100, Math.round(d/t*100)) : 0;
}

// ── status ───────────────────────────────────────────────────────────────────
async function fetchStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    currentCfg = d.config || {};
    const v = d.vram || {};
    const ok = d.health?.status === 'ok';
    const dot = document.getElementById('health-dot');
    dot.className = ok ? 'ok' : 'err';
    const pct = v.total ? Math.round(v.used_mib/v.total_mib*100) : 0;
    const model = d.active_model ? d.active_model.split('/').pop() : '—';
    document.getElementById('status-card').innerHTML = `
      <div class="row">
        <div class="kv"><span>Active model</span><span>${model}</span></div>
        <div class="kv"><span>Server</span><span class="${ok?'active-badge':''}">
          ${ok ? '● healthy' : '○ unreachable'}</span></div>
        <div class="kv"><span>Context</span><span>${(currentCfg.context||0).toLocaleString()} tokens</span></div>
        <div class="kv"><span>Temperature</span><span>${currentCfg.temp ?? '—'}</span></div>
        <div class="kv" style="min-width:180px">
          <span>VRAM  ${v.used_mib||0} / ${v.total_mib||0} MiB (${pct}%)</span>
          <div class="vram-bar"><div class="vram-bar-fill" style="width:${pct}%"></div></div>
        </div>
      </div>`;
    buildCfgGrid('cfg-grid', currentCfg, false);
    document.getElementById('refresh-ts').textContent = 'updated '+new Date().toLocaleTimeString();
  } catch(e) {
    document.getElementById('health-dot').className = 'err';
  }
}

// ── config grid ──────────────────────────────────────────────────────────────
function buildCfgGrid(gridId, cfg, forSwitch) {
  const grid = document.getElementById(gridId);
  if (!grid) return;
  grid.innerHTML = CFG_FIELDS.map(f => `
    <div class="form-group">
      <label>${f.label}</label>
      <input type="${f.type}" step="${f.step}"
             id="${forSwitch?'sw-':'cfg-'}${f.key}"
             value="${cfg[f.key] ?? ''}">
    </div>`).join('');
}
function readCfgGrid(prefix) {
  const out = {};
  CFG_FIELDS.forEach(f => {
    const el = document.getElementById(prefix+f.key);
    if (el && el.value !== '') out[f.key] = f.type==='number' ? Number(el.value) : el.value;
  });
  return out;
}

// ── models ───────────────────────────────────────────────────────────────────
async function fetchModels() {
  try {
    const r = await fetch('/api/models');
    const models = await r.json();
    const tbody = document.getElementById('models-body');
    if (!models.length) {
      tbody.innerHTML = '<tr><td colspan="4" style="color:#555">No models in /models</td></tr>';
      return;
    }
    tbody.innerHTML = models.map(m => `
      <tr>
        <td>${m.filename}</td>
        <td style="color:#888">${m.size_gb} GB</td>
        <td>${m.active ? '<span class="active-badge">● active</span>' : ''}</td>
        <td style="text-align:right">
          ${!m.active ? `<button onclick="openSwitch('${m.filename}')">Switch to this</button> ` : ''}
          ${!m.active ? `<button class="danger" onclick="deleteModel('${m.filename}')">Delete</button>` : ''}
        </td>
      </tr>`).join('');
  } catch(e) {}
}

function openSwitch(filename) {
  switchTarget = filename;
  document.getElementById('switch-section').style.display = '';
  document.getElementById('switch-section').scrollIntoView({behavior:'smooth',block:'center'});
  buildCfgGrid('switch-cfg-grid', currentCfg, true);
  document.getElementById('switch-msg').style.display = 'none';
}

async function doSwitch() {
  if (!switchTarget) return;
  const body = {filename: switchTarget, ...readCfgGrid('sw-')};
  const btn = event.target; btn.disabled = true;
  try {
    const r = await fetch('/api/switch', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d = await r.json();
    if (r.ok) {
      msg('switch-msg', 'Switching to '+switchTarget+' — server restarting…', 'ok');
      document.getElementById('switch-section').style.display = 'none';
      switchTarget = null;
    } else {
      msg('switch-msg', d.error || 'Error', 'err');
    }
  } catch(e) { msg('switch-msg', e.message, 'err'); }
  btn.disabled = false;
}

async function deleteModel(filename) {
  if (!confirm('Delete '+filename+'?')) return;
  const r = await fetch('/api/models/'+encodeURIComponent(filename), {method:'DELETE'});
  const d = await r.json();
  msg('models-msg', d.message || d.error, r.ok ? 'ok' : 'err');
  fetchModels();
}

// ── download ─────────────────────────────────────────────────────────────────
async function startDownload() {
  const repo = document.getElementById('dl-repo').value.trim();
  const file = document.getElementById('dl-file').value.trim();
  const token = document.getElementById('dl-token').value.trim();
  if (!repo || !file) { msg('dl-msg','repo ID and filename are required','err'); return; }
  const body = {repo_id:repo, filename:file};
  if (token) body.token = token;
  const btn = event.target; btn.disabled = true;
  try {
    const r = await fetch('/api/download', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d = await r.json();
    if (r.ok) {
      msg('dl-msg','Download started: '+file+' (id: '+d.id+')','ok');
      document.getElementById('dl-file').value = '';
    } else {
      msg('dl-msg', d.error || 'Error', 'err');
    }
  } catch(e) { msg('dl-msg', e.message, 'err'); }
  btn.disabled = false;
}

async function cancelDownload(id) {
  await fetch('/api/downloads/'+id, {method:'DELETE'});
  fetchDownloads();
}

async function fetchDownloads() {
  try {
    const r = await fetch('/api/downloads');
    const list = await r.json();
    const empty = document.getElementById('downloads-empty');
    const tbl = document.getElementById('downloads-table');
    if (!list.length) { empty.style.display=''; tbl.style.display='none'; return; }
    empty.style.display='none'; tbl.style.display='';
    document.getElementById('downloads-body').innerHTML = list.map(d => {
      const pct = d.progress_pct ?? 0;
      const statusColor = d.status==='done'?'#4caf50':d.status==='error'?'#f44336':d.status==='cancelled'?'#888':'#90caf9';
      return `<tr>
        <td>${d.filename}</td>
        <td style="color:#666">${d.repo_id}</td>
        <td style="min-width:120px">
          ${fmtSize(d.downloaded)} / ${fmtSize(d.total)}
          <div class="prog-bar"><div class="prog-bar-fill" style="width:${pct}%"></div></div>
        </td>
        <td><span style="color:${statusColor}">${d.status} ${d.status==='downloading'?pct+'%':''}</span>
            ${d.error?`<br><span style="color:#888;font-size:10px">${d.error}</span>`:''}
        </td>
        <td>${d.status==='downloading'?`<button class="danger" onclick="cancelDownload('${d.id}')">Cancel</button>`:''}</td>
      </tr>`;
    }).join('');
  } catch(e) {}
}

// ── server restart ────────────────────────────────────────────────────────────
async function restartServer() {
  const body = readCfgGrid('cfg-');
  const btn = event.target; btn.disabled = true;
  try {
    const r = await fetch('/api/server/restart', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d = await r.json();
    msg('cfg-msg', r.ok ? 'Restarting server…' : (d.error||'Error'), r.ok?'ok':'err');
  } catch(e) { msg('cfg-msg', e.message, 'err'); }
  btn.disabled = false;
}

// ── token ─────────────────────────────────────────────────────────────────────
async function fetchToken() {
  const r = await fetch('/api/token');
  const d = await r.json();
  document.getElementById('token-label').textContent = d.token_set ? 'Saved token: ●●●●●●●● (set)' : 'No token saved';
}
async function saveToken() {
  const token = document.getElementById('token-input').value.trim();
  if (!token) { msg('token-msg','Enter a token','err'); return; }
  const r = await fetch('/api/token', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})});
  const d = await r.json();
  msg('token-msg', d.message||d.error, r.ok?'ok':'err');
  document.getElementById('token-input').value = '';
  fetchToken();
}
async function deleteToken() {
  const r = await fetch('/api/token', {method:'DELETE'});
  const d = await r.json();
  msg('token-msg', d.message||d.error, r.ok?'ok':'err');
  fetchToken();
}

// ── logs ──────────────────────────────────────────────────────────────────────
async function fetchLogs() {
  const n = document.getElementById('log-lines').value;
  try {
    const r = await fetch('/api/server/logs?lines='+n);
    const d = await r.json();
    const pre = document.getElementById('log-out');
    pre.textContent = d.lines?.join('\n') || '(empty)';
    pre.scrollTop = pre.scrollHeight;
  } catch(e) {}
}

// ── polling ───────────────────────────────────────────────────────────────────
async function refresh() {
  await Promise.all([fetchStatus(), fetchModels(), fetchDownloads(), fetchToken(), fetchLogs()]);
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


@app.get("/")
def ui():
    return render_template_string(UI_HTML)


@app.get("/api")
def index():
    cfg = load_config()
    return jsonify({
        "service": "llama model manager",
        "active_model": active_model(),
        "config": cfg,
        "endpoints": [
            "GET  /api/models",
            "GET  /api/status",
            "POST /api/download         {repo_id, filename, token?}",
            "GET  /api/downloads",
            "GET  /api/downloads/<id>",
            "DELETE /api/downloads/<id>",
            "POST /api/switch           {filename, context?, temp?, top_k?, top_p?, min_p?, repeat_penalty?, np?, batch_size?}",
            "POST /api/server/restart   {context?, temp?, top_k?, top_p?, min_p?, repeat_penalty?, np?, batch_size?}",
            "GET  /api/server/logs      ?lines=50",
            "POST /api/token            {token}",
            "GET  /api/token",
            "DELETE /api/token",
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

    for key in ("context", "temp", "top_k", "top_p", "min_p", "repeat_penalty", "np", "batch_size"):
        if key in body:
            cfg[key] = body[key]

    save_config(cfg)

    t = threading.Thread(target=restart_llama, args=(cfg,), daemon=True)
    t.start()

    return jsonify({"message": f"Switching to {filename}", "config": cfg})


@app.post("/api/server/restart")
def server_restart():
    """Restart llama-server with updated args, keeping the current model."""
    body = request.get_json(force=True) or {}
    cfg = load_config()

    for key in ("context", "temp", "top_k", "top_p", "min_p", "repeat_penalty", "np", "batch_size"):
        if key in body:
            cfg[key] = body[key]

    if not cfg.get("model"):
        return jsonify({"error": "no model configured — use /api/switch first"}), 400

    save_config(cfg)

    t = threading.Thread(target=restart_llama, args=(cfg,), daemon=True)
    t.start()

    return jsonify({"message": "Restarting llama-server with new args", "config": cfg})


@app.get("/api/server/logs")
def server_logs():
    """Tail the llama-server log."""
    n = int(request.args.get("lines", 50))
    try:
        result = subprocess.run(["tail", f"-{n}", "/var/log/llama-server.log"],
                                capture_output=True, text=True)
        return jsonify({"lines": result.stdout.splitlines()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
