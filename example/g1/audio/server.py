#!/usr/bin/env python3
"""
Temirbek web control panel.
Runs on the robot and serves a mobile-friendly UI on port 5000.
Access from phone: http://100.68.225.43:5000  (via Tailscale)
"""

import os
import sys
import signal
import subprocess
from pathlib import Path
from flask import Flask, jsonify, request

# ── path / env setup (mirrors g1_audio_alem.py) ──────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def _load_env():
    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            os.environ.setdefault(key.strip(), value.strip())

_load_env()

# ── config ────────────────────────────────────────────────────────────────────
NETWORK_INTERFACE = os.getenv("ROBOT_NETWORK_INTERFACE", "eth0")
SCRIPT = str(Path(__file__).parent / "g1_audio_alem.py")
STATE_FILE   = "/tmp/temirbek_state"
GESTURE_FILE = "/tmp/temirbek_gesture"

# ── process handle ────────────────────────────────────────────────────────────
_proc: subprocess.Popen = None

def _is_running() -> bool:
    return _proc is not None and _proc.poll() is None

def _read_state() -> str:
    if not _is_running():
        return "idle"
    try:
        return Path(STATE_FILE).read_text().strip() or "starting"
    except Exception:
        return "starting"

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return _HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

@app.route("/start", methods=["POST"])
def start():
    global _proc
    if _is_running():
        return jsonify(ok=False, error="already running")
    try:
        _proc = subprocess.Popen(
            [sys.executable, SCRIPT, NETWORK_INTERFACE],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(_REPO_ROOT),
        )
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, error=str(e))

@app.route("/stop", methods=["POST"])
def stop():
    global _proc
    if not _is_running():
        return jsonify(ok=False, error="not running")
    _proc.terminate()
    try:
        _proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _proc.kill()
    _proc = None
    return jsonify(ok=True)

def _read_gesture() -> str:
    try:
        return Path(GESTURE_FILE).read_text().strip()
    except Exception:
        return ""

@app.route("/status")
def status():
    return jsonify(running=_is_running(), state=_read_state(), gesture=_read_gesture())

# ── inline HTML (single-file, no templates) ───────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Temirbek</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f0f13;
    color: #e8e8f0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 36px;
    padding: 24px;
  }
  h1 { font-size: 2rem; font-weight: 700; letter-spacing: 0.04em; }
  #status-row {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.1rem;
  }
  #dot {
    width: 14px; height: 14px;
    border-radius: 50%;
    background: #555;
    transition: background 0.3s;
  }
  #dot.idle     { background: #555; }
  #dot.starting { background: #f0a500; animation: pulse 1s infinite; }
  #dot.listening{ background: #22c55e; animation: pulse 1s infinite; }
  #dot.thinking { background: #f0a500; animation: pulse 0.6s infinite; }
  #dot.speaking { background: #3b82f6; animation: pulse 0.8s infinite; }
  @keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
  }
  #btn {
    width: 200px; height: 200px;
    border-radius: 50%;
    border: none;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    cursor: pointer;
    transition: transform 0.12s, box-shadow 0.12s;
    color: #fff;
  }
  #btn:active { transform: scale(0.94); }
  #btn.start {
    background: #16a34a;
    box-shadow: 0 0 40px #16a34a88;
  }
  #btn.stop {
    background: #dc2626;
    box-shadow: 0 0 40px #dc262688;
  }
  #btn:disabled {
    background: #333;
    box-shadow: none;
    cursor: default;
  }
  #gesture-label {
    font-size: 0.95rem;
    color: #888;
    min-height: 1.2em;
    letter-spacing: 0.03em;
    font-style: italic;
  }
</style>
</head>
<body>
<h1>Temirbek</h1>
<div id="status-row">
  <div id="dot" class="idle"></div>
  <span id="status-label">Stopped</span>
</div>
<div id="gesture-label"></div>
<button id="btn" class="start" onclick="toggle()">START</button>

<script>
const STATE_LABELS = {
  idle: "Stopped",
  starting: "Starting…",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
};

let running = false;

async function toggle() {
  const btn = document.getElementById('btn');
  btn.disabled = true;
  const url = running ? '/stop' : '/start';
  await fetch(url, { method: 'POST' });
  btn.disabled = false;
  poll();
}

async function poll() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    running = d.running;
    const state = d.state || (running ? 'starting' : 'idle');
    const dot   = document.getElementById('dot');
    const label = document.getElementById('status-label');
    const btn   = document.getElementById('btn');
    dot.className = state;
    label.textContent = STATE_LABELS[state] || state;
    btn.textContent = running ? 'STOP' : 'START';
    btn.className   = running ? 'stop' : 'start';
    const gesture = document.getElementById('gesture-label');
    gesture.textContent = (d.gesture && state === 'speaking') ? d.gesture : '';
  } catch(e) {}
}

setInterval(poll, 1500);
poll();
</script>
</body>
</html>"""

# ── entry point ───────────────────────────────────────────────────────────────
def _shutdown(sig, frame):
    if _is_running():
        _proc.terminate()
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)

if __name__ == "__main__":
    print(f"Temirbek web panel → http://100.68.225.43:5000")
    print(f"Interface: {NETWORK_INTERFACE}  Script: {SCRIPT}")
    app.run(host="0.0.0.0", port=5000, debug=False)
