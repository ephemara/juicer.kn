#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import urllib.request

# Machine-local defaults. Override with $COMFYUI_HOME / $COMFYUI_APP / $JUICER_PY
# rather than editing these lines when the box moves.
COMFYUI_HOME = os.environ.get("COMFYUI_HOME", r"S:\Local\ComfyUI_windows_portable")
COMFY_ROOT = os.environ.get("COMFYUI_APP", os.path.join(COMFYUI_HOME, "ComfyUI"))
PY_BIN = os.environ.get("JUICER_PY", os.path.join(COMFYUI_HOME, "python_embeded", "python.exe"))
LOG_FILE = os.environ.get(
    "JUICER_COMFY_LOG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "comfy_server.log"),
)

def is_running():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False

def start():
    if is_running():
        print("ComfyUI server is already running on port 8188.")
        return True

    print("Starting ComfyUI server in background...")
    log_fp = open(LOG_FILE, "w", encoding="utf-8")
    cmd = [
        PY_BIN,
        "-s",
        os.path.join(COMFY_ROOT, "main.py"),
        "--listen", "127.0.0.1",
        "--port", "8188",
        "--disable-auto-launch",
        "--fp16-vae"
    ]
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, cwd=COMFY_ROOT)
    print(f"Spawned ComfyUI PID: {proc.pid}")

    t0 = time.time()
    while time.time() - t0 < 40:
        if is_running():
            print(f"ComfyUI server online in {time.time() - t0:.2f}s!")
            return True
        time.sleep(1)

    print("[ERROR] Timeout waiting for ComfyUI to start. Check comfy_server.log")
    return False

if __name__ == "__main__":
    start()
