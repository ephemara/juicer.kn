#!/usr/bin/env python3
"""scripts/build.py — juicer.kn Automated Build Script
Amalgamates kain/core/*.kn and compiles native executable juicer.exe (and jc.exe).
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def find_kain():
    # The Kain compiler monorepo is always a sibling folder named "kain/"
    # (i.e. the project root above us). Never hardcode a drive letter.
    local = ROOT.parent / "kain" / ".kain" / "bin" / "kain.exe"
    if local.exists():
        return str(local)
    which = shutil.which("kain")
    if which:
        return which
    return "kain"

def run_cmd(cmd):
    print(f"--> {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, cwd=ROOT, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)

def main():
    kain = find_kain()
    print("================================================================================")
    print(" juicer.kn — DSP Post-Training Model Pruner & Harmonic Squeezer")
    print(f" Compiler: {kain}")
    print("================================================================================")

    # 0. Regenerate GPU kernels (source: kain/gpu/*.kn; output: .kain/gpu/, gitignored)
    print("\n[Step 0/3] Regenerating GPU artifacts kain/gpu/fusion_kernel.kn -> .kain/gpu/ ...")
    (ROOT / ".kain" / "gpu").mkdir(parents=True, exist_ok=True)
    run_cmd([kain, "gpu-artifacts", "kain/gpu/fusion_kernel.kn",
              "--target", "cuda", "--no-residency",
              "--output", ".kain/gpu/fusion"])

    # 1. Amalgamation
    print("\n[Step 1/3] Amalgamating kain/core/*.kn -> kain/juicer.kn ...")
    run_cmd([kain, "amalgamate", "--raw", "kain/core", "-o", "kain/juicer.kn"])

    # 2. Build Native LLVM
    print("\n[Step 2/3] Compiling native LLVM binary juicer.exe ...")
    run_cmd([kain, "build", "kain/juicer.kn", "--target", "llvm", "-o", "juicer.exe"])

    # 3. Copy alias
    print("\n[Step 3/3] Emitting CLI alias jc.exe ...")
    shutil.copy2(ROOT / "juicer.exe", ROOT / "jc.exe")

    print("\n================================================================================")
    print(" Build complete! juicer.exe & jc.exe ready to run.")
    print("================================================================================")

if __name__ == "__main__":
    main()
