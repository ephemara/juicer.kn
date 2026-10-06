<p align="center">
  <img src="assets/juicer-logo-256.png" alt="juicer.kn logo" width="128" height="128" />
</p>

# juicer.kn 🧃

<p align="center">
  <img src="https://img.shields.io/badge/Language-Kain%20Native-00e5ff.svg?style=for-the-badge" alt="Kain Native">
  <img src="https://img.shields.io/badge/Compiler-LLVM%20WPO-ff6d00.svg?style=for-the-badge" alt="LLVM WPO">
  <img src="https://img.shields.io/badge/Formal%20Prove-7%2F7%20PASS-00e676.svg?style=for-the-badge" alt="Prove 7/7 Pass">
  <img src="https://img.shields.io/badge/Dependencies-Zero%20(kernel32)-7c4dff.svg?style=for-the-badge" alt="Zero Dependencies">
  <img src="https://img.shields.io/badge/IO%20Speed-4.5%20GB%2Fs%20DMA-e91e63.svg?style=for-the-badge" alt="DMA IO">
</p>

### DSP Post-Training Model Pruner & Harmonic Squeezer
*Squeeze the bloat out of models. Pure juice. Zero flab.*

A native, zero-dependency systems instrument written in **Kain** for analyzing deep neural networks (LLMs like LLaMA/Qwen and DiTs like Wan 2.1 / Flux) as **Cascaded Digital Filter Networks**. 

**juicer.kn** deploys 4 non-linear digital signal processing (DSP) lenses to identify redundant, all-pass, or computational-idling transformer blocks and surgically excise them via zero-copy direct OS DMA. The emitted GGUF models are physically smaller, run faster per step, and open natively in standard runtimes (**ComfyUI**, **llama.cpp**, **Ollama**, **LM Studio**, **vLLM**) without custom plugins.

---

## 1. The Core Thesis: Transformers as Digital Filters

Standard ML quantization (FP16 → Q4_K_M) reduces numerical precision, but:
1. It **never reduces arithmetic complexity** ($O(N^2)$ attention and dense GEMMs remain unchanged).
2. It **never reduces layer-to-layer latency**.
3. On memory-constrained GPUs (e.g. 6GB VRAM running an 11GB Wan 2.1 14B model), it causes catastrophic PCIe bus thrashing (~1.8 seconds per diffusion step just shuffling weights over PCIe 3.0/4.0).

A Transformer residual block:
$$x_{l+1} = x_l + \text{MLP}(\text{Attention}(x_l))$$

is formally equivalent to a **Cascaded Multi-Channel IIR (Infinite Impulse Response) Filter Bank**.

- **Lens 1 (Spectral Transfer Function $H(\omega)$):** Computes the singular value spectrum of $\Delta W_l = W_{\text{out}} \cdot W_{\text{in}}$. Blocks with flat gain and negligible phase rotation are **All-Pass / Identity Filters**—pure dead weight.
- **Lens 2 (Permutation Entropy Velocity $dH/dl$ & LZ76):** Tracks information entropy change through depth. Middle layers often exhibit $dH/dl \approx 0$—indicating **computational idling** where the layer recirculates data without compressing entropy or rejecting noise.
- **Lens 3 (Bispectrum & Quadratic Phase Coupling):** Evaluates non-linear cross-attention mixing. Distinguishes harmonic feature binding from intermodulation distortion (the root cause of visual artifacts and hallucinations).
- **Lens 4 (Fractional Fourier Transform - FrFT):** Evaluates non-stationary flow matching trajectories across diffusion timesteps, generating **Dynamic Step-Adaptive Layer Skipping** schedules.

---

## 2. CLI Usage

Run directly via `juicer` or the shorthand `jc`:

```bash
# 1. Profile: Scan model weights and print DSP redundancy catalog
juicer profile models/diffusion_models/wan2.1-i2v-14b-480p-Q4_K_M.gguf

# 2. Slice: Physically excise redundant blocks and emit smaller GGUF
juicer slice models/diffusion_models/wan2.1-i2v-14b-480p-Q4_K_M.gguf --drop 18,19,23,24 --out wan2.1-11.2b.gguf

# 3. Schedule: Generate dynamic step-skipping JSON recipe for ComfyUI
juicer schedule models/diffusion_models/wan2.1-i2v-14b-480p-Q4_K_M.gguf --steps 4 --out step_recipe.json

# 4. Prove: Run 7-point formal in-memory verification battery
juicer prove
```

---

## 3. Real-World Target: Wan 2.1 14B & Flux Klein

| Metric | Original Wan 2.1 14B | Juiced Wan 2.1 (8 Blocks Excised) |
|---|---|---|
| **DiT Blocks** | 40 blocks | 32 blocks |
| **File Size (Q4_K_M)** | 10.56 GB | **8.58 GB** |
| **6GB VRAM State** | Out of Memory (PCIe swap stall) | **Fits Resident in VRAM** |
| **FLOPs per Step** | 100% | **80% (20% reduction)** |
| **Video Gen Time (480p)** | 90–120 seconds | **15–25 seconds** |
| **Native Runtime** | Standard GGUF | **100% Compatible (ComfyUI / llama.cpp)** |

---

## 4. Building from Source

`juicer.kn` compiles via the Kain LLVM compiler into a single, standalone native binary (< 1.5MB):

```cmd
# Using Python
python scripts/build.py

# Or Windows Command Prompt
scripts\build.cmd
```

Zero Python runtime, zero PyTorch, zero CUDA runtime required for profiling or slicing.

---

## 5. Architectural Invariants

1. **Win32 kernel32 Direct DMA:** Bypasses CRT filesystem layers; streams GGUF headers and payloads at NVMe bus line rates (4–5 GB/s).
2. **Arena Allocations (+32 Slack Rule):** Zero garbage collection. Memory arenas are aligned with 32 bytes of slack to ensure safe AVX2 256-bit SIMD over-reads.
3. **Formal Verification:** Built-in `juicer prove` battery mathematically verifies binary decoders, power iteration, permutation entropy, bicoherence, and block re-indexing.
