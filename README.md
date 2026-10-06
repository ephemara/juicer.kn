<p align="center">
  <img src="assets/juicer-logo-256.png" alt="juicer.kn logo" width="128" height="128" />
</p>

# juicer.kn 🧃

<p align="center">
  <img src="https://img.shields.io/badge/Language-Kain%20Native-00e5ff.svg?style=for-the-badge" alt="Kain Native">
  <img src="https://img.shields.io/badge/Compiler-LLVM%20WPO-ff6d00.svg?style=for-the-badge" alt="LLVM WPO">
  <img src="https://img.shields.io/badge/Formal%20Prove-14--Point%20Battery-00e676.svg?style=for-the-badge" alt="14-Point Prove Battery">
  <img src="https://img.shields.io/badge/Dependencies-Zero%20(kernel32)-7c4dff.svg?style=for-the-badge" alt="Zero Dependencies">
  <img src="https://img.shields.io/badge/IO%20Speed-4.5%20GB%2Fs%20DMA-e91e63.svg?style=for-the-badge" alt="DMA IO">
</p>

### Post-Training Model Compression Workbench: Profile → Squeeze → Quantize → Slice → Fuse → Schedule → Prove
*Squeeze the bloat out of models. Pure juice. Zero flab.*

**juicer.kn** (`jc`) is a native, zero-dependency compression toolchain written in **Kain** for post-training surgery on deep neural networks — **LLMs** (LLaMA / Qwen / Mistral) and **DiTs** (Wan 2.1, Flux, Chroma). It analyzes models as **Cascaded Digital Filter Networks** through 4 DSP lenses, then physically compresses them: mixed-precision auto-quantization against a VRAM budget, structural block excision with loader-safe healing, LoRA baking, and step-skip scheduling.

Everything it emits is **standard GGUF** — it loads natively in **llama.cpp, Ollama, LM Studio, ComfyUI, and vLLM** with no plugins. Think of it this way: llama.cpp is the runtime, **juicer is the workbench that builds what the runtime eats** — including its own native quantizers (Q2_K / Q4_0 / Q6_K / Q8_0, plus GPU Q4_K / Q6_K paths), so the full pipeline from raw BF16 master to VRAM-resident GGUF happens in one toolchain.

This is **not a one-and-done tool**. It's an iterative instrument: profile to find the flab, squeeze to a VRAM target, slice dead blocks, fuse your LoRAs, prove the math — re-profile and repeat until the model fits your GPU.

---

## 1. The Pipeline

```text
raw BF16 safetensors ──► profile ──► squeeze ──► slice ──► fuse ──► prove ──► resident GGUF
   (or GGUF)              (4 DSP      (--target-    (--drop      (bake        (14-pt
                           lenses)     vram)         + --heal)    LoRAs)       battery)
```

| Command | What it does |
|---|---|
| `profile <model.gguf\|.safetensors>` | Streams real weights via kernel32 DMA and runs all 4 DSP lenses per block. Verdicts: **EXCISE** (spectral + entropy agree), **CANDIDATE** (single-lens hit — never auto-pruned), or keep |
| `squeeze <model> --target-vram 5.5GB --out out.gguf` | Zero-calibration spectral auto-quant: greedy knapsack assigns mixed Q8_0 / Q4_0 / Q2_K per tensor to land under your VRAM budget |
| `quantize <model> [--lora adapter.st] [--type q4_k\|q6_k\|q8_0\|q4_0\|q2_k] --out out.gguf` | Native block re-quantizer (AVX2 Q8_0/Q4_0, super-block Q2_K/Q4_K/Q6_K) with optional lossless LoRA fusion. GPU Tensor-Core Q4_K/Q6_K path for raw masters |
| `slice <model.gguf> --drop 18,19,23,24 --out out.gguf [--heal]` | Physically excises blocks: contiguous re-index, aligned offset rewrite, in-place `*.block_count` KV patch (llama.cpp/Ollama-safe). `--heal` folds residual energy into downstream norms so loaders stay happy |
| `fuse <base.gguf> <lora.safetensors> [--strength 1.0] [--drop list] --out fused.gguf` | Bakes LoRA / LoKR adapters into weights. `--native` runs 100% in-Kain (on-device GEMM, Q8_0 out, E2E max err 0.0038) |
| `schedule <model> --steps 4 --out recipe.json` | Emits dynamic step-adaptive block-skipping recipe for ComfyUI, driven by the FrFT lens |
| `prove` | Runs the 14-point in-memory formal verification battery (decoders, lenses, KV-patch, quantizer SNR, roundtrips) |
| `gpu-probe` / `gpu-fuse-test` | Verifies the native CUDA bridge (context, VRAM, Kain-PTX JIT) and on-device FuseLoRA numerics (bit-exact, err 0) |

---

## 2. The Core Thesis: Transformers as Digital Filters

Standard quantization (FP16 → Q4_K_M) reduces numerical precision, but:
1. It **never reduces arithmetic complexity** ($O(N^2)$ attention and dense GEMMs remain unchanged).
2. It **never reduces layer-to-layer latency**.
3. On a memory-constrained GPU (e.g. 6 GB VRAM running an 11 GB model), it causes PCIe bus thrashing every step.

A Transformer residual block:
$$x_{l+1} = x_l + \text{MLP}(\text{Attention}(x_l))$$

is formally equivalent to a **Cascaded Multi-Channel IIR Filter Bank stage**.

- **Lens 1 — Spectral Transfer $H(\omega)$:** Power iteration + whole-block Frobenius energy, kurtosis, and condition number over dense samples of every 2D projection. Flat gain + no phase rotation = **all-pass / identity = dead weight**.
- **Lens 2 — Permutation Entropy Velocity $dH/dl$ + LZ76:** 5D Lehmer entropy, Jensen-Shannon complexity, Kaspar–Schuster compressibility. Middle layers with $dH/dl \approx 0$ are **computationally idling**.
- **Lens 3 — Bispectrum / Bicoherence $b^2$:** Kim & Powers IRPD bicoherence with Hann windowing. Separates harmonic feature binding from intermodulation distortion (the root of artifacts and hallucinations).
- **Lens 4 — Fractional Fourier Transform:** O(N log N) Ozaktas/Pei-Ding chirp engine evaluating non-stationary flow-matching trajectories → **dynamic step-adaptive skip schedules**.

**Dual-lens rule:** EXCISE requires spectral *and* entropy to agree. Single-lens hits are CANDIDATE — nominated, never pruned.

---

## 3. Verified Receipts (not projections)

Real runs on real weights. Estimates are labeled as estimates — see `AGENTS.md` for the full REAL-vs-threshold honesty ledger.

| Run | Result |
|---|---|
| Flux Klein 9B BF16 master (16.91 GB `.safetensors`) → Q4_K GGUF | **4.76 GB in 35.4 s**, single-pass GPU Tensor Cores |
| Same master + 2 LoRAs baked in float (UNLOCKED_V2 + snofs) → Q4_K | **4.76 GB in 61.3 s**, 112 tensors fused |
| Same master → Q2_K, 100% native CPU path | **2.78 GB**, gguf-py + ComfyUI-GGUF clean, **rendered in ComfyUI — works** |
| Flux Klein 4B BF16 (7.22 GB) → Q4_K_M via GPU | **2.03 GB in 14.7 s**, rendered **indistinguishable from base** (visual parity verified) |
| Flux Klein 9B Q4_K_M + consistency LoRA, 4 single-blocks dropped | **5.05 GB**, 185 tensors / 28 blocks, fused in 16.8 s on RTX 3000 |
| Flux Klein 4B FP8 master → mixed-precision squeeze | **2.36 GB** (42/1/42 mix), gguf-py clean |
| Slice + heal | 25 → 23 blocks, heal-applied, loader-safe |
| `fuse --native` end-to-end | max err **0.0038** vs exact (Q8_0 noise floor) |
| Quantizer SNR (prove fixtures) | Q8_0 **49.5 dB**, Q4_0 **25.0 dB**, Q2_K **16.1 dB** |
| On-device FuseLoRA vs CPU reference | **bit-exact, err 0** |

*Original design target: Wan 2.1 14B (40 blocks, 10.56 GB Q4_K_M) → excise ~8 dead blocks → VRAM-resident on 6 GB cards. Proven mechanically on the slicer (playable reduced-block Wan GGUFs in ComfyUI); per-model quality verdicts always come from rendering, never from the lenses alone.*

> **Encoder pairing note (Flux Klein):** the 4B model pairs with `qwen3-4b-heretic` via CLIPLoader type `flux2`; the 9B uses the uncensored Q6_K clip via CLIPLoaderGGUF. Wrong clip = silent garbage, regardless of quantization.

---

## 4. Building from Source

`juicer.kn` compiles via the Kain LLVM compiler into a single standalone native binary (< 1.5 MB):

```cmd
# Using Python
python scripts/build.py

# Or Windows Command Prompt
scripts\build.cmd
```

Zero Python runtime, zero PyTorch, zero CUDA toolkit required for profile / squeeze / slice / native quantize / prove. (The GPU Q4_K/Q6_K Tensor-Core path and legacy fusion use the sanctioned `scripts/` Python helpers; the CUDA bridge itself — nvcuda dynamic loader + Win64 trampoline + Kain-PTX JIT — is zero-dependency.)

---

## 5. Architectural Invariants

1. **Win32 kernel32 Direct DMA:** Bypasses CRT filesystem layers; streams GGUF headers and payloads at NVMe line rates (4–5 GB/s).
2. **Arena Allocations (+32 Slack Rule):** Zero GC. Arenas carry 32 bytes of slack for safe AVX2 256-bit SIMD over-reads.
3. **llama.cpp-exact codecs:** The native Q2_K super-block codec is bit-verified against torch dequant; container-validity gating keeps every emitted file loadable.
4. **Native GPU, no toolkit:** `gpu_bridge.kn` (nvcuda loader + 59-byte trampoline + Driver API) and `fusion_kernel.kn` (Kain shader compute: LoRA GEMM + LoKR Kronecker) — JIT'd PTX with no CUDA/PyTorch install.
5. **Formal verification:** `jc prove` — 14 in-memory batteries over decoders, lenses, KV-patch invariants, quantizer SNR, and SafeTensors→GGUF roundtrips.

---

## 6. Docs & Internals

- `docs/ARCHITECTURE.md` — system design
- `docs/DSP_LENSES.md` — the four lenses in depth
- `AGENTS.md` — operating rules + REAL-vs-placeholder ledger (required reading before quoting any number)
- `catalog.tsv` / `memory.tsv` — module catalog + dated engineering log
