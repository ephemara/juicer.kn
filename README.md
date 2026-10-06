<p align="center">
  <img src="assets/juicer-logo-256.png" alt="juicer.kn" width="128" height="128">
</p>

<h1 align="center">juicer.kn</h1>

<p align="center">
  <b>Post-training compression workbench for transformers and diffusion models.</b><br>
  Profile &middot; squeeze &middot; quantize &middot; slice &middot; fuse &middot; schedule &middot; prove.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Kain_native-00e5ff.svg" alt="Kain native">
  <img src="https://img.shields.io/badge/compiler-LLVM_WPO-ff6d00.svg" alt="LLVM WPO">
  <img src="https://img.shields.io/badge/verification-14--point_battery-00e676.svg" alt="14-point battery">
  <img src="https://img.shields.io/badge/dependencies-zero_(kernel32)-7c4dff.svg" alt="Zero dependencies">
  <img src="https://img.shields.io/badge/IO-4.5_GB%2Fs_DMA-e91e63.svg" alt="DMA IO">
</p>

`juicer.kn` (`jc`) is a native, zero-dependency toolchain for post-training compression of deep neural networks — large language models and diffusion transformers. It analyzes models as cascaded digital filter networks, then physically compresses them: mixed-precision auto-quantization against a VRAM budget, structural block excision with loader-safe healing, adapter fusion, and step-skip scheduling.

All output is **standard GGUF**. Files produced by `juicer.kn` load unmodified in common GGUF runtimes (llama.cpp, Ollama, LM Studio, ComfyUI, vLLM). If the runtime is the engine, `juicer.kn` is the workbench that builds its fuel — including native quantizers (Q2_K / Q4_0 / Q6_K / Q8_0, plus GPU Q4_K / Q6_K paths), so the full path from raw BF16 master to VRAM-resident model runs inside one toolchain.

---

## Features

- **DSP redundancy analysis** — four independent lenses (spectral transfer, permutation-entropy velocity, bicoherence, fractional Fourier) score every block on real streamed weights. Excision requires two-lens agreement; single-lens hits are candidates only, never auto-pruned.
- **VRAM-targeted auto-quantization** — `squeeze` solves a per-tensor mixed-precision assignment (Q8_0 / Q4_0 / Q2_K) against a `--target-vram` budget. No calibration data required.
- **Native quantizers** — AVX2 Q8_0 / Q4_0 and super-block Q2_K / Q4_K / Q6_K codecs implemented in pure Kain, with container-validity gating so every emitted file stays loader-clean.
- **Structural slicing with healing** — block excision with contiguous re-indexing, aligned offset rewrites, in-place `*.block_count` metadata patching, and optional residual-energy folding into downstream norms (`--heal`).
- **Adapter fusion** — bake LoRA / LoKR adapters directly into weights, in float on GPU Tensor Cores or 100% natively in-Kain (`--native`).
- **Step-skip scheduling** — FrFT-driven dynamic block-skipping recipes emitted as JSON for step-adaptive inference.
- **Formal verification** — a 14-point in-memory prove battery over decoders, lenses, metadata patching, quantizer SNR, and format roundtrips. Run it with `jc prove`.
- **Zero-dependency native GPU** — a built-in CUDA driver bridge (dynamic loader + Win64 trampoline + PTX JIT). No CUDA toolkit, PyTorch, or Python runtime on the compression path.

---

## Usage

```bash
# Profile: score every block with all four DSP lenses
jc profile model.gguf

# Squeeze: mixed-precision auto-quant to fit a VRAM budget
jc squeeze model.safetensors --target-vram 5.5GB --out squeezed.gguf

# Quantize: explicit precision target, optional adapter baked in
jc quantize model.gguf --type q4_k --out model-q4_k.gguf
jc quantize model.safetensors --lora adapter.safetensors --type q6_k --out fused-q6_k.gguf

# Slice: excise dead blocks, heal the seams
jc slice model.gguf --drop 18,19,23,24 --out slim.gguf --heal

# Fuse: bake an adapter into a base model
jc fuse base.gguf adapter.safetensors --strength 1.0 --out fused.gguf
jc fuse base.gguf adapter.safetensors --native --out fused-q8_0.gguf

# Schedule: step-adaptive skip recipe
jc schedule model.gguf --steps 4 --out recipe.json

# Prove: 14-point formal verification battery
jc prove
```

Inputs route automatically: `.gguf` (v2/v3, F32/F16/BF16/Q8_0/Q4_0/Q4_K) and raw `.safetensors` masters (BF16/F16/F32, incl. FP8) are both first-class.

---

## Background: transformers as digital filters

Conventional quantization lowers numerical precision but never lowers arithmetic complexity — attention stays quadratic and every layer still executes every step. On memory-constrained GPUs this means the bus, not the ALUs, sets the pace.

A residual block

$$x_{l+1} = x_l + \text{MLP}(\text{Attention}(x_l))$$

is formally a cascaded multi-channel IIR filter stage. Some stages are all-pass (flat gain, no phase rotation) or idling (zero entropy velocity — recirculating state without transforming it). Those stages are cost without transformation, and they are what `juicer.kn` finds and removes. See `docs/DSP_LENSES.md` for the full treatment.

---

## Representative results

Measured runs on real weights (lenses nominate, rendering decides — per `AGENTS.md`, verdict thresholds remain unvalidated and quality is always confirmed by inference, never by scores alone):

| Run | Outcome |
|---|---|
| 16.9 GB BF16 diffusion master → Q4_K, single-pass GPU Tensor Cores | 4.76 GB in ~35 s |
| Same master + two adapters baked in float → Q4_K | 4.76 GB in ~60 s, 112 tensors fused |
| Same master → Q2_K, fully native CPU path | 2.78 GB; third-party GGUF tooling parses clean; renders correctly |
| 7.2 GB BF16 master → Q4_K, GPU path | 2.03 GB in ~15 s; rendered output visually at parity with base |
| Block-drop + adapter fusion (4 blocks excised) | 5.05 GB, 185 tensors / 28 blocks, fused in ~17 s |
| FP8 master → mixed-precision squeeze | 2.36 GB mixed mix, third-party tooling clean |
| Native Q2_K SNR (prove fixture) | 16.1 dB |
| Native Q8_0 / Q4_0 SNR (prove fixtures) | 49.5 dB / 25.0 dB |
| In-Kain GPU fusion vs exact reference | max err 0.0038 (Q8_0 noise floor); GEMM/LoKR paths bit-exact |

---

## Building from source

Compiles via the Kain LLVM compiler into a single standalone native binary (< 1.5 MB):

```cmd
python scripts/build.py
rem or
scripts\build.cmd
```

No Python, PyTorch, or CUDA toolkit is required for profiling, squeezing, slicing, native quantization, or proving.

---

## Verification and honesty

`jc prove` runs the full 14-point battery. The measurement layer reads real weights off disk over kernel32 DMA; the analytical thresholds behind EXCISE verdicts are documented as unvalidated in `AGENTS.md`, which also carries the complete REAL-vs-placeholder ledger. Read it before quoting any number from this repo.

---

## Documentation

- `docs/ARCHITECTURE.md` — system design
- `docs/DSP_LENSES.md` — the four analysis lenses
- `AGENTS.md` — operating rules and the honesty ledger
- `catalog.tsv` / `memory.tsv` — module catalog and engineering log

---

## License

MIT — see [LICENSE](LICENSE).
