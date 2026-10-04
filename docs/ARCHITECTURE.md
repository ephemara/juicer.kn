# juicer.kn — Architectural Specification

## 1. System Philosophy

`juicer.kn` is engineered according to three core systems principles:

1. **The SQLite / BusyBox Doctrine:**
   Source code is maintained in strictly modular, domain-isolated `.kn` files during development:
   - `_common.kn`: Win32 DMA FFI, binary decoders, memory arenas, string helpers.
   - `gguf_parser.kn`: Fast zero-copy GGUF v2/v3 header & tensor directory reader.
   - `spectral_lens.kn`: Lens 1 — Transfer function $H(\omega)$ & eigenvalue attenuation.
   - `entropy_lens.kn`: Lens 2 — Permutation entropy velocity $dH/dl$ & LZ76 complexity.
   - `bispectrum_lens.kn`: Lens 3 — Normalized bicoherence ($b^2$) & quadratic phase coupling.
   - `frft_lens.kn`: Lens 4 — Fractional Fourier transform chirp tracking & step schedules.
   - `slicer.kn`: Physical zero-copy GGUF excision & topology rewriter.
   - `scheduler.kn`: JSON step-skipping recipe generator.
   - `prove.kn`: 7-point in-memory formal verification battery.
   - `dispatch.kn`: Unified CLI entry point (`main()`).

   During build, `kain amalgamate --raw kain/core -o kain/juicer.kn` fuses the entire project into a single compilation unit, which LLVM compiles with Whole Program Optimization (WPO) into a single standalone executable (`juicer.exe` / `jc.exe`).

2. **Direct OS DMA (`kernel32`):**
   Standard Python/C++ file I/O traverses multiple layers of abstraction (CRT buffers, page caches, vector copies). `juicer.kn` uses direct Windows Win32 API calls (`k_CreateFileA`, `k_SetFilePointer`, `k_ReadFile`, `k_WriteFile`) streaming into aligned arenas. This yields real measured read throughput of 4.5+ GB/s directly from NVMe drives.

3. **Arena Allocations (+32 Slack Rule):**
   Zero runtime garbage collection. All working memory is allocated via:
   ```kain
   alloc_zeroed(n + 32, "Byte")
   ```
   The trailing 32 bytes of slack guarantee that AVX2 256-bit SIMD registers can safely over-read array bounds in hot loops without triggering page fault boundaries or segmentation violations.

---

## 2. GGUF Parsing & Zero-Copy Slicing State Machine

```
   ┌────────────────────────────────────────────────────────┐
   │            Source GGUF File (e.g. 10.56 GB)            │
   └───────────┬────────────────────────────────┬───────────┘
               │                                │
    [ 1. DMA Header Read ]           [ 2. Zero-Copy Payload ]
    - Magic ('GGUF')                 - Excluded blocks skipped
    - KV metadata                    - Surviving blocks streamed
    - Tensor directory (1303 items)    at NVMe line speed (4.5 GB/s)
               │                                │
               ▼                                ▼
    [ 3. Block Index Rewriter ]      [ 4. Contiguous Packing ]
    - drop: [18, 19, 23, 24]         - Recomputed offsets
    - blocks.20.* -> blocks.18.*     - 32-byte alignment padding
    - tensor_count updated           - Pristine GGUF emitted
               │                                │
               └───────────────┬────────────────┘
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │         Juiced Sliced GGUF (e.g. 8.58 GB)              │
   │    Natively runnable in ComfyUI, llama.cpp, Ollama     │
   └────────────────────────────────────────────────────────┘
```

---

## 3. Contiguous Block Re-indexing Invariant

When Transformer blocks $D = \{d_1, d_2, \dots, d_k\}$ are excised:
For every surviving block $b \notin D$, its new block index $b'$ is computed as:
$$b' = b - |\{d \in D : d < b\}|$$

Tensor names are rewritten accordingly:
- `blocks.<old>.self_attn.k.weight` $\to$ `blocks.<new>.self_attn.k.weight`
- `double_blocks.<old>.img_mod.weight` $\to$ `double_blocks.<new>.img_mod.weight`

This guarantees that downstream model loaders (which iterate $0 \le i < \text{block\_count}$) find a continuous, gap-free layer topology.
