# AGENTS.md — juicer.kn

Onboarding + operating rules for any AI agent working in this repository.
Read this before touching code.

> **START HERE: `docs/ARCHITECTURE.md` & `docs/DSP_LENSES.md`** — the design.
> This file describes *how to work in this repo* and *how the Kain language works*.
> The docs describe *what we are building and why*: a bare-metal DSP instrument that
> treats Transformer / DiT residual blocks as **cascaded digital filter stages**, proves
> which ones are all-pass or idling, and physically excises them from a **GGUF** at
> NVMe DMA speed — then bakes LoRAs (Lightning / motion / style) straight into the
> remaining weights — so a 10.56 GB model becomes a 4–6 GB one that runs **resident in
> 6 GB VRAM** instead of thrashing PCIe for minutes per clip.
>
> **The Kain compiler monorepo is always a sibling folder named `kain/` — reference it
> as `../kain/`. Never hardcode a drive letter.** All Kain projects in this workspace
> live in the same parent directory; the parent is implied, never written. The only
> path that ever moves is the parent itself, and relative paths absorb that for free.
> The fast compiler binary is `../kain/.kain/bin/kain.exe`; if that is absent, `kain`
> is on PATH and self-resolves its `KAIN_HOME`. Same binary, same behaviour.
>
> **Sibling repos (all peers, all reachable relative to this repo):**
> - **TurboKain (`../TurboKain/`)** — the **production Kain reference code ground truth**:
>   21–23 instruments, 30,000+ lines of real shipped Kain proving whole-program
>   amalgamation, AVX2 `converge` fast lanes, `alloc_zeroed(n + 32, "Byte")` arena
>   discipline, and 77× faster `kernel32` DMA IO. Read it before guessing syntax,
>   system patterns, or real-world Kain architecture. **Read-only reference — never edit.**
> - **k_AI_n (`../k_AI_n/`)** — sibling AI-engine project (native multimodal engine in
>   pure Kain). Shares this repo's vendored Kain baseline verbatim (see §4).
> - **Kain compiler (`../kain/`)** — the language compiler repo. Source of truth for the
>   runtime, stdlib, and benchmarks. **Read-only reference — never edit, never build it.**
> - **MarkScript (`../MarkScript/`)** — `.md` campaign notebooks that script Kain exes.
> - **kainc (`../kainc/`)** — compiler-adjacent tooling.

---

## 1. What `juicer.kn` Is (The Mission)

`juicer.kn` (`jc`) is a **native, zero-dependency systems instrument written in Kain**
for post-training compression of deep neural networks — **LLMs** (LLaMA / Qwen / Mistral)
and **diffusion transformers** (Wan 2.1, Flux, Chroma).

Quantization (FP16 → Q4_K_M) lowers numerical precision but **never lowers arithmetic
complexity**: $O(N^2)$ attention and dense GEMMs are untouched, and on a 6 GB GPU an
11 GB model still thrashes the PCIe bus every step. `juicer.kn` attacks the *other*
axis: **structural redundancy**.

A Transformer residual block

$$x_{l+1} = x_l + \text{MLP}(\text{Attention}(x_l))$$

is formally a **cascaded multi-channel IIR filter stage**. Some stages are all-pass
(flat gain, no phase rotation) or idling ($dH/dl \approx 0$ — they recirculate latent
state without compressing entropy). Those are dead weight: cost without transformation.

### What ships today (all live, all receipted)

| Capability | Command | State |
|---|---|---|
| **GGUF Direct DMA Parse** — header, KV directory, tensor descriptors, DiT block topology | `jc profile` | ✅ 10.56 GB / 1,303 tensors / 40 blocks scanned in ~40 ms |
| **Physical Zero-Copy Slicer** — excise blocks, recompute offsets, re-index `blocks.*` contiguously, patch `block_count`, DMA-stream survivors | `jc slice` | ✅ emits a valid GGUF (ComfyUI / llama.cpp / Ollama / LM Studio) |
| **LoRA Fuse Engine** — bake rank-K adapters into base weights (exact FP32/FP16 add for 1D biases/norms; Tensor-Core GEMM + inline dequant + fast GPU Q4_K quant for 2D quantized linears) | `jc fuse` | ✅ ~0.9 s per block; fuse **and** slice in one pass |
| **Dynamic Step-Skip Scheduler** — emits a JSON step→block recipe for ComfyUI / vLLM / llama.cpp | `jc schedule` | ✅ |
| **Formal Prove Battery** — 8 in-memory mathematical batteries, green in <1 s | `jc prove` | ✅ 8/8 PASS |

Progressive carving: a 40-block Wan 2.1 → 24 → 20 → 18 → 16 blocks, each losing
~247 MB; the practical target is **≤ 20 blocks ≈ 5 GB so the whole thing sits in 6 GB
VRAM and PCIe offload drops to zero** (the 10–28× speed multiplier is the *bus* jump
from ~10 GB/s PCIe to ~288 GB/s GDDR6, not the FLOPs).

### The two calibration layers still maturing

1. **Live weight dequantization in the lenses.** Today the lenses read block descriptors,
   tensor topology, and quantization headers to synthesize per-block transfer curves. Adding
   an inline Q4_K/Q8_0 dequantizer kernel lets Lens 1/2 crunch the *exact* unscaled weights
   for pinpoint singular values.
2. **Downstream metadata sensitivity.** After dropping blocks, `block_count`-style metadata
   must be patched correctly (LLaMA uses `llama.block_count`; the ComfyUI GGUF loader
   enumerates `blocks.*`). Verify a sliced model actually boots before shipping a profile.

---

## 2. What Kain Is (The Language Substrate)

Kain is a real, shipped systems language — **not** a scripting toy, a DSL, or a research
sketch. It compiles through **LLVM to native `.exe`** (also `.dll`, `.so`, `.obj`, `.a`),
has a full REPL/TUI, and is paranoid by design: **Z3 theorem provers and CBMC formal
assertions are integrated throughout the compiler and runtime** — 500+ Z3 proof packs,
380+ SMT-LIB2 files, 10,000+ CBMC assertions ship with it. The assumption is *all code
is fundamentally broken until mathematically proven otherwise.*

Python-like syntax is camouflage over an inverted machine — **Kain is not Rust; do not
write Rust-with-Kain-syntax.** The compiler, not the programmer, owns state, mutation,
dispatch, timing, coupling, layout, and handoff.

The decision ladder, top-down (stop at the first rung that fits):

```
L7 systems    actor · collapse/observe/decay · spawn/send/on
L6 stones     axiom · shatter · teleport
L5 temporal   pulse · resonate · dampen · every · jitter
L4 stage      orchestrate · stage · deps · policy · residency · transfer
L3 dispatch   converge · spec · fast · capability · verify
L2 integrity  patch · law
L1 authority  world · entangle · surface
L0 plain      fn · struct · let · mut · enum · trait · impl
```

**Effects lattice:** functions declare effect signatures (`with Pure`, `with IO`,
`with Async`, `with GPU`, `with Reactive`, `with Unsafe`). An un-annotated function
cannot execute hidden side effects. `juicer.kn` is almost entirely `with Unsafe` — it
writes through raw pointers and calls Win32.

**Dual host + GPU:** Kain compiles CPU host code and GPU compute kernels
(`shader compute`) from the same `.kn` file, emitting SPIR-V / PTX / HLSL. `juicer.kn`
does *not* need this — its hot path is **CPU + kernel32 DMA**. (The CUDA work lives in
the Python `fuse_engine.py` sidecar, see §6.)

For `juicer.kn` the constructs that actually matter are **`fn` / `struct` / `let` /
`while` / `if`** (the whole `kain/core` suite is plain L0 with `@extern` FFI), plus
`converge` (spec + verified AVX2 fast lanes) and `collapse`/`observe`/`decay` (arena
lifecycles) from TurboKain when you start touching bulk byte loops.

---

## 3. CLI Commands & Build Workflow

### The build pipeline (run from the repo root)

```bash
# 1. Pack the flat core suite into a single whole-program translation unit:
kain amalgamate --raw kain/core -o kain/juicer.kn

# 2. Compile via LLVM Whole-Program Optimization:
kain build kain/juicer.kn --target llvm -o juicer.exe
cp juicer.exe jc.exe          # CLI shorthand alias
```

Repo wrappers (already de-hardcoded to resolve the compiler relatively):

```cmd
python scripts\build.py        # amalgamate -> build -> copy alias
scripts\build.cmd              # same, pure cmd
```

Both look for the compiler at `../kain/.kain/bin/kain.exe` first, then fall back to
`kain` on PATH. Keep it that way — **no drive letters.**

### Running the instrument

```cmd
jc help                                     # operational handbook
jc prove                                    # 8-battery formal in-memory verification
jc profile <model.gguf>                     # DSP redundancy catalog + recommended --drop
jc slice  <model.gguf> --drop 18,19,23,24 --out juiced.gguf
jc fuse   <model.gguf> <lora.safetensors> [--strength 1.0] [--drop <list>] --out fused.gguf
jc schedule <model.gguf> --steps 4 --out recipe.json
```

`juicer.exe` and `jc.exe` are byte-identical; `jc.cmd` is a shim that forwards `%*` to
`%~dp0juicer.exe`, so `jc <cmd>` works from anywhere inside the repo.

### General Kain commands

```bash
kain check kain/core/_common.kn            # typecheck, no artifacts
kain check kain/core/_common.kn --json     # machine-readable diagnostics
kain build kain/core/<mod>.kn --target llvm -o <mod>.exe   # single-module spike build
kain run  <file>.kn --target llvm -- <args>  # compile + run + pass argv
kain run dev <file>.kn                     # watch + re-run on change
kain test kain/ --json                     # compiletest-directive tests
kain fmt  kain/ --check                    # formatting check (--write to fix)
kain clean --scope build                   # drop build artifacts
kain doctor                                # environment / wiring diagnostics
kain repl                                  # TUI: edit + compile to LLVM live
kain -c 'println("hi")' -r -t llvm         # one-shot probe (no argv with -c)
```

Full surface: `docs/kain/tsv/cli_commands.tsv`.

### Two commands — do NOT confuse them

| Command | What it is | Use it? |
|---|---|---|
| **`kain`** | the **fast** native compiler binary (launches instantly) | ✅ **always, for all juicer.kn work** |
| `kaindev` | the Bazel dev auto-sync shim (rebuilds the compiler from source) | ❌ **never**, unless actively editing the compiler itself |

**Never prepend `.kain/bin` or a Bazel directory to PATH to "fix" anything.** `kain`
is already on PATH and resolves its own `KAIN_HOME` + runtime library. If you see
`failed to start bazel`, you ran `kaindev` by mistake.

### Stdlib discovery (`Unknown identifier` ≠ broken code)

Kain finds the stdlib by searching, in order: `KAIN_STDLIB_PATH` → `$KAIN_HOME/stdlib`
→ ancestors of the running `kain.exe` → ancestors of the cwd. **Builtins** (`print`,
`fs_exists`, `runtime_*`) are compiled in; **module symbols** (`process_user_args`,
`parse_int`, `os_getenv`) come from `stdlib/*.kn` **on disk**. If a correctly-imported
module reports `Unknown identifier`, it is an environment/discovery problem — check
`kain doctor`. Do not rewrite the import. Do not edit the toolchain.

The stdlib source lives at `../kain/stdlib/` (~71 `.kn` modules). Signature lookup
without reading source: **`docs/kain/tsv/stdlib.tsv`** (`module · symbol · kind ·
signature · purpose`). Grep it before guessing.

**`build` is the gate.** `kain check` cannot synthesize pointers to verify `converge`
blocks over `ptr` parameters — equivalence receipts live in `main()` exit codes.

---

## 4. The Vendored Kain Baseline (`docs/kain/`)

Agents do not know Kain. It is learned from **this repo's vendored corpus**, never by
leaving to read the compiler repo. `docs/kain/` is **byte-identical to
`../k_AI_n/docs/kain/`** — the two repos deliberately carry the same baseline so either
one is self-contained and portable. Treat it as **read-only vendor content**: do not
edit, reformat, or add to it. If the baseline needs updating, update both copies.

- **`docs/kain/tsv/` — THE LEAN TRUTH TABLES (grep here first!).** 36 tables mapping the
  whole language with zero prose bloat:
  - `cli_commands.tsv` — every CLI command, option, and flag.
  - `keywords.tsv` — all 111 keywords, lexical groups, layers.
  - `stdlib.tsv` — every stdlib symbol: `module · symbol · kind · signature · purpose`.
  - `decision_ladder.tsv` — L0 → L7.
  - `converge.tsv` / `orchestrate.tsv` — dispatch and stage-graph grammar.
  - `ownership.tsv` — `collapse` / `observe` / `decay` / arena lifecycles.
  - `error_codes.tsv` + `error_codes_parser.tsv` + `error_codes_typechecker.tsv`.
  - `troubleshooting.tsv` — failure modes first, prose last.
- **Guides:** `KAIN_BY_EXAMPLE.md` (tutorial with compilable snippets),
  `KEYWORDS.MD` (111-keyword dictionary), `SHADER_GPU.MD` (compute shaders / dispatch),
  `SYSTEMS_PROGRAMMING.MD` (**the metal guide** — inline asm, fences, prefetch, huge
  pages, NUMA, `comptime` tables, the CRUSHER fusion pattern), `GLOSSARY.md`, `CATALOG.md`.
- **`stdlib.kn` — the amalgamated stdlib (1.5 MB, ~39,300 lines).** Need the exact body
  of a stdlib function? Grep it directly:
  `grep -n "pub fn process_user_args" docs/kain/stdlib.kn`.
- **Examples:** `examples/CRUSHER.kn`, `examples/metal.kn`, `examples/core_os.kn`,
  `examples/fusion_chain.kn`, `examples/sieve-pattern.kn` (the mold every TurboKain exe
  follows: `converge` spec + AVX2 lanes + arena buffers), `examples/keyword_crucible.kn`;
  plus `_llm_proto_examples/` (tokenizers, embeddings, transformer kernels),
  `_gpu_cpu_examples/`, `_shader_examples/`, `training/kain_omni.kn` (one compilable file
  exercising every layer L0–L7 — the single best teacher in the corpus).
- **`THE_MESSIAH.KN` — the ultra file:** ~6,029 modules, ~795,000 lines, ~36 MB of raw
  Kain amalgamation. The ground truth for how Kain is written *at scale*.
  **Never `read` the whole file** — it is a reference corpus, not a dependency, not
  something to build or import. Grep it or read slices around a hit:
  `grep -n "converge " docs/kain/THE_MESSIAH.KN | head`.

---

## 5. The 4 DSP Lenses (what `juicer` actually computes)

| # | Module | Signal-theoretic object | Verdict it mints |
|---|---|---|---|
| 1 | `spectral_lens.kn` | Transfer function $H(\omega)$ — power iteration on $\Delta W_l = W_{out} \cdot W_{in}$ → spectral radius $\rho$, condition number $\kappa$, dB gain | `is_all_pass` → **EXCISE** |
| 2 | `entropy_lens.kn` | 3D Lehmer ordinal permutation entropy $H_{PE}$, velocity $v_H = dH/dl$, Kaspar–Schuster LZ76 complexity | `is_idling` → **EXCISE** |
| 3 | `bispectrum_lens.kn` | Higher-order spectral analysis: normalized bicoherence $b^2$ → quadratic phase coupling vs. intermodulation | harmonic binding vs. artifact source |
| 4 | `frft_lens.kn` | Fractional Fourier transform chirp matched filtering over diffusion flow trajectories | step→block skip schedule |

Verdicts are printed as a table with the **exact `--drop` list** and reclaimed bytes.
`jc profile` never writes a model; `jc slice` never decides what to drop.

---

## 6. Repo Layout & The Flat Core Doctrine

> **WHY FLAT BEATS MICRO-FOLDER SPRAWL:**
> Deep nested trees (`src/a/b/c/d/`) fragment context across editor tabs, drift relative
> imports, and hide zero-copy buffer lifecycles. Following the battle-tested TurboKain
> architecture, **all core modules live in a single flat directory: `kain/core/`.**
>
> 1. **Zero navigation friction** — every module is an immediate peer.
> 2. **Deterministic ASCII amalgamation** — `_common.kn` sorts first (`_` < `b`), so
>    shared allocators, FFI declarations, constants, and assertions are declared before
>    any module references them.
> 3. **Clean build** — one `amalgamate --raw` + one `build`.

```
juicer.kn/ (repo root)
├── AGENTS.md                  # this file
├── README.md                  # public-facing pitch + CLI quickstart
├── jc.cmd                     # %~dp0juicer.exe %*  (CLI shim)
├── juicer.exe / jc.exe        # build outputs (gitignored)
│
├── docs/
│   ├── ARCHITECTURE.md        # systems spec: DMA state machine, slicing, invariants
│   ├── DSP_LENSES.md          # the four lenses, math + verdict semantics
│   └── kain/                  # VENDORED KAIN BASELINE — identical to ../k_AI_n/docs/kain
│       ├── tsv/               #   36 lean lookup tables (grep first)
│       ├── examples/ training/ _llm_proto_examples/ _gpu_cpu_examples/ _shader_examples/
│       ├── KAIN_BY_EXAMPLE.md  KEYWORDS.MD  SHADER_GPU.MD  SYSTEMS_PROGRAMMING.MD
│       ├── stdlib.kn          #   amalgamated stdlib (grep target)
│       └── THE_MESSIAH.KN     #   795k-line ultra reference (grep only, never read whole)
│
├── kain/
│   ├── core/                  # FLAT CORE SUITE — modular source, one concern per file
│   │   ├── _common.kn         # kernel32 FFI, +32-slack arenas, LE decoders, parsers, fmt
│   │   ├── gguf_parser.kn     # zero-copy GGUF v2/v3 reader + DiT block topology detection
│   │   ├── spectral_lens.kn   # Lens 1
│   │   ├── entropy_lens.kn    # Lens 2
│   │   ├── bispectrum_lens.kn # Lens 3
│   │   ├── frft_lens.kn       # Lens 4 + step schedule generator
│   │   ├── slicer.kn          # physical excision + contiguous re-index + header rewrite
│   │   ├── scheduler.kn       # JSON step-skip recipe emitter
│   │   ├── fuse.kn            # LoRA fusion driver (delegates to scripts/fuse_engine.py)
│   │   ├── prove.kn           # 8-battery in-memory formal verification suite
│   │   └── dispatch.kn        # the ONLY main(); CLI router + profile table printer
│   ├── juicer.kn              # raw amalgamation of kain/core/*.kn (build artifact)
│   └── .kain/                 # intermediate LLVM artifacts (gitignored)
│
├── scripts/
│   ├── build.py / build.cmd   # amalgamate -> build -> emit jc.exe
│   ├── fuse_engine.py         # CUDA Tensor-Core fuse core (needs a CUDA-enabled python)
│   ├── benchmark_wan.py       # end-to-end ComfyUI inference benchmark (aiohttp driver)
│   ├── start_comfy.py         # launch/attach the local ComfyUI server for benchmarks
│   └── memlog.kn / memlog.exe # native ledger helper — logs every file change
│
├── models/                    # SYMLINK to the ComfyUI models dir (gitignored)
├── .kain/{cache,out,reports}/ # build intermediates (gitignored)
├── memory.tsv                 # append-only change log — EVERY file change gets a row
└── catalog.tsv                # module/artifact ledger (status + receipt per row)
```

### `models/` is a symlink, not a vendored tree

`models/` points at the local ComfyUI `models/` directory (currently a symlink, so it
does not exist on a fresh checkout and **must never be committed**). Never hardcode the
target; take `--models-dir` / env, or pass the model path as an argv argument — which is
what every `jc` command already does.

### `scripts/fuse_engine.py` — the one sanctioned Python

The rule everywhere else in this repo is **Kain computes, Python never re-implements a
lane**. The fuse engine is the deliberate exception: it must run Tensor-Core GEMM +
inline GGUF dequant + re-quant, and the only thing on the box that can do that is a
CUDA-enabled Python with `torch` + `gguf` (the ComfyUI embedded interpreter). So:

- `kain/core/fuse.kn` is a **thin driver**: it validates argv, resolves an interpreter,
  and shells out. All policy and all fusion math live in `scripts/fuse_engine.py`.
- **Interpreter resolution (never hardcode a drive letter permanently):**
  `$JUICER_PY` → `$JUICER_COMFY_PY` → `S:/Local/ComfyUI_windows_portable/python_embeded/python.exe`
  (machine-local default, overridable) → `python` on PATH.
- **ComfyUI location overrides:** `$COMFYUI_HOME` (root of the portable install),
  `$COMFYUI_APP` (the `ComfyUI/` app dir), `$COMFYUI_OUTPUT` (output dir),
  `$JUICER_COMFY_GGUF` (the `ComfyUI-GGUF` custom-node dir providing `dequant`),
  `$JUICER_COMFY_LOG`. Set these when porting; do not edit the constants.
- Everything else (`profile`, `slice`, `schedule`, `prove`) is **100% native, zero
  Python, zero PyTorch, zero CUDA runtime**.

---

## 7. Non-Negotiable Rules

1. **You write programs in Kain; you do NOT modify the Kain compiler.**
   It lives outside this repo at `../kain/`. If code fails to compile, assume your Kain
   is wrong first. Blame the compiler only with a minimal isolated repro + receipt.
2. **No hardcoded drive letters or absolute paths in committed source.**
   The compiler is `../kain/`. Models come from argv or `--models-dir`. The parent
   directory is implied, never written. Machine-local facts (the CUDA interpreter, the
   ComfyUI install) are **env-overridable defaults** only: `$JUICER_PY`,
   `$JUICER_COMFY_PY`, `$COMFYUI_HOME`, `$COMFYUI_APP`, `$COMFYUI_OUTPUT`,
   `$JUICER_COMFY_GGUF`, `$JUICER_COMFY_LOG`. Every default must be replaceable without
   editing source.
3. **`build` is the gate, not `check`.**
   `kain check` cannot verify `converge`/pointer code. Verification receipts live in
   `main()` exit codes and the prove battery.
4. **Thresholds and dimensions are configuration data.**
   No magic numbers scattered through logic loops — DFT sizes, iteration counts,
   all-pass dB cutoffs, and idling tolerances belong in `_common.kn` alongside the
   existing `EXIT_*` / `GGUF_MAGIC` / `PI` constants.
5. **Arena allocations carry +32 bytes of SIMD slack.**
   `alloc_zeroed(n + 32, "Byte")` for **any** buffer a bulk loop can touch. Non-negotiable.
6. **Exactly one `decay` per arena per function** on the success path.
   Error paths leak; the OS reclaims on exit.
7. **Zero framework tax in the execution path.**
   No Python / PyTorch / CUDA DLL dependency in `juicer.exe`. It is a standalone native
   binary under 1.5 MB with `kernel32` as its only import family.
8. **Never write a sliced model without re-indexing it.**
   Dropping `blocks.18` and leaving `blocks.19..39` in place produces a model that opens
   and then fails. Contiguous renumbering to `blocks.0..blocks.N-1` plus the
   `*_block_count` metadata patch is part of the slice, not a follow-up.
9. **Prove before shipping a profile.** "It looks redundant" is not a status. A status is
   "Lens 1 all-pass at −19.7 dB, Lens 2 velocity 0.00, 4 blocks excised, 8.58 GB → 8.58 GB
   output opens in ComfyUI, `jc prove` 8/8 green."
10. **Receipts for everything.** Benchmark numbers come from a run, not an estimate.
    Estimate tables are labeled as estimates.
11. **Check `memory.tsv` before work; log with `scripts/memlog.exe` after.**
    Never touch code without reading recent history. Never finish a turn without logging
    your file modifications. No silent edits.

---

## 8. Common Pitfalls (Bled-For Knowledge — Do Not Rediscover)

- **`and` / `or` do not short-circuit.**
  ```kn
  // CRASHES on empty argv:
  if len(args) > 0 and args[0] == "--help": ...

  // CORRECT — nested:
  if len(args) > 0:
      if args[0] == "--help": ...
  ```
- **Bulk arenas require `+32` SIMD padding.** Bulk memory loops auto-vectorize into AVX2
  32-byte stores and overrun exact-size buffers by up to 31 bytes → exit code 127.
  `alloc_zeroed(nbytes + 32, "Byte")`, always. Debug heaps tolerate the overrun, which
  hides the bug until release.
- **`failed to start bazel`** → you invoked the dev shim (`kaindev`). Use plain `kain`.
- **Byte load/store semantics (exact).**
  - Read byte `i` as `Int`: `mem_load(ptr_offset(buf, i, "Byte"), "Int") & 255`
    (**not** `mem_load "Byte" as Int` — that reinterprets the 8-byte word).
  - Store byte `i`: `mem_store(ptr_offset(buf, i, "Byte"), v as Byte, "Byte")`.
  - `_common.kn` already wraps this: `bref(buf, i)` / `bstore(buf, i, v)`.
- **Parenthesize every bitwise/shift mix.** Precedence is not C-like and bites twice:
  - `if mem_load(...) & 255 != MAGIC:` is a **hard parse error** → `if (mem_load(...) & 255) != MAGIC:`
  - `acc = acc + byte_v << shift` **compiles and silently misaccumulates** → `acc = acc + (byte_v << shift)`.
    When a decoder reads back wrong, suspect missing parens **first**.
- **Effect-annotated functions must declare a return type.** A loop-bodied
  `pub fn f(...) with Unsafe:` with no `-> Type` segfaults at the call site (exit 127/139).
  Always `-> Int` + explicit `return 0`. Single-expression stores may survive without it —
  do not rely on that.
- **Never shadow an arena pointer with a same-named local.** `var v: Int` inside a loop
  while `let v: ptr<Byte>` is in scope corrupts ownership tracking: `check` passes, then
  `decay v` crashes with exit 127. Grep for `var <arena-name>` before trusting any decay crash.
- **One concern per loop.** A loop body that mixes two byte-stream codecs can silently
  drop one stream's writes. Split into two loops — this took one tool from 3/5 to 5/5
  with zero other changes.
- **Reserved words that bite:** `out`, `share`, `match`, `policy`, `deps`, `fast`, `spec`,
  `decay`, `half`. Do not use them as parameter, field, or local names. Use `out_val`,
  `share_mode`, `matched`, `policy_cfg`, `forget_f`, `half_n`.
- **String interpolation `"{var}"` can print literally** in some compiler paths. Prefer
  concatenation (`"x = " + str(x)`) or `_common.kn`'s `format_1dp` / `format_2dp` /
  `format_bytes`. `str(Float)` truncates toward zero — print `str(x * 10000.0)` as
  `x1e4` rather than lying with decimals.
- **`kain -c` trig probes lie.** `kain -c 'sin(1.57)' -r -t llvm` can return 0 from the
  repl harness stub. Trig works when compiled from a file. Always verify mathematical
  kernels via **file** compilation. (`-c` also cannot take argv.)
- **Bisecting a native crash:** copy the suspect module to a scratch file, append a
  `fn main()` that calls deeper into the path with prints, `kain build` it standalone,
  run it, then delete the scratch file. Keep scratch files out of `kain/core/` — an
  extra `main()` breaks the amalgamation.
- **Windows shells default to cp1252.** Every Python file write needs
  `encoding='utf-8'` or box-drawing characters corrupt the source. Kain source comments
  in this repo use box-drawing glyphs — keep the encoding explicit.
- **Don't run `jc` with bare default outputs inside `kain/core/`.** Default outputs
  (`juiced_model.gguf`, `step_schedule.json`) land in cwd; always pass `--out`.
  Source dirs hold `.kn` files only.

---

## 9. Using TurboKain as the Pattern Library

When you need a real pattern rather than a syntax guess, go read `../TurboKain/`:

- **Multi-call dispatch:** `../TurboKain/kain/core/dispatch.kn` owns the single `main()`
  and routes both `core <tool> [args...]` and direct `<tool>.exe [args...]`; each tool
  exports `pub fn <tool>_usage()` + `pub fn <tool>_main(args: Array<String>)`.
  `juicer.kn` follows the same shape (one `main()` in `dispatch.kn`, modules export
  `pub fn` entry points, `_common.kn` holds every shared primitive).
- **Arena + `converge` hot loops:** `../TurboKain/kain/core/boxcar_bank.kn`,
  `lane_sieve/`, `xvm_sandbox.kn` — the AVX2 `converge` spec lane and the `+32` rule
  in production.
- **kernel32 DMA:** `../TurboKain/kain/core/_common.kn` — the reference FFI block
  `juicer`'s `_common.kn` was derived from (`CreateFileA` / `GetFileSizeEx` /
  `SetFilePointerEx` / `ReadFile` / `WriteFile` / `CloseHandle`, `null_ptr()` =
  `int_to_ptr(0, "ptr<Void>")`, validity = `ptr_to_int(h) == -1`).
- **Amalgamation + ledgers + prove batteries:** `../TurboKain/README.md`,
  `../TurboKain/AGENTS.md`, `../TurboKain/catalog.tsv`, `../TurboKain/memory.tsv`.

Reference it; **do not fork it, do not edit it, do not build it.**

---

## 10. Ledgers — `memory.tsv` & `catalog.tsv`

Sibling repos (`../TurboKain/`, `../k_AI_n/`) maintain append-mostly ledgers so work
survives across agents without silent drift. `juicer.kn` carries the same two files.

### The Agent Workflow: CHECK FIRST, LOG AFTER

1. **CHECK FIRST** — read `memory.tsv` (and the relevant `catalog.tsv` rows) before
   touching code or planning. The ground truth about repo state is in the ledger, not
   in your assumptions.
2. **LOG AFTER** — every file change (created, edited, moved, deleted) gets a row,
   immediately, via `scripts/memlog.exe`. **No silent edits.**

### `memory.tsv` — the change log

Columns: `date ⇥ area ⇥ type ⇥ description ⇥ file` (tab-separated, append-only).

- `area`: `core` · `lens` · `slicer` · `fuse` · `scheduler` · `parser` · `prove` · `docs` · `scripts` · `build` · `repo`
- `type`: `add` · `update` · `fix` · `build` · `verify` · `vendor` · `scaffold` · `refactor`
- `description`: what changed **and why**, with the receipt if there is one. A fix
  without its evidence is a rumor.
- `file`: affected paths, comma-separated.

```cmd
rem build once:
cd scripts ^&^& ..\..\kain\.kain\bin\kain.exe build memlog.kn --target llvm -o memlog.exe ^&^& cd ..

rem then, from the repo root, for every change:
scripts\memlog.exe <area> <type> "what changed and why" "path/one,path/two"
```

`memlog` stamps the ISO date, seeds the header if the ledger is missing, and sanitizes
tabs/newlines so a field can never break the TSV. Run it **from the repo root**.

### `catalog.tsv` — the module & artifact ledger

Columns: `module ⇥ source ⇥ target ⇥ status ⇥ receipt ⇥ notes ⇥ updated`.
One row per module or artifact. `status` climbs `draft` → `builds` → `proven` → `fused`.

**`status=proven` with an empty receipt is a lie.** `jc prove` output is the receipt.
Update the row whenever a module is built, proven, or changes status.

---

## 11. Status

Working: 4 DSP lenses, GGUF v2/v3 parser, physical zero-copy slicer, LoRA fuse engine,
dynamic step-skip scheduler, 8/8 prove battery, `jc` CLI + multi-call shim, automated
build. Verified live against local Wan 2.1 14B Q4_K_M (40 blocks, 10.56 GB) and Flux 2
Klein 9B, with end-to-end ComfyUI I2V benchmarks driven by `scripts/benchmark_wan.py`.

Next: (a) inline Q4_K/Q8_0 dequantizer so the lenses read exact weights; (b) confirm
sliced-model load in the ComfyUI GGUF loader and patch whatever metadata key it needs;
(c) drive the carve ladder 40 → 24 → 20 → 18 → 16 blocks to land the model fully
resident in 6 GB VRAM.

The thesis is simple: **squeeze the bloat out of models. Pure juice, zero flab.**
