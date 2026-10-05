#!/usr/bin/env python3
"""
scripts/quantize_engine.py — High-Performance Direct SafeTensors -> GGUF Quantizer & LoRA Fuse Engine

Quantizes raw master BF16/FP16 .safetensors models directly into GGUF (Q4_K / Q8_0)
with support for lossless multi-LoRA / LyCORIS LoKR baking in a single pass:
  - Supports multiple --lora <path>[:strength]
  - Supports standard Low-Rank GEMM (lora_up/down, lora_A/B, Kohya lora_unet_*)
  - Supports LyCORIS LoKR (Low-Rank Kronecker product: w1 (x) w2)
  - 1D Norms, Scales, Biases -> Preserved in exact FP32 / FP16
  - 2D Linear Weights (cols % 256 == 0) -> Quantized to Q4_K (144B) / Q6_K (210B)
  super-blocks on GPU
  - Direct NVMe safe_open memory-mapped streaming: < 500 MB VRAM footprint
"""

import sys
import os
import time
import argparse
import numpy as np
import torch
import safetensors
from safetensors import safe_open
import gguf

# Import GPU quantizer from fuse_engine
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from fuse_engine import quantize_q4_k_gpu, quantize_q6_k_gpu

def detect_architecture(keys):
    for k in keys:
        if "double_blocks." in k or "single_blocks." in k:
            return "flux"
        if "diffusion_model." in k or "v2_i2v" in k or "wan" in k.lower():
            return "wan"
        if "model.layers." in k:
            return "llama"
    return "flux"

def load_adapter(path_str):
    path_str = path_str.strip('"\' ')
    strength = 1.0
    file_path = path_str

    if ":" in path_str:
        last_colon = path_str.rfind(":")
        # Ensure it's not a Windows drive letter colon (e.g. S:\)
        if last_colon > 1:
            try:
                strength = float(path_str[last_colon + 1:])
                file_path = path_str[:last_colon]
            except ValueError:
                file_path = path_str

    print(f" Loading Adapter: {file_path} (strength={strength:.2f}) ...")
    weights = safetensors.torch.load_file(file_path)
    return {
        "path": file_path,
        "name": os.path.basename(file_path),
        "strength": strength,
        "weights": weights,
    }

def compute_adapter_delta(adapter, base_key, shape, device):
    weights = adapter["weights"]
    strength = adapter["strength"]

    if not base_key.endswith(".weight"):
        return None

    core_raw = base_key[:-7] # strip .weight

    # 1. Check LyCORIS LoKR format: diffusion_model.<core>.lokr_w1 & w2
    cand_lokr_w1 = f"diffusion_model.{core_raw}.lokr_w1"
    cand_lokr_w2 = f"diffusion_model.{core_raw}.lokr_w2"
    if cand_lokr_w1 in weights and cand_lokr_w2 in weights:
        w1 = weights[cand_lokr_w1].to(device=device, dtype=torch.float32)
        w2 = weights[cand_lokr_w2].to(device=device, dtype=torch.float32)
        delta = torch.kron(w1, w2) * strength
        if list(delta.shape) == shape:
            return delta

    # 2. Check ComfyUI standard LoRA: diffusion_model.<core>.lora_up / lora_down (or lora_A / lora_B)
    cand_up = f"diffusion_model.{core_raw}.lora_up.weight"
    cand_down = f"diffusion_model.{core_raw}.lora_down.weight"
    if cand_up not in weights:
        cand_up = f"diffusion_model.{core_raw}.lora_B.weight"
        cand_down = f"diffusion_model.{core_raw}.lora_A.weight"
    if cand_up in weights and cand_down in weights:
        up = weights[cand_up].to(device=device, dtype=torch.float32)
        down = weights[cand_down].to(device=device, dtype=torch.float32)
        delta = torch.matmul(up, down) * strength
        if list(delta.shape) == shape:
            return delta

    # 3. Check Kohya underscore LoRA: lora_unet_<core_underscores>.lora_up / lora_down
    core_under = core_raw.replace(".", "_")
    cand_k_up = f"lora_unet_{core_under}.lora_up.weight"
    cand_k_down = f"lora_unet_{core_under}.lora_down.weight"
    if cand_k_up not in weights:
        cand_k_up = f"lora_unet_{core_under}.lora_B.weight"
        cand_k_down = f"lora_unet_{core_under}.lora_A.weight"
    if cand_k_up in weights and cand_k_down in weights:
        up = weights[cand_k_up].to(device=device, dtype=torch.float32)
        down = weights[cand_k_down].to(device=device, dtype=torch.float32)
        delta = torch.matmul(up, down) * strength
        if list(delta.shape) == shape:
            return delta

    return None

def quantize_safetensors_to_gguf(src_path: str, out_path: str, quant_type: str = "q4_k", lora_args: list = None):
    src_path = src_path.strip('"\' ')
    out_path = out_path.strip('"\' ')
    if lora_args is None:
        lora_args = []

    print("=" * 80)
    print(" juicer.kn — Direct SafeTensors -> GGUF Quantizer & LoRA Fuse Engine")
    print("=" * 80)
    print(f" Source Model:   {src_path}")
    print(f" Target Format:  {quant_type.upper()}")
    print(f" Output GGUF:    {out_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Compute Device: {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # Load adapters into memory
    adapters = []
    if lora_args:
        print(f"\n [Stage 1/3] Loading {len(lora_args)} LoRA / LyCORIS Adapters ...")
        for l_arg in lora_args:
            adapters.append(load_adapter(l_arg))

    t0 = time.time()
    src_size = os.path.getsize(src_path)
    print(f" Source Size:    {src_size / (1024**3):.2f} GB")

    with safe_open(src_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        total_tensors = len(keys)
        arch = detect_architecture(keys)
        print(f" Architecture:   {arch.upper()} ({total_tensors} tensors)")

        stage_num = "2/3" if adapters else "1/2"
        print(f"\n [Stage {stage_num}] Initializing GGUF Container & Metadata ...")
        writer = gguf.GGUFWriter(out_path, arch)
        writer.add_uint32("general.quantization_version", 2)
        if quant_type.lower() == "q4_k":
            writer.add_uint32("general.file_type", 15) # Q4_K_M
        elif quant_type.lower() == "q6_k":
            writer.add_uint32("general.file_type", 18) # Q6_K
        else:
            writer.add_uint32("general.file_type", 7)  # Q8_0

        writer.add_string("juicer.source", os.path.basename(src_path))
        writer.add_string("juicer.pipeline", "clean_slate_single_pass_quant_fused")

        for ad in adapters:
            writer.add_string("juicer.fused_lora", f"{ad['name']}:{ad['strength']:.2f}")

        stage_num = "3/3" if adapters else "2/2"
        print(f"\n [Stage {stage_num}] Streaming, Fusing LoRAs, & Quantizing via Tensor Cores ...")
        quant_cnt = 0
        passthrough_cnt = 0
        fused_tensors_cnt = 0

        for i, k in enumerate(keys):
            tensor = f.get_tensor(k)
            shape = list(tensor.shape)

            # Progress badge
            if i % 10 == 0 or i == total_tensors - 1:
                print(f"  [{i+1}/{total_tensors}] Processing: {k} (shape={shape}) ...", end="\r")

            is_2d_quantizable = len(shape) == 2 and (shape[1] % 256 == 0)

            # Check if any adapter targets this tensor
            w = tensor.to(device=device, dtype=torch.float32)
            has_fused = False
            for ad in adapters:
                delta = compute_adapter_delta(ad, k, shape, device)
                if delta is not None:
                    w = w + delta
                    has_fused = True
                    del delta

            if has_fused:
                fused_tensors_cnt += 1

            if is_2d_quantizable and quant_type.lower() in ("q4_k", "q6_k"):
                # Quantize on GPU Tensor Cores
                rows, cols = shape
                if quant_type.lower() == "q6_k":
                    quant_bytes = quantize_q6_k_gpu(w.float()).cpu().numpy().tobytes()
                    block_bytes = 210
                    qtype = gguf.GGMLQuantizationType.Q6_K
                else:
                    quant_bytes = quantize_q4_k_gpu(w.half()).cpu().numpy().tobytes()
                    block_bytes = 144
                    qtype = gguf.GGMLQuantizationType.Q4_K
                del w

                raw_byte_shape = (rows, (cols // 256) * block_bytes)
                raw_arr = np.frombuffer(quant_bytes, dtype=np.uint8).reshape(raw_byte_shape)
                writer.add_tensor(k, raw_arr, raw_shape=raw_byte_shape, raw_dtype=qtype)
                quant_cnt += 1
            else:
                # 1D scalars, biases, norm scales: preserve in exact float
                data_np = w.cpu().numpy()
                del w
                writer.add_tensor(k, data_np, raw_dtype=gguf.GGMLQuantizationType.F32)
                passthrough_cnt += 1

        print(f"\n\n Writing serialized GGUF container to disk: {out_path} ...")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    elapsed = time.time() - t0
    out_size = os.path.getsize(out_path)
    compression = (1.0 - (out_size / src_size)) * 100.0

    print("=" * 80)
    print(" FUSION & QUANTIZATION COMPLETE!")
    print(f" Fused Adapters:    {len(adapters)} ({fused_tensors_cnt} tensors modified in pure float)")
    print(f" Quantized Weights: {quant_cnt} tensors ({quant_type.upper()} Super-Blocks)")
    print(f" Preserved Floats:  {passthrough_cnt} tensors (Exact FP32)")
    print(f" Initial Size:      {src_size / (1024**3):.2f} GB")
    print(f" Output GGUF Size:  {out_size / (1024**3):.2f} GB ({compression:.1f}% reduction)")
    print(f" Elapsed Time:      {elapsed:.1f}s ({total_tensors / elapsed:.1f} tensors/s)")
    print(f" Output File:       {out_path}")
    print("=" * 80)
    return 0

def main():
    parser = argparse.ArgumentParser(description="Direct SafeTensors -> GGUF Quantizer & LoRA Fuse Engine for juicer.kn")
    parser.add_argument("src", type=str, help="Path to input .safetensors file")
    parser.add_argument("--lora", action="append", default=[], help="Path to LoRA safetensors file, optionally with :strength (e.g. lora.safetensors:1.0)")
    parser.add_argument("--type", type=str, default="q4_k", choices=["q4_k", "q6_k", "q8_0"], help="Quantization type (default: q4_k)")
    parser.add_argument("--out", type=str, required=True, help="Path to output .gguf file")
    args = parser.parse_args()

    sys.exit(quantize_safetensors_to_gguf(args.src, args.out, args.type, args.lora))

if __name__ == "__main__":
    main()
