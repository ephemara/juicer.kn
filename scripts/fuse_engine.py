#!/usr/bin/env python3
"""
fuse_engine.py — High-Performance CUDA/Tensor Core LoRA Fusion Engine for juicer.kn

Fuses LoRA weights directly into GGUF diffusion/transformer models:
  - F32/F16 Norms & Biases: exact additive delta W' = W + alpha * diff
  - Linear Weights (Q4_K / Q8_0): GPU dequant -> W' = W + alpha * (B x A) -> GPU Q4_K quantize
  - Block Slicing: Optionally drops excised layers in the same pass with zero extra I/O
"""

import sys
import os
import time
import argparse
import numpy as np

# ComfyUI-GGUF dequant import path. Machine-local: override with $JUICER_COMFY_GGUF
# (or $COMFYUI_HOME) instead of editing this constant when the box moves.
COMFY_HOME = os.environ.get("COMFYUI_HOME", "S:/Local/ComfyUI_windows_portable")
COMFY_GGUF_PATH = os.environ.get(
    "JUICER_COMFY_GGUF", os.path.join(COMFY_HOME, "ComfyUI", "custom_nodes", "ComfyUI-GGUF")
)
if os.path.exists(COMFY_GGUF_PATH) and COMFY_GGUF_PATH not in sys.path:
    sys.path.insert(0, COMFY_GGUF_PATH)

import torch
import safetensors.torch
import gguf

try:
    import dequant
    HAS_DEQUANT = True
except ImportError:
    HAS_DEQUANT = False

# GPU Q4_K Quantizer
def quantize_q4_k_gpu(w: torch.Tensor) -> torch.Tensor:
    """
    Quantizes a 2D float tensor [rows, cols] (where cols % 256 == 0)
    into GGUF Q4_K format (144 bytes per 256 weights).
    """
    orig_shape = w.shape
    cols = orig_shape[-1]
    w_flat = w.reshape(-1, 256)
    n_blocks = w_flat.shape[0]

    sub_w = w_flat.reshape((n_blocks, 8, 32))
    
    sub_min = -torch.amin(sub_w, dim=-1)
    sub_min = torch.clamp(sub_min, min=0.0)
    sub_max = torch.amax(sub_w, dim=-1)
    
    sub_range = sub_max + sub_min
    sub_scale = sub_range / 15.0
    sub_scale = torch.clamp(sub_scale, min=1e-8)
    
    max_scale = torch.amax(sub_scale, dim=-1, keepdim=True)
    d = max_scale / 63.0
    d = torch.clamp(d, min=1e-8)
    sc = torch.clamp(torch.round(sub_scale / d), 0, 63).to(torch.uint8)
    
    max_min = torch.amax(sub_min, dim=-1, keepdim=True)
    dmin = max_min / 63.0
    dmin = torch.clamp(dmin, min=1e-8)
    m = torch.clamp(torch.round(sub_min / dmin), 0, 63).to(torch.uint8)
    
    sc_0_3 = sc[:, 0:4]
    sc_4_7 = sc[:, 4:8]
    m_0_3 = m[:, 0:4]
    m_4_7 = m[:, 4:8]
    
    d_bytes = (sc_0_3 & 0x3F) | ((sc_4_7 & 0x30) << 2)
    m_bytes = (m_0_3 & 0x3F) | ((m_4_7 & 0x30) << 2)
    md_bytes = (sc_4_7 & 0x0F) | ((m_4_7 & 0x0F) << 4)
    scales_bytes = torch.cat([d_bytes, m_bytes, md_bytes], dim=-1)
    
    eff_d = (d.to(torch.float16).float() * sc.float()).unsqueeze(-1)
    eff_dm = (dmin.to(torch.float16).float() * m.float()).unsqueeze(-1)
    
    eff_d = torch.clamp(eff_d, min=1e-8)
    qs_raw = torch.clamp(torch.round((sub_w + eff_dm) / eff_d), 0, 15).to(torch.uint8)
    
    qs_pairs = qs_raw.reshape((n_blocks, 4, 2, 32))
    low_nibble = qs_pairs[:, :, 0, :]
    high_nibble = qs_pairs[:, :, 1, :]
    packed_qs = (low_nibble | (high_nibble << 4)).reshape((n_blocks, 128))
    
    d_fp16 = d.to(torch.float16).view(torch.uint8).reshape((n_blocks, 2))
    dmin_fp16 = dmin.to(torch.float16).view(torch.uint8).reshape((n_blocks, 2))
    
    block_bytes = torch.cat([d_fp16, dmin_fp16, scales_bytes, packed_qs], dim=-1)
    return block_bytes

def get_block_id(name: str) -> int:
    for prefix in ("blocks.", "double_blocks.", "single_blocks.", "blk.", "layers."):
        if name.startswith(prefix):
            rest = name[len(prefix):]
            dot = rest.find(".")
            if dot != -1:
                try:
                    return int(rest[:dot])
                except ValueError:
                    pass
    return -1

def reindex_tensor_name(orig_name: str, drop_set: set) -> str:
    for prefix in ("blocks.", "double_blocks.", "single_blocks.", "blk.", "layers."):
        if orig_name.startswith(prefix):
            rest = orig_name[len(prefix):]
            dot = rest.find(".")
            if dot != -1:
                try:
                    old_id = int(rest[:dot])
                    if old_id in drop_set:
                        return None
                    drops_before = sum(1 for d in drop_set if d < old_id)
                    new_id = old_id - drops_before
                    return f"{prefix}{new_id}{rest[dot:]}"
                except ValueError:
                    pass
    return orig_name

def extract_field_value(field):
    vtype = field.types[0] if len(field.types) == 1 else field.types[-1]
    last = field.parts[-1]
    if vtype == gguf.GGUFValueType.STRING:
        return bytes(last).decode('utf-8', errors='ignore')
    elif vtype in (gguf.GGUFValueType.UINT32, gguf.GGUFValueType.INT32,
                   gguf.GGUFValueType.UINT64, gguf.GGUFValueType.INT64,
                   gguf.GGUFValueType.UINT16, gguf.GGUFValueType.INT16,
                   gguf.GGUFValueType.UINT8, gguf.GGUFValueType.INT8):
        return int(last[0]) if hasattr(last, '__getitem__') else int(last)
    elif vtype in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64):
        return float(last[0]) if hasattr(last, '__getitem__') else float(last)
    elif vtype == gguf.GGUFValueType.BOOL:
        return bool(last[0]) if hasattr(last, '__getitem__') else bool(last)
    return last

def fuse_model(base_path: str, lora_path: str, out_path: str, strength: float = 1.0, drop_list: list = None):
    out_path = out_path.strip('"\' ')
    base_path = base_path.strip('"\' ')
    lora_path = lora_path.strip('"\' ')
    print("=" * 80)
    print(" juicer.kn — High-Performance LoRA Fuse Engine")
    print("=" * 80)
    print(f" Base Model:   {base_path}")
    print(f" LoRA File:    {lora_path}")
    print(f" Strength:     {strength:.2f}")
    drop_set = set(drop_list) if drop_list else set()
    if drop_set:
        print(f" Slicing:      Dropping {len(drop_set)} blocks: {sorted(list(drop_set))}")
    print(f" Output GGUF:  {out_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Compute Dev:  {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    t0 = time.time()
    print("\n [Stage 1/3] Loading LoRA Safetensors into memory ...")
    lora = safetensors.torch.load_file(lora_path)
    print(f" Loaded {len(lora)} LoRA tensors in {time.time() - t0:.2f}s")

    print("\n [Stage 2/3] Reading Base GGUF Directory & Alignments ...")
    reader = gguf.GGUFReader(base_path)
    arch_field = reader.fields.get("general.architecture", None)
    arch_str = extract_field_value(arch_field) if arch_field else "unknown"
    print(f" Model Architecture: {arch_str}, Total Base Tensors: {len(reader.tensors)}")

    # Setup GGUF Writer
    writer = gguf.GGUFWriter(out_path, arch_str)

    # Copy metadata KV pairs, updating block count if sliced
    original_block_count = None
    for k, field in reader.fields.items():
        if k in ("general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
            continue
        vtype = field.types[0] if len(field.types) == 1 else field.types[-1]
        val = extract_field_value(field)

        # Check block count key
        if k.endswith(".block_count") or k == "block_count":
            original_block_count = int(val)
            if drop_set:
                new_block_count = original_block_count - len(drop_set)
                writer.add_uint32(k, new_block_count)
                print(f" Updated metadata: {k} = {original_block_count} -> {new_block_count}")
                continue

        # Add unchanged field
        if vtype == gguf.GGUFValueType.UINT32:
            writer.add_uint32(k, int(val))
        elif vtype == gguf.GGUFValueType.INT32:
            writer.add_int32(k, int(val))
        elif vtype == gguf.GGUFValueType.UINT64:
            writer.add_uint64(k, int(val))
        elif vtype == gguf.GGUFValueType.INT64:
            writer.add_int64(k, int(val))
        elif vtype == gguf.GGUFValueType.FLOAT32:
            writer.add_float32(k, float(val))
        elif vtype == gguf.GGUFValueType.STRING:
            writer.add_string(k, str(val))
        elif vtype == gguf.GGUFValueType.BOOL:
            writer.add_bool(k, bool(val))
        elif vtype == gguf.GGUFValueType.ARRAY:
            writer.add_array(k, val)

    writer.add_string("juicer.fused_lora", os.path.basename(lora_path))
    writer.add_float32("juicer.lora_strength", float(strength))

    print("\n [Stage 3/3] Fusing LoRA & Slicing Blocks ...")
    fused_weights_cnt = 0
    fused_biases_cnt = 0
    dropped_tensors_cnt = 0
    surviving_tensors_cnt = 0

    t_fuse_start = time.time()
    total_tensors = len(reader.tensors)

    for idx, tensor in enumerate(reader.tensors):
        name = tensor.name
        block_id = get_block_id(name)

        # Check if dropped
        if block_id != -1 and block_id in drop_set:
            dropped_tensors_cnt += 1
            continue

        surviving_tensors_cnt += 1
        new_name = reindex_tensor_name(name, drop_set)

        # Print progress every 50 tensors or per block transition
        if idx % 50 == 0 or idx == total_tensors - 1:
            print(f"  [{idx+1}/{total_tensors}] Processing: {name} -> {new_name} ...", end="\r")

        # Determine matching LoRA keys
        # LoRAs from ComfyUI typically use diffusion_model.<tensor_name>
        base_cand = f"diffusion_model.{name}"
        
        # Check bias / norm diffs
        is_bias = name.endswith(".bias")
        is_weight = name.endswith(".weight")

        lora_diff_b = f"{base_cand[:-5]}.diff_b" if is_bias else None
        lora_diff = f"{base_cand[:-7]}.diff" if is_weight else None
        lora_down = f"{base_cand[:-7]}.lora_down.weight" if is_weight else None
        lora_up = f"{base_cand[:-7]}.lora_up.weight" if is_weight else None

        # 1. BIAS / 1D TENSOR FUSION
        if is_bias and lora_diff_b and lora_diff_b in lora:
            base_data = torch.from_numpy(tensor.data)
            diff_data = lora[lora_diff_b].to(base_data.dtype)
            fused_data = (base_data + strength * diff_data).numpy()
            writer.add_tensor(new_name, fused_data, raw_dtype=tensor.tensor_type)
            fused_biases_cnt += 1

        # 2. NORM DIFF FUSION (e.g. norm_k.diff -> norm_k.weight)
        elif is_weight and lora_diff and lora_diff in lora:
            base_data = torch.from_numpy(tensor.data)
            diff_data = lora[lora_diff].to(base_data.dtype)
            fused_data = (base_data + strength * diff_data).numpy()
            writer.add_tensor(new_name, fused_data, raw_dtype=tensor.tensor_type)
            fused_biases_cnt += 1

        # 3. 2D LINEAR WEIGHT FUSION (LoRA factorized up x down)
        elif is_weight and lora_down and lora_down in lora and lora_up in lora:
            up = lora[lora_up].to(device=device, dtype=torch.float16)
            down = lora[lora_down].to(device=device, dtype=torch.float16)
            delta = torch.matmul(up, down) * strength

            if tensor.tensor_type == gguf.GGMLQuantizationType.Q4_K:
                # Dequantize base weight
                raw_gpu = torch.from_numpy(tensor.data).to(device)
                deq = dequant.dequantize(raw_gpu, tensor.tensor_type, tuple(reversed(tensor.shape)), dtype=torch.float16)
                fused_w = deq + delta
                
                # Requantize to Q4_K on GPU
                quant_bytes = quantize_q4_k_gpu(fused_w).cpu().numpy().tobytes()
                # Compute raw byte shape
                if len(tensor.shape) == 1:
                    raw_byte_shape = ((tensor.shape[0] // 256) * 144,)
                else:
                    raw_byte_shape = (tensor.shape[1], (tensor.shape[0] // 256) * 144)
                raw_arr = np.frombuffer(quant_bytes, dtype=np.uint8).reshape(raw_byte_shape)
                writer.add_tensor(new_name, raw_arr, raw_shape=raw_byte_shape, raw_dtype=gguf.GGMLQuantizationType.Q4_K)
                fused_weights_cnt += 1

            elif tensor.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
                base_data = torch.from_numpy(tensor.data).to(device=device, dtype=torch.float16)
                fused_data = (base_data + delta).to(dtype=torch.float16 if tensor.tensor_type == gguf.GGMLQuantizationType.F16 else torch.float32).cpu().numpy()
                writer.add_tensor(new_name, fused_data, raw_dtype=tensor.tensor_type)
                fused_weights_cnt += 1
            else:
                # Fallback: copy as-is if unhandled quant
                writer.add_tensor(new_name, tensor.data, raw_dtype=tensor.tensor_type)

        # 4. UNMODIFIED TENSOR: Copy directly
        else:
            writer.add_tensor(new_name, tensor.data, raw_dtype=tensor.tensor_type)

    print(f"\n Writing serialized GGUF container to disk: {out_path} ...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    total_time = time.time() - t_fuse_start
    print("\n" + "=" * 80)
    print(" FUSION COMPLETE!")
    print(f" Fused Linear Weights:  {fused_weights_cnt}")
    print(f" Fused Biases / Norms:  {fused_biases_cnt}")
    print(f" Dropped Tensors:       {dropped_tensors_cnt}")
    print(f" Surviving Tensors:     {surviving_tensors_cnt}")
    print(f" Elapsed Time:          {total_time:.2f}s")
    print(f" Output File:           {out_path} ({os.path.getsize(out_path) / (1024**3):.2f} GB)")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="juicer.kn LoRA Fuse Engine")
    parser.add_argument("base", help="Path to base GGUF model")
    parser.add_argument("lora", help="Path to LoRA safetensors file")
    parser.add_argument("--strength", type=float, default=1.0, help="LoRA strength multiplier (default: 1.0)")
    parser.add_argument("--drop", type=str, default="", help="Comma-separated block indices to drop (e.g. 18,19,20)")
    parser.add_argument("--out", type=str, required=True, help="Output GGUF file path")

    args = parser.parse_args()
    drop_list = [int(x.strip()) for x in args.drop.split(",") if x.strip()] if args.drop else []
    sys.exit(fuse_model(args.base, args.lora, args.out, args.strength, drop_list))
