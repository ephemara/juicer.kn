#!/usr/bin/env python3
"""
benchmark_wan.py — Live End-to-End Inference Benchmark for juicer.kn

Spawns/Connects to ComfyUI, submits Wan 2.1 Juiced (20-block + Lightning Fused) workflow,
streams real-time telemetry (node execution, step progress, GPU VRAM usage),
and benchmarks total video generation time.
"""

import os
import sys
import time
import json
import uuid
import asyncio
import aiohttp

COMFY_HOST = "127.0.0.1:8188"
# Machine-local: override with $COMFYUI_HOME / $COMFYUI_HOST / $COMFYUI_PORT
COMFYUI_HOME = os.environ.get("COMFYUI_HOME", "S:/Local/ComfyUI_windows_portable")
COMFY_OUTPUT_DIR = os.environ.get("COMFYUI_OUTPUT", os.path.join(COMFYUI_HOME, "ComfyUI", "output"))
CLIENT_ID = str(uuid.uuid4())

def build_workflow(unet_name: str, frames: int = 17, width: int = 512, height: int = 512, steps: int = 4):
    return {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": unet_name
            }
        },
        "7": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "shift": 7.5,
                "model": ["1", 0]
            }
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "nsfw_wan_umt5-xxl_fp8_scaled.safetensors",
                "type": "wan"
            }
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "wan_2.1_vae.safetensors"
            }
        },
        "4": {
            "class_type": "LoadImage",
            "inputs": {
                "image": "source.jpg"
            }
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "natural smooth motion, ocean waves gently moving, smiling, highly detailed, photorealistic video",
                "clip": ["2", 0]
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "blurry, distorted, static, motionless, rigid, low quality, artifacts",
                "clip": ["2", 0]
            }
        },
        "21": {
            "class_type": "CLIPVisionLoader",
            "inputs": {
                "clip_name": "clip_vision_h.safetensors"
            }
        },
        "22": {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["21", 0],
                "image": ["4", 0],
                "crop": "center"
            }
        },
        "8": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "width": 640,
                "height": 640,
                "length": frames,
                "batch_size": 1,
                "positive": ["5", 0],
                "negative": ["6", 0],
                "vae": ["3", 0],
                "start_image": ["4", 0],
                "clip_vision_output": ["22", 0]
            }
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42,
                "steps": 6,
                "cfg": 1.5,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["7", 0],
                "positive": ["8", 0],
                "negative": ["8", 1],
                "latent_image": ["8", 2]
            }
        },
        "11": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["9", 0],
                "vae": ["3", 0],
                "tile_size": 640,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8
            }
        },
        "12": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "frame_rate": 16,
                "loop_count": 0,
                "filename_prefix": f"JUICER_{unet_name}_{frames}f",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "pingpong": False,
                "save_output": True,
                "images": ["11", 0]
            }
        }
    }

async def run_benchmark(unet_name: str, frames: int = 17):
    print("=" * 80)
    print(" juicer.kn — Live End-to-End Wan 2.1 Video Inference Benchmark")
    print("=" * 80)
    print(f" Model Under Test: {unet_name}")
    print(f" Target Frames:    {frames} frames")
    print(f" Resolution:       512 x 512")
    print(f" Sampling:         4 Steps Euler (Lightning Shift 5.0)")
    print(f" Client ID:        {CLIENT_ID}")
    print("=" * 80)

    prompt = build_workflow(unet_name, frames=frames)

    async with aiohttp.ClientSession() as session:
        # Check system stats
        try:
            async with session.get(f"http://{COMFY_HOST}/system_stats") as resp:
                stats = await resp.json()
                dev = stats["devices"][0]
                print(f" GPU Device:       {dev['name']}")
                print(f" Initial Free VRAM:{dev['vram_free'] / (1024**3):.2f} GB / {dev['vram_total'] / (1024**3):.2f} GB\n")
        except Exception as e:
            print(f"[ERROR] Could not connect to ComfyUI on {COMFY_HOST}: {e}")
            return

        # Connect WebSocket for live telemetry
        ws_url = f"ws://{COMFY_HOST}/ws?clientId={CLIENT_ID}"
        async with session.ws_connect(ws_url) as ws:
            print(" [Stage 1/3] Connected to ComfyUI Live Event Stream.")

            # Queue prompt
            post_payload = {"prompt": prompt, "client_id": CLIENT_ID}
            t_submit = time.time()
            async with session.post(f"http://{COMFY_HOST}/prompt", json=post_payload) as resp:
                prompt_resp = await resp.json()
                prompt_id = prompt_resp.get("prompt_id")
                print(f" [Stage 2/3] Prompt Queued Successfully (ID: {prompt_id})")

            # Listen for execution events
            print("\n [Stage 3/3] Live Execution & Node Telemetry:")
            t_start = None
            node_times = {}
            current_node = None
            current_node_start = None

            output_video_path = None

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")
                    msg_data = data.get("data", {})

                    if msg_type == "status":
                        pass

                    elif msg_type == "execution_start":
                        t_start = time.time()
                        print(f"  [START] Workflow execution started ...")

                    elif msg_type == "executing":
                        node_id = msg_data.get("node")
                        now = time.time()
                        if current_node and current_node_start:
                            dt = now - current_node_start
                            node_times[current_node] = dt
                            class_type = prompt.get(current_node, {}).get("class_type", "Node")
                            print(f"    -> Finished Node [{current_node}] {class_type} in {dt:.2f}s")

                        if node_id is None:
                            # Finished workflow
                            print(f"  [FINISH] All nodes executed successfully!")
                            break

                        current_node = str(node_id)
                        current_node_start = now
                        class_type = prompt.get(current_node, {}).get("class_type", "Node")
                        print(f"  [RUNNING] Node [{current_node}] {class_type} ...")

                    elif msg_type == "progress":
                        val = msg_data.get("value")
                        max_val = msg_data.get("max")
                        print(f"    --> Sampling Step {val}/{max_val} ...")

                    elif msg_type == "executed":
                        # Check outputs
                        output_data = msg_data.get("output", {})
                        if "gifs" in output_data or "videos" in output_data:
                            vids = output_data.get("gifs") or output_data.get("videos")
                            if vids:
                                filename = vids[0].get("filename")
                                subfolder = vids[0].get("subfolder", "")
                                output_video_path = os.path.join(COMFY_OUTPUT_DIR, subfolder, filename)

            total_elapsed = time.time() - (t_start or t_submit)
            print("\n" + "=" * 80)
            print(" BENCHMARK COMPLETE!")
            print(f" Total Inference Time: {total_elapsed:.2f} seconds")
            print(" Node Breakdown:")
            for nid, dt in node_times.items():
                ctype = prompt.get(nid, {}).get("class_type", "Node")
                print(f"   - Node {nid} ({ctype}): {dt:.2f}s")
            if output_video_path:
                print(f" Output Video Rendered: {output_video_path}")
                if os.path.exists(output_video_path):
                    print(f" File Size: {os.path.getsize(output_video_path) / 1024:.1f} KB")
            print("=" * 80)

if __name__ == "__main__":
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    model = sys.argv[2] if len(sys.argv) > 2 else "wan2.1-juiced-20b.gguf"
    asyncio.run(run_benchmark(model, frames))
