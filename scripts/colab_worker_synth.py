#!/usr/bin/env python3
"""
Colab Remote Voice Synthesis Worker for LeLe Storybook Video Engine.
Executes directly inside Google Colab VM with NVIDIA CUDA GPU accelerator (Tesla T4/L4/A100).
Synthesizes authentic Chinese story audio sections using official k2-fsa/OmniVoice
zero-shot neural voice cloning at 24,000 Hz mono PCM 16-bit.
Uses speed=0.85 (15% reduction) for deliberate, natural pacing and zero distortion.
"""

import os
import sys
import json
import time
import glob
import shutil
import tarfile
import argparse
import subprocess
from pathlib import Path

print("🚀 [Colab Worker] Starting Official k2-fsa/OmniVoice Voice Synthesis on Google Colab VM...", flush=True)

# 1. Ensure official k2-fsa/OmniVoice and soundfile are installed
try:
    import torch
    import omnivoice
    from omnivoice import OmniVoice
    import soundfile as sf
    import numpy as np
    print("✓ Official OmniVoice and dependencies already installed.", flush=True)
except ImportError:
    print("📦 Installing official k2-fsa/OmniVoice and soundfile via pip...", flush=True)
    cmd = [sys.executable, "-m", "pip", "install", "-q", "omnivoice", "soundfile"]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("⚠️ Standard PyPI install failed, attempting git repository install...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/k2-fsa/OmniVoice.git", "soundfile"],
            check=True
        )
    import torch
    import omnivoice
    from omnivoice import OmniVoice
    import soundfile as sf
    import numpy as np
    print("✓ Official OmniVoice installed successfully.", flush=True)

# 2. Hardware Enforcement (Strict CUDA GPU Check)
if not torch.cuda.is_available():
    raise RuntimeError(
        "❌ CUDA GPU is NOT available on this Colab VM! "
        "OmniVoice neural synthesis strictly requires NVIDIA GPU acceleration (Tesla T4/L4/A100). "
        "CPU execution is forbidden per policy."
    )

device = "cuda:0"
dtype = torch.float16
gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"⚡ Hardware Accelerator: {device} ({gpu_name}, {gpu_mem:.2f} GB VRAM) | Precision: {dtype}", flush=True)

# 3. Load official k2-fsa/OmniVoice Model
print("📥 Loading official k2-fsa/OmniVoice model from Hugging Face Hub (load_asr=False)...", flush=True)
t_model_start = time.time()
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=device,
    dtype=dtype,
    load_asr=False
)
model_load_time = time.time() - t_model_start
print(f"✓ OmniVoice model successfully loaded in {model_load_time:.2f}s.", flush=True)

# 4. Resilient Reference Voice Audio & Transcript Lookup
ref_candidates = [
    Path("/content/reference.wav"),
    Path("/reference.wav"),
    Path("reference.wav"),
    Path("/root/reference.wav"),
    Path("/content/drive/MyDrive/reference.wav")
]
REF_WAV = None
for c in ref_candidates:
    if c.exists() and c.stat().st_size > 0:
        REF_WAV = c
        break

if not REF_WAV:
    matches = glob.glob("/**/reference.wav", recursive=True)
    for m in matches:
        p = Path(m)
        if p.exists() and p.stat().st_size > 0:
            REF_WAV = p
            break

if not REF_WAV or not REF_WAV.exists():
    raise FileNotFoundError(f"Reference voice audio not found. Searched {ref_candidates} and filesystem.")

ref_txt_candidates = [
    Path("/content/reference.txt"),
    Path("/reference.txt"),
    Path("reference.txt"),
    Path("/root/reference.txt"),
    REF_WAV.with_suffix(".txt")
]
ref_text = "不求与人相比，但求超越自己。"
for tc in ref_txt_candidates:
    if tc.exists():
        try:
            with open(tc, "r", encoding="utf-8") as f:
                t = f.read().strip()
                if t:
                    ref_text = t
                    break
        except Exception:
            pass

# Verify reference audio loadable
ref_info = sf.info(str(REF_WAV))
print(f"🎙️ Reference Voice ({REF_WAV}): {ref_info.duration:.2f}s, SR={ref_info.samplerate}Hz, Ch={ref_info.channels} | Transcript: '{ref_text}'", flush=True)

# 5. Pre-compute Reusable Voice Clone Prompt (Prevents Redundant Feature Extraction)
print("🎯 Pre-computing reusable VoiceClonePrompt from reference audio and text...", flush=True)
t_prompt_start = time.time()
voice_prompt = model.create_voice_clone_prompt(
    ref_audio=str(REF_WAV),
    ref_text=ref_text,
    preprocess_prompt=True
)
prompt_compute_time = time.time() - t_prompt_start
print(f"✓ VoiceClonePrompt pre-computed in {prompt_compute_time:.2f}s.", flush=True)

# 6. Job Manifest & Target Script Configuration
parser = argparse.ArgumentParser(description="OmniVoice Colab Voice Synthesis Worker")
parser.add_argument("--row-id", type=int, default=None, help="Story row ID")
parser.add_argument("--speed", type=float, default=0.85, help="Speaking speed factor (default: 0.85)")
args, _ = parser.parse_known_args()

MANIFEST_PATH = Path("/content/job_manifest.json")
if MANIFEST_PATH.exists():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    row_id = args.row_id if args.row_id is not None else manifest.get("row_id", 2)
    script_items = manifest.get("script_items", {})
    target_speed = args.speed if args.speed != 0.85 else manifest.get("speed", 0.85)
else:
    row_id = args.row_id if args.row_id is not None else 2
    target_speed = args.speed
    script_items = {
        "title": "吃菜的大狼",
        "scene1": "深山里住着一只大灰狼，名叫罗罗。",
        "scene2": "森林里的小动物们都很怕他，一见到他就跑。",
        "scene3": "别害怕，我不吃肉，我只喜欢吃胡萝卜和白菜！",
        "scene4": "小兔子们放心地笑了，大家围着罗罗一起开心地吃蔬菜火锅。",
        "vocab_1": "大灰狼",
        "vocab_2": "蔬菜",
        "vocab_3": "胡萝卜",
        "vocab_4": "白菜",
        "vocab_5": "火锅",
        "vocab": "大灰狼 蔬菜 胡萝卜 白菜 火锅",
        "outro_loop": "这些生词来自故事……",
    }

OUTPUT_DIR = Path(f"/content/voice_row_{row_id}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n🎨 Synthesizing {len(script_items)} story sections for Row #{row_id} with OmniVoice (speed={target_speed})...", flush=True)
results = {}

# 7. Synthesize Sections Sequentially
for sec_name, text in script_items.items():
    out_file = OUTPUT_DIR / f"{sec_name}.wav"
    t0 = time.time()
    
    # Generate speech with pre-computed voice prompt and target speed
    audios = model.generate(
        text=text,
        voice_clone_prompt=voice_prompt,
        speed=target_speed,
        language="zh"
    )
    elapsed = time.time() - t0

    if not audios or len(audios) == 0 or len(audios[0]) == 0:
        raise RuntimeError(f"Zero samples generated for section: {sec_name}")

    samples = np.array(audios[0], dtype=np.float32)
    
    # Peak Normalization with 0.85 - 0.95 Safety Headroom (Target 0.90)
    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        target_headroom = 0.90
        samples = (samples / peak) * target_headroom
    else:
        raise RuntimeError(f"Synthesized audio for section {sec_name} contains pure silence.")

    # Export to standard 24,000 Hz mono 16-bit Little-Endian PCM WAV
    sf.write(str(out_file), samples, samplerate=model.sampling_rate, subtype="PCM_16")

    # Quality Metrics Calculation
    duration = len(samples) / float(model.sampling_rate)
    rms = float(np.sqrt(np.mean(samples ** 2)) * 32768.0)
    peak_int = int(np.max(np.abs(samples * 32768.0)))
    clipped_samples = int(np.sum(np.abs(samples * 32768.0) >= 32767))
    clipping_ratio = float(clipped_samples / len(samples))

    results[sec_name] = {
        "filename": f"{sec_name}.wav",
        "duration": round(duration, 3),
        "rms": round(rms, 2),
        "peak": peak_int,
        "clipping_ratio": clipping_ratio,
        "chars": len(text),
        "synth_time_sec": round(elapsed, 3),
        "speed": target_speed,
        "sample_rate": model.sampling_rate,
        "channels": 1,
        "bit_depth": 16
    }
    print(f"  ✓ {sec_name:<12} | {duration:5.2f}s | RMS: {rms:6.1f} | Peak: {peak_int:5d} | Synth in {elapsed:4.2f}s | '{text}'", flush=True)

    # Clean GPU memory between iterations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 8. Package into Archive (/content/voice_row_{row_id}.tar.gz and /voice_row_{row_id}.tar.gz)
ARCHIVE_PATH = Path(f"/content/voice_row_{row_id}.tar.gz")
with tarfile.open(str(ARCHIVE_PATH), "w:gz") as tar:
    tar.add(str(OUTPUT_DIR), arcname=f"voice_row_{row_id}")

try:
    shutil.copyfile(str(ARCHIVE_PATH), f"/voice_row_{row_id}.tar.gz")
except Exception as exc:
    print(f"⚠️ Notice: could not mirror archive to root: {exc}", flush=True)

print(f"\n📦 Packaged all {len(results)} WAV files to {ARCHIVE_PATH} ({ARCHIVE_PATH.stat().st_size} bytes)", flush=True)

# 9. Stdout Completion Contract
print("\n[COLAB_SYNTH_COMPLETE]", flush=True)
completion_payload = {
    "status": "SUCCESS",
    "row_id": row_id,
    "engine": "k2-fsa/OmniVoice",
    "device": device,
    "gpu_name": gpu_name,
    "archive_path": str(ARCHIVE_PATH),
    "sections_count": len(results),
    "speed": target_speed,
    "sample_rate": model.sampling_rate,
    "total_duration_sec": round(sum(r["duration"] for r in results.values()), 3),
    "total_synth_time_sec": round(sum(r["synth_time_sec"] for r in results.values()), 3),
    "sections": results
}
print(json.dumps(completion_payload, indent=2), flush=True)
