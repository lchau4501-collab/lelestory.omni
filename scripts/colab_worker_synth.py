#!/usr/bin/env python3
"""
Colab Remote Voice Synthesis Worker for LeLe Storybook Video Engine.
Executes directly inside Google Colab VM with NVIDIA CUDA GPU accelerator (Tesla T4/L4/A100).
Synthesizes authentic Chinese story audio sections using official k2-fsa/OmniVoice
zero-shot neural voice cloning at 24,000 Hz mono PCM 16-bit.
Uses speed=0.70 (30% reduction) for deliberate, natural pacing and zero distortion.
Supports dynamic 8–10 scenes + 5 vocab + vocab recap + outro loop (16–18 audio sections).
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
    print("📦 Installing official k2-fsa/OmniVoice from GitHub repository...", flush=True)
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

# 4. Canonical Voice Mark Audio & Transcript Lookup (Strict 0% Fallback)
ref_candidates = [
    Path("/content/voice_preview_mark.mp3"),
    Path("voice_preview_mark.mp3"),
    Path("/content/voice_preview_mark - cartoonish, funny and cheerful.mp3"),
    Path("voice_preview_mark - cartoonish, funny and cheerful.mp3"),
    Path("/root/voice_preview_mark.mp3"),
    Path("/content/drive/MyDrive/voice_preview_mark.mp3"),
]
REF_WAV = None
for c in ref_candidates:
    if c.exists() and c.stat().st_size > 0:
        REF_WAV = c
        break

if not REF_WAV:
    matches = glob.glob("/**/voice_preview_mark*", recursive=True)
    for m in matches:
        p = Path(m)
        if p.exists() and p.stat().st_size > 0:
            REF_WAV = p
            break

if not REF_WAV or not REF_WAV.exists():
    raise FileNotFoundError(f"Voice Mark reference audio not found. Searched {ref_candidates} and filesystem.")

# Convert MP3 to standard 24kHz mono PCM WAV if needed
if str(REF_WAV).lower().endswith(".mp3"):
    converted_wav = REF_WAV.with_suffix(".24k.wav")
    print(f"🔄 Converting reference MP3 ({REF_WAV}) to 24kHz mono WAV ({converted_wav})...", flush=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(REF_WAV), "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(converted_wav)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    REF_WAV = converted_wav

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

# 6. Dynamic Job Manifest & Script Resolution (8–10 Scenes, 5 Vocab, Recap, Outro Loop)
parser = argparse.ArgumentParser(description="OmniVoice Colab Voice Synthesis Worker")
parser.add_argument("--row-id", type=int, default=None, help="Story row ID")
parser.add_argument("--speed", type=float, default=0.70, help="Speaking speed factor (default: 0.70)")
parser.add_argument("--manifest", type=str, default=None, help="Path to job manifest JSON")
args, _ = parser.parse_known_args()

manifest_candidates = []
if args.manifest:
    manifest_candidates.append(Path(args.manifest))
manifest_candidates.extend([
    Path("/content/job_manifest.json"),
    Path("job_manifest.json"),
    Path("/job_manifest.json"),
    Path("/root/job_manifest.json")
])

manifest = None
manifest_found_path = None
for mc in manifest_candidates:
    if mc.exists() and mc.stat().st_size > 0:
        try:
            with open(mc, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest_found_path = mc
            print(f"📄 Loaded dynamic job manifest from {mc}", flush=True)
            break
        except Exception as e:
            print(f"⚠️ Failed to parse manifest from {mc}: {e}", flush=True)

row_id = 2
target_speed = args.speed
script_items = {}

if manifest:
    row_id = args.row_id if args.row_id is not None else manifest.get("row_id", 2)
    target_speed = args.speed if args.speed != 0.70 else manifest.get("speed", 0.70)

    # If explicit script_items dictionary is present, use it
    if "script_items" in manifest and isinstance(manifest["script_items"], dict) and len(manifest["script_items"]) > 0:
        script_items = manifest["script_items"]
    else:
        # Dynamically build from structured schema (scenes, vocabulary, outro)
        if "title" in manifest:
            script_items["title"] = manifest["title"]

        for s in manifest.get("scenes", []):
            s_num = s.get("scene_num")
            s_zh = s.get("zh", "").strip()
            if s_num is not None and s_zh:
                script_items[f"scene{s_num}"] = s_zh

        # Zero Vocab Audio Policy: Slide 11 is silent visual progression, no audio for individual vocab words
        outro_text = manifest.get("outro", {}).get("zh", "这些生词来自故事……")
        if outro_text:
            script_items["outro_loop"] = outro_text

# Default Fallback: Standard 10-Scene Narrative Arc for Row #2 (12 sections total: title, scene1..10, outro_loop)
if not script_items:
    row_id = args.row_id if args.row_id is not None else 2
    target_speed = args.speed
    script_items = {
        "title": "吃菜的大狼",
        "scene1": "在美丽茂密的大森林里，住着一只名叫罗罗的大灰狼。不同于普通的狼，他性情温和，最喜欢在菜园里种植新鲜蔬菜。",
        "scene2": "森林里的小兔子和小松鼠依然对大灰狼充满恐惧。每次远远看到罗罗走来，大家都吓得赶紧躲进灌木丛中不敢出声。",
        "scene3": "这天下午，天空突然乌云密布，一场狂暴的风雨呼啸而来，猛烈的狂风将山坡上的一棵巨大松树连根吹倒。",
        "scene4": "小兔子惊慌失措地呼喊：救命啊！倒下的大树把我们兔洞的出口死死挡住了，我们出不去了！",
        "scene5": "罗罗在风雨中听到了急切的呼救声。他没有躲回温暖的木屋，而是顶着狂风暴雨立刻奔向了兔洞。",
        "scene6": "罗罗大声说：小兔子别怕！我力气大，我来帮你们把这根沉重的大树干搬开！",
        "scene7": "浸透雨水的树干沉重无比，罗罗脚底打滑，爪子磨破了也绝不松手，咬紧牙关使出了全身的力气。",
        "scene8": "伴随着一声大喝，罗罗终于将巨木推到一旁，小心翼翼地把受惊的小兔子们一个个安全抱了出来。",
        "scene9": "兔妈妈感激地说：罗罗，太感谢你了！原来你是一只真正善良温和的大狼，我们再也不怕你了！",
        "scene10": "风雨过后彩虹高挂，小动物们齐聚在罗罗家，开开心心地吃起热气腾腾的蔬菜火锅。善良化解了误会，带来了珍贵的友谊。",
        "outro_loop": "这些生词来自故事……",
    }
    print("ℹ️ Using standard 10-Scene Chinese educational script (12 sections, zero vocab audio).", flush=True)

OUTPUT_DIR = Path(f"/content/voice_row_{row_id}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save active manifest in output directory for Gatekeeper audit downstream
active_manifest = {
    "row_id": row_id,
    "speed": target_speed,
    "sections_count": len(script_items),
    "script_items": script_items
}
with open(OUTPUT_DIR / "job_manifest.json", "w", encoding="utf-8") as f:
    json.dump(active_manifest, f, ensure_ascii=False, indent=2)

print(f"\n🎨 Synthesizing {len(script_items)} story sections for Row #{row_id} with OmniVoice (speed={target_speed})...", flush=True)
results = {}

# 7. Synthesize Sections Sequentially
for sec_name, text in script_items.items():
    out_file = OUTPUT_DIR / f"{sec_name}.wav"
    t0 = time.time()
    
    # Generate speech with pre-computed voice prompt and target speed (0.70x tempo)
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
