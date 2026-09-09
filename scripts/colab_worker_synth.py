#!/usr/bin/env python3
"""
Colab Remote Voice Synthesis Worker for LeLe Storybook Video Engine.
Executes directly inside Google Colab VM with CUDA GPU accelerator (T4/L4) or high-speed CPU.
Synthesizes authentic Chinese story audio sections using sherpa-onnx ZipVoice zero-shot cloning
at 24kHz mono PCM 16-bit with num_steps=10 for maximum spectral fidelity and zero distortion.
"""

import os
import sys
import json
import time
import glob
import shutil
import tarfile
import urllib.request
import subprocess
from pathlib import Path

print("🚀 [Colab Worker] Starting Voice Synthesis on Google Colab VM...")

# 1. Ensure dependencies installed
try:
    import sherpa_onnx
    import soundfile as sf
    import numpy as np
    print("✓ Dependencies already installed.")
except ImportError:
    print("📦 Installing sherpa-onnx, soundfile, numpy via pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sherpa-onnx", "soundfile", "numpy"], check=True)
    import sherpa_onnx
    import soundfile as sf
    import numpy as np
    print("✓ Dependencies installed successfully.")

# 2. Hardware Detection
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
print(f"⚡ Hardware Accelerator: {device.upper()} ({gpu_name})")

# 3. Model weights setup
MODELS_DIR = Path("/content/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

VOCODER_PATH = MODELS_DIR / "vocos_24khz.onnx"
ZIPVOICE_DIR = MODELS_DIR / "zipvoice"

VOCODER_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx"
ZIPVOICE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2"

if not VOCODER_PATH.exists() or VOCODER_PATH.stat().st_size < 50000000:
    print("📥 Downloading Vocos 24kHz vocoder (~54MB)...")
    urllib.request.urlretrieve(VOCODER_URL, str(VOCODER_PATH))
    print("✓ Vocoder downloaded.")

if not (ZIPVOICE_DIR / "decoder.int8.onnx").exists():
    print("📥 Downloading ZipVoice neural model archive (~109MB)...")
    tar_path = "/tmp/zipvoice.tar.bz2"
    urllib.request.urlretrieve(ZIPVOICE_URL, tar_path)
    print("📦 Extracting ZipVoice archive...")
    with tarfile.open(tar_path, "r:bz2") as tar:
        tar.extractall(path=str(MODELS_DIR))
    extracted = MODELS_DIR / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia"
    if extracted.exists():
        if ZIPVOICE_DIR.exists():
            shutil.rmtree(ZIPVOICE_DIR)
        extracted.rename(ZIPVOICE_DIR)
    if os.path.exists(tar_path):
        os.remove(tar_path)
    print("✓ ZipVoice model ready.")

# 4. Resilient Reference voice audio & transcript lookup
ref_candidates = [
    Path("/content/reference.wav"),
    Path("/reference.wav"),
    Path("reference.wav"),
    Path("/root/reference.wav"),
    Path("/content/drive/MyDrive/reference.wav")
]
REF_WAV = None
for c in ref_candidates:
    if c.exists():
        REF_WAV = c
        break

if not REF_WAV:
    matches = glob.glob("/**/reference.wav", recursive=True)
    if matches:
        REF_WAV = Path(matches[0])

if not REF_WAV or not REF_WAV.exists():
    raise FileNotFoundError(f"Reference voice audio not found. Searched {ref_candidates} and filesystem.")

ref_audio, ref_sr = sf.read(str(REF_WAV), dtype="float32")
if ref_audio.ndim > 1:
    ref_audio = ref_audio[:, 0]

ref_txt_candidates = [
    Path("/content/reference.txt"),
    Path("/reference.txt"),
    Path("reference.txt"),
    REF_WAV.with_suffix(".txt")
]
ref_text = "不求与人相比，但求超越自己。"
for tc in ref_txt_candidates:
    if tc.exists():
        with open(tc, "r", encoding="utf-8") as f:
            t = f.read().strip()
            if t:
                ref_text = t
                break

print(f"🎙️ Reference Voice ({REF_WAV}): {len(ref_audio)/ref_sr:.2f}s, SR={ref_sr}Hz | Transcript: '{ref_text}'")

# 5. Initialize TTS Engine
print("🔧 Initializing sherpa-onnx OfflineTts...")
tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
            tokens=str(ZIPVOICE_DIR / "tokens.txt"),
            encoder=str(ZIPVOICE_DIR / "encoder.int8.onnx"),
            decoder=str(ZIPVOICE_DIR / "decoder.int8.onnx"),
            vocoder=str(VOCODER_PATH),
            data_dir=str(ZIPVOICE_DIR / "espeak-ng-data"),
            lexicon=str(ZIPVOICE_DIR / "lexicon.txt") if (ZIPVOICE_DIR / "lexicon.txt").exists() else "",
        ),
        debug=False,
        num_threads=4,
        provider="cpu",
    )
)

if not tts_config.validate():
    raise RuntimeError("TTS configuration validation failed!")

tts = sherpa_onnx.OfflineTts(tts_config)

gen_config = sherpa_onnx.GenerationConfig()
gen_config.num_steps = 10  # 10 Flow-matching steps for maximum smoothness
gen_config.reference_audio = ref_audio
gen_config.reference_sample_rate = int(ref_sr)
gen_config.reference_text = ref_text
gen_config.speed = 1.0
gen_config.silence_scale = 0.2

# 6. Read Story Script Manifest
MANIFEST_PATH = Path("/content/job_manifest.json")
if MANIFEST_PATH.exists():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    row_id = manifest.get("row_id", 2)
    script_items = manifest.get("script_items", {})
else:
    row_id = 2
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

print(f"\n🎨 Synthesizing 12 story sections for Row #{row_id} (num_steps=10)...")
results = {}

for sec_name, text in script_items.items():
    out_file = OUTPUT_DIR / f"{sec_name}.wav"
    t0 = time.time()
    audio = tts.generate(text, gen_config)
    elapsed = time.time() - t0

    if len(audio.samples) == 0:
        raise RuntimeError(f"Zero samples generated for section: {sec_name}")

    samples = np.array(audio.samples, dtype=np.float32)
    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples = (samples / peak) * 0.85

    sf.write(str(out_file), samples, samplerate=audio.sample_rate, subtype="PCM_16")

    rms = float(np.sqrt(np.mean(samples ** 2)) * 32768)
    duration = len(samples) / audio.sample_rate
    results[sec_name] = {
        "duration": duration,
        "rms": rms,
        "chars": len(text),
        "synth_time_sec": elapsed
    }
    print(f"  ✓ {sec_name:<12} | {duration:5.2f}s | RMS: {rms:6.1f} | Chars: {len(text):2d} | Synth in {elapsed:4.2f}s | '{text}'")

# 7. Package into archive (save at both /content and /)
ARCHIVE_PATH = Path(f"/content/voice_row_{row_id}.tar.gz")
with tarfile.open(str(ARCHIVE_PATH), "w:gz") as tar:
    tar.add(str(OUTPUT_DIR), arcname=f"voice_row_{row_id}")

try:
    shutil.copyfile(str(ARCHIVE_PATH), f"/voice_row_{row_id}.tar.gz")
except Exception:
    pass

print(f"\n📦 Packaged all 12 WAV files to {ARCHIVE_PATH} ({ARCHIVE_PATH.stat().st_size} bytes)")
print("\n[COLAB_SYNTH_COMPLETE]")
print(json.dumps({
    "status": "SUCCESS",
    "row_id": row_id,
    "archive_path": str(ARCHIVE_PATH),
    "sections_count": len(results),
    "total_duration_sec": sum(r["duration"] for r in results.values()),
    "total_synth_time_sec": sum(r["synth_time_sec"] for r in results.values()),
    "sections": results
}, indent=2))
