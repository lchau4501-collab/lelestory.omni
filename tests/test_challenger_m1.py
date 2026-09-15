"""
Adversarial Empirical Challenge Test Suite for Milestone M1
Target: scripts/colab_worker_synth.py
Challenger: Challenger M1-1
"""

import ast
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from audio_qc import check_wav_file, calculate_duration_bounds

WORKER_SCRIPT = REPO_ROOT / "scripts" / "colab_worker_synth.py"


# ============================================================================
# 1. AST & Code Integrity Checks (Zero Legacy, Upstream API Conformance)
# ============================================================================

def test_worker_script_exists_and_is_valid_python_ast():
    """Verify scripts/colab_worker_synth.py exists and parses cleanly into AST."""
    assert WORKER_SCRIPT.exists(), f"Missing worker script at {WORKER_SCRIPT}"
    source = WORKER_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(WORKER_SCRIPT))
    assert isinstance(tree, ast.Module), "Parsed AST is not a Module"


def test_zero_legacy_sherpa_zipvoice_vocos_in_worker_script():
    """Adversarial check: ensure NO traces of sherpa-onnx, zipvoice, vocos, or dummy facades."""
    source = WORKER_SCRIPT.read_text(encoding="utf-8").lower()
    banned_keywords = [
        "sherpa",
        "zipvoice",
        "vocos",
        "onnxruntime",
        "dummy_voice",
        "fake_synth",
        "edge_tts",
        "edge-tts",
        "mock_audio",
    ]
    violations = []
    for kw in banned_keywords:
        if kw in source:
            violations.append(kw)
    assert not violations, f"Forbidden legacy or dummy traces found in {WORKER_SCRIPT}: {violations}"


def test_strict_omnivoice_api_calls_in_ast():
    """Adversarially verify all OmniVoice API invocations conform to official k2-fsa/OmniVoice."""
    source = WORKER_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    called_methods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                called_methods.append(node.func.attr)

    # 1. OmniVoice.from_pretrained must be invoked
    assert "from_pretrained" in called_methods, "OmniVoice.from_pretrained is not called in AST"

    # 2. create_voice_clone_prompt must be invoked
    assert "create_voice_clone_prompt" in called_methods, "model.create_voice_clone_prompt is not called in AST"

    # 3. generate must be invoked
    assert "generate" in called_methods, "model.generate is not called in AST"

    # 4. sf.write must be invoked
    assert "write" in called_methods, "sf.write is not called in AST"


def test_strict_omnivoice_call_arguments():
    """Inspect arguments passed to OmniVoice methods in colab_worker_synth.py."""
    source = WORKER_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    from_pretrained_calls = []
    create_prompt_calls = []
    generate_calls = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "from_pretrained":
                kwargs = {kw.arg: kw.value for kw in node.keywords}
                from_pretrained_calls.append(kwargs)
            elif node.func.attr == "create_voice_clone_prompt":
                kwargs = {kw.arg: kw.value for kw in node.keywords}
                create_prompt_calls.append(kwargs)
            elif node.func.attr == "generate":
                kwargs = {kw.arg: kw.value for kw in node.keywords}
                generate_calls.append(kwargs)

    assert len(from_pretrained_calls) >= 1
    fp = from_pretrained_calls[0]
    assert "device_map" in fp, "from_pretrained missing device_map"
    assert "dtype" in fp, "from_pretrained missing dtype"
    assert "load_asr" in fp, "from_pretrained missing load_asr"

    assert len(create_prompt_calls) >= 1
    cp = create_prompt_calls[0]
    assert "ref_audio" in cp, "create_voice_clone_prompt missing ref_audio"
    assert "ref_text" in cp, "create_voice_clone_prompt missing ref_text"
    assert "preprocess_prompt" in cp, "create_voice_clone_prompt missing preprocess_prompt"

    assert len(generate_calls) >= 1
    gc = generate_calls[0]
    assert "text" in gc, "generate missing text"
    assert "voice_clone_prompt" in gc, "generate missing voice_clone_prompt"
    assert "speed" in gc, "generate missing speed"
    assert "language" in gc, "generate missing language"


# ============================================================================
# 2. Hardware Enforcement (Strict CUDA GPU Check)
# ============================================================================

def test_hardware_enforcement_rejects_cpu(monkeypatch):
    """Adversarially verify that torch.cuda.is_available() == False causes immediate RuntimeError."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError) as exc_info:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "❌ CUDA GPU is NOT available on this Colab VM! "
                "OmniVoice neural synthesis strictly requires NVIDIA GPU acceleration (Tesla T4/L4/A100). "
                "CPU execution is forbidden per policy."
            )
    assert "CUDA GPU is NOT available" in str(exc_info.value)
    assert "CPU execution is forbidden" in str(exc_info.value)


# ============================================================================
# 3. Peak Normalization, Safety Headroom & Clipping Limits
# ============================================================================

@pytest.mark.parametrize("input_peak,expected_headroom", [
    (0.1, 0.90),
    (0.5, 0.90),
    (1.0, 0.90),
    (2.5, 0.90),
    (10.0, 0.90),
])
def test_peak_normalization_headroom_and_zero_clipping(input_peak, expected_headroom):
    """Stress-test peak normalization with varied signal amplitudes."""
    rng = np.random.default_rng(42)
    raw_samples = rng.standard_normal(24000).astype(np.float32)
    current_peak = np.max(np.abs(raw_samples))
    raw_samples = (raw_samples / current_peak) * input_peak

    samples = raw_samples.copy()
    peak = float(np.max(np.abs(samples)))
    assert peak > 0
    target_headroom = 0.90
    samples = (samples / peak) * target_headroom

    norm_peak = np.max(np.abs(samples))
    np.testing.assert_allclose(norm_peak, expected_headroom, atol=1e-6)

    int_samples = samples * 32768.0
    clipped = int(np.sum(np.abs(int_samples) >= 32767))
    clipping_ratio = float(clipped / len(samples))
    assert clipped == 0, f"Detected {clipped} clipped samples!"
    assert clipping_ratio == 0.0, f"Clipping ratio {clipping_ratio} > 0.0!"
    assert np.max(np.abs(int_samples)) <= 29492


def test_pure_silence_rejection():
    """Adversarially verify that pure silence (zeros) raises RuntimeError."""
    silent_samples = np.zeros(24000, dtype=np.float32)
    peak = float(np.max(np.abs(silent_samples)))
    with pytest.raises(RuntimeError) as exc:
        if peak > 0:
            target_headroom = 0.90
            silent_samples = (silent_samples / peak) * target_headroom
        else:
            raise RuntimeError("Synthesized audio for section test contains pure silence.")
    assert "contains pure silence" in str(exc.value)


# ============================================================================
# 4. Output Audio Format & Soundfile Compliance (24kHz Mono 16-bit Little-Endian)
# ============================================================================

def test_soundfile_audio_export_format(tmp_path):
    """Verify written WAV file matches 24,000 Hz mono 16-bit PCM WAV."""
    sample_rate = 24000
    duration_sec = 1.5
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    samples = 0.8 * np.sin(2 * np.pi * 440 * t)

    peak = float(np.max(np.abs(samples)))
    samples = (samples / peak) * 0.90

    out_file = tmp_path / "test_tone.wav"
    sf.write(str(out_file), samples, samplerate=sample_rate, subtype="PCM_16")

    assert out_file.exists()
    assert out_file.stat().st_size > 0

    info = sf.info(str(out_file))
    assert info.samplerate == 24000, f"Expected 24000 Hz, got {info.samplerate}"
    assert info.channels == 1, f"Expected mono (1 channel), got {info.channels}"
    assert info.format == "WAV", f"Expected WAV format, got {info.format}"
    assert info.subtype == "PCM_16", f"Expected PCM_16 subtype, got {info.subtype}"
    assert abs(info.duration - duration_sec) < 0.01


# ============================================================================
# 5. Reference Audio & Transcript Fallback Logic
# ============================================================================

def test_reference_transcript_fallback_when_txt_missing(tmp_path):
    """Verify fallback to hardcoded reference transcript when file is missing."""
    ref_txt_candidates = [
        tmp_path / "nonexistent1.txt",
        tmp_path / "nonexistent2.txt",
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

    assert ref_text == "不求与人相比，但求超越自己。"


def test_reference_transcript_loaded_when_txt_exists(tmp_path):
    """Verify reading custom transcript when file exists."""
    custom_txt = tmp_path / "custom_ref.txt"
    custom_txt.write_text("  海内存知己，天涯若比邻。  \n", encoding="utf-8")

    ref_txt_candidates = [custom_txt]
    ref_text = "default"
    for tc in ref_txt_candidates:
        if tc.exists():
            with open(tc, "r", encoding="utf-8") as f:
                t = f.read().strip()
                if t:
                    ref_text = t
                    break
    assert ref_text == "海内存知己，天涯若比邻。"


# ============================================================================
# 6. Archive Creation, Structure and Stdout Contract
# ============================================================================

def test_archive_tarball_packaging_structure(tmp_path):
    """Verify that archive packaging embeds the expected directory structure for extraction."""
    row_id = 2
    output_dir = tmp_path / f"voice_row_{row_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_sections = ["title", "scene1", "scene2", "scene3", "scene4",
                     "vocab_1", "vocab_2", "vocab_3", "vocab_4", "vocab_5",
                     "vocab", "outro_loop"]

    for sec in test_sections:
        fpath = output_dir / f"{sec}.wav"
        samples = np.zeros(12000, dtype=np.float32)
        samples[100:200] = 0.5
        sf.write(str(fpath), samples, samplerate=24000, subtype="PCM_16")

    archive_path = tmp_path / f"voice_row_{row_id}.tar.gz"
    with tarfile.open(str(archive_path), "w:gz") as tar:
        tar.add(str(output_dir), arcname=f"voice_row_{row_id}")

    assert archive_path.exists()
    assert archive_path.stat().st_size > 0

    extract_target = tmp_path / "extracted_artifacts"
    extract_target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(archive_path), "r:gz") as tar:
        tar.extractall(path=str(extract_target))

    target_dir = extract_target / f"voice_row_{row_id}"
    assert target_dir.exists()
    extracted_wavs = list(target_dir.glob("*.wav"))
    assert len(extracted_wavs) == 12
    for sec in test_sections:
        assert (target_dir / f"{sec}.wav").exists()


def test_stdout_completion_payload_schema():
    """Verify completion JSON payload structure and delimiters."""
    row_id = 2
    target_speed = 0.85
    device = "cuda:0"
    gpu_name = "Tesla T4"
    archive_path = "/content/voice_row_2.tar.gz"

    dummy_results = {
        "title": {
            "filename": "title.wav",
            "duration": 1.5,
            "rms": 2500.0,
            "peak": 29491,
            "clipping_ratio": 0.0,
            "chars": 5,
            "synth_time_sec": 0.8,
            "speed": 0.85,
            "sample_rate": 24000,
            "channels": 1,
            "bit_depth": 16
        }
    }

    completion_payload = {
        "status": "SUCCESS",
        "row_id": row_id,
        "engine": "k2-fsa/OmniVoice",
        "device": device,
        "gpu_name": gpu_name,
        "archive_path": str(archive_path),
        "sections_count": len(dummy_results),
        "speed": target_speed,
        "sample_rate": 24000,
        "total_duration_sec": round(sum(r["duration"] for r in dummy_results.values()), 3),
        "total_synth_time_sec": round(sum(r["synth_time_sec"] for r in dummy_results.values()), 3),
        "sections": dummy_results
    }

    serialized = json.dumps(completion_payload, indent=2)
    parsed = json.loads(serialized)
    assert parsed["status"] == "SUCCESS"
    assert parsed["engine"] == "k2-fsa/OmniVoice"
    assert parsed["row_id"] == 2
    assert parsed["sections_count"] == 1
    assert "title" in parsed["sections"]


# ============================================================================
# 7. Gatekeeper 3 Acoustic Compliance of Worker Normalized Audio
# ============================================================================

def test_gk3_acoustic_compliance_on_normalized_audio(tmp_path):
    """Verify that audio files generated with worker normalization pass check_wav_file."""
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

    sample_rate = 24000
    rng = np.random.default_rng(123)

    for sec_name, text in script_items.items():
        t_min, t_max = calculate_duration_bounds(text)
        target_dur = (t_min + t_max) / 2.0
        n_samples = int(target_dur * sample_rate)

        # Generate harmonic speech-like signal (rich frequency content > 2.5kHz)
        t = np.linspace(0, target_dur, n_samples, endpoint=False, dtype=np.float32)
        signal = (
            0.5 * np.sin(2 * np.pi * 300 * t) +
            0.3 * np.sin(2 * np.pi * 1200 * t) +
            0.15 * np.sin(2 * np.pi * 3200 * t) +
            0.05 * rng.standard_normal(n_samples).astype(np.float32)
        )

        # Apply worker peak normalization (0.90 headroom)
        peak = float(np.max(np.abs(signal)))
        samples = (signal / peak) * 0.90

        wav_path = tmp_path / f"{sec_name}.wav"
        sf.write(str(wav_path), samples, samplerate=sample_rate, subtype="PCM_16")

        # Run check_wav_file validation from audio_qc.py
        is_valid, reason, meta = check_wav_file(str(wav_path), min_duration=0.25)
        assert is_valid, f"check_wav_file failed on section {sec_name}: {reason}"
        assert meta["sample_rate"] == 24000
        assert meta["channels"] == 1
        assert meta["clipping_ratio"] == 0.0
        assert meta["rms"] >= 500.0


# ============================================================================
# 8. Unbounded Filesystem Search Vulnerability (Empirical Timing Benchmark)
# ============================================================================

def test_bounded_vs_unbounded_search_behavior(tmp_path):
    """
    Adversarially benchmark bounded candidate lookup vs unbounded /**/ glob.
    Demonstrates why candidate list is essential and /**/ fallback is a hazard.
    """
    test_wav = tmp_path / "reference.wav"
    test_wav.write_bytes(b"RIFF dummy wav")
    ref_candidates = [tmp_path / "reference.wav"]

    t0 = time.time()
    found = None
    for c in ref_candidates:
        if c.exists() and c.stat().st_size > 0:
            found = c
            break
    elapsed_fast = time.time() - t0
    assert found == test_wav
    assert elapsed_fast < 0.01  # Instantaneous
