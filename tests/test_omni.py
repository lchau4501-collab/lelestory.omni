"""
Unit and Integration Test Suite for LeLe Storybook OmniVoice Voice Generation Engine.
Tests all refactored core modules:
- ModelCacheManager (cache_manager.py)
- Audio QC & Duration Bounds (audio_qc.py)
- OmniVoiceEngine / OmniTTS (omni_tts.py)
- ParallelOrchestrator (parallel_orchestrator.py)
- Gatekeeper3 Quality Auditor & Self-Healing (gatekeeper3.py)
"""

import os
import sys
import json
import wave
import struct
import math
import shutil
import pytest
from unittest.mock import patch, MagicMock

# Add src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from cache_manager import ModelCacheManager, OMNIVOICE_CACHE_DIR, K2FSA_CACHE_DIR
from audio_qc import (
    check_wav_file,
    calculate_duration_bounds,
    check_duration_bounds,
    strip_punctuation,
    calculate_rms,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CHANNELS,
    DEFAULT_BIT_DEPTH,
    DEFAULT_MIN_RMS
)
from omni_tts import (
    OmniVoiceEngine,
    apply_tempo_scaling,
    build_atempo_filter_chain,
    OmniTTS,
    generate_pcm_speech_wav,
    ROW_2_SCRIPT_TEXTS,
    OUTRO_LOOP_TEXT_EXACT,
    REFERENCE_SAMPLE_ID,
    REFERENCE_SAMPLE_FILENAME
)
from parallel_orchestrator import (
    ParallelOrchestrator,
    SUB_WORKFLOWS,
    WORKFLOW_KEYS
)
from gatekeeper3 import (
    Gatekeeper3,
    STORY_AUDIO_SPEC
)


# ============================================================================
# Helper Functions to synthesize test WAV fixtures
# ============================================================================

def make_test_wav(
    filepath: str,
    duration: float = 1.0,
    sample_rate: int = 24000,
    channels: int = 1,
    sampwidth: int = 2,
    amplitude: float = 3000.0,
    freq: float = 440.0,
    include_harmonics: bool = True
):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    num_frames = int(sample_rate * duration)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(num_frames):
            t = float(i) / float(sample_rate)
            if include_harmonics:
                val = amplitude * (0.6 * math.sin(2.0 * math.pi * freq * t) + 0.25 * math.sin(2.0 * math.pi * 3000.0 * t) + 0.15 * math.sin(2.0 * math.pi * 6000.0 * t))
            else:
                val = amplitude * math.sin(2.0 * math.pi * freq * t)
            if sampwidth == 2:
                ival = max(-32768, min(32767, int(val)))
                for ch in range(channels):
                    frames.extend(struct.pack("<h", ival))
            elif sampwidth == 1:
                # 8-bit unsigned
                uval = max(0, min(255, int(val / 256.0 + 128.0)))
                for ch in range(channels):
                    frames.append(uval)
        wf.writeframes(frames)
    return filepath


# ============================================================================
# 1. Cache Manager Tests
# ============================================================================

def test_model_cache_manager_initialization(tmp_path):
    omni_cache = str(tmp_path / "omnivoice")
    k2fsa_cache = str(tmp_path / "k2_fsa")
    mgr = ModelCacheManager(omnivoice_dir=omni_cache, k2fsa_dir=k2fsa_cache)

    assert os.path.isdir(omni_cache)
    assert os.path.isdir(k2fsa_cache)
    paths = mgr.get_cache_paths()
    assert paths["omnivoice"] == omni_cache
    assert paths["k2_fsa"] == k2fsa_cache


def test_model_cache_manager_ensure_cached(tmp_path):
    omni_cache = str(tmp_path / "omnivoice")
    k2fsa_cache = str(tmp_path / "k2_fsa")
    config_file = str(tmp_path / "models.json")
    with open(config_file, "w") as f:
        json.dump({"models": [{"name": "test_model", "path": omni_cache}]}, f)

    mgr = ModelCacheManager(config_path=config_file, omnivoice_dir=omni_cache, k2fsa_dir=k2fsa_cache)
    assert mgr.ensure_models_cached() is True

    status = mgr.verify_cache()
    assert status["omnivoice"] is True
    assert status["k2_fsa"] is True

    # Confirm checkpoint placeholders exist
    assert os.path.exists(os.path.join(omni_cache, "omnivoice_weights.bin"))
    assert os.path.exists(os.path.join(k2fsa_cache, "vocoder.onnx"))


def test_model_cache_manager_pinned_voice_sample(tmp_path):
    omni_cache = str(tmp_path / "omnivoice")
    k2fsa_cache = str(tmp_path / "k2_fsa")
    mgr = ModelCacheManager(omnivoice_dir=omni_cache, k2fsa_dir=k2fsa_cache)
    sample_path = mgr.ensure_voice_sample_cached()

    assert os.path.exists(sample_path)
    assert sample_path.endswith(os.path.join("voice_samples", "reference.wav"))

    valid, reason, meta = check_wav_file(sample_path)
    assert valid is True
    assert meta["sample_rate"] == 24000
    assert meta["channels"] == 1
    assert meta["sampwidth"] == 2
    assert meta["rms"] >= 500.0

    # Test cache persistence on subsequent call
    mtime1 = os.path.getmtime(sample_path)
    sample_path2 = mgr.ensure_voice_sample_cached()
    assert sample_path2 == sample_path
    assert os.path.getmtime(sample_path2) == mtime1


# ============================================================================
# 2. Audio QC & Acoustic Validation Tests
# ============================================================================

def test_audio_qc_valid_wav(tmp_path):
    wav_path = str(tmp_path / "valid.wav")
    make_test_wav(wav_path, duration=1.5, sample_rate=24000, channels=1, sampwidth=2, amplitude=4000.0)

    valid, reason, meta = check_wav_file(wav_path)
    assert valid is True
    assert meta["sample_rate"] == 24000
    assert meta["channels"] == 1
    assert meta["sampwidth"] == 2
    assert meta["rms"] >= 500.0
    assert meta["duration"] >= 1.4


def test_audio_qc_wrong_sample_rate(tmp_path):
    wav_path = str(tmp_path / "wrong_sr.wav")
    make_test_wav(wav_path, sample_rate=44100)

    valid, reason, meta = check_wav_file(wav_path, expected_sample_rate=24000)
    assert valid is False
    assert "Invalid sample rate" in reason
    assert meta["sample_rate"] == 44100


def test_audio_qc_stereo_rejected(tmp_path):
    wav_path = str(tmp_path / "stereo.wav")
    make_test_wav(wav_path, sample_rate=24000, channels=2)

    valid, reason, meta = check_wav_file(wav_path)
    assert valid is False
    assert "Invalid channels" in reason
    assert meta["channels"] == 2


def test_audio_qc_wrong_bit_depth(tmp_path):
    wav_path = str(tmp_path / "8bit.wav")
    make_test_wav(wav_path, sample_rate=24000, channels=1, sampwidth=1)

    valid, reason, meta = check_wav_file(wav_path)
    assert valid is False
    assert "Invalid sample width" in reason


def test_audio_qc_silence_rejected(tmp_path):
    wav_path = str(tmp_path / "silent.wav")
    make_test_wav(wav_path, sample_rate=24000, channels=1, sampwidth=2, amplitude=50.0)

    valid, reason, meta = check_wav_file(wav_path, min_rms=500.0)
    assert valid is False
    assert "below silence threshold" in reason


def test_audio_qc_missing_and_empty_file(tmp_path):
    non_existent = str(tmp_path / "missing.wav")
    v1, r1, _ = check_wav_file(non_existent)
    assert v1 is False
    assert "does not exist" in r1

    empty_file = str(tmp_path / "empty.wav")
    with open(empty_file, "wb"):
        pass
    v2, r2, _ = check_wav_file(empty_file)
    assert v2 is False
    assert "0 bytes" in r2


def test_duration_bounds_calculation():
    # Short words (N <= 3): [0.5s, 5.0s] for 0.85x tempo
    t_min, t_max = calculate_duration_bounds("火锅")
    assert t_min == 0.5
    assert t_max == 5.0

    t_min, t_max = calculate_duration_bounds("大灰狼")
    assert t_min == 0.5
    assert t_max == 5.0

    # Sentence: "吃菜的大狼" (N=5 non-punctuation chars)
    # T_min = max(1.0, round(5 * 0.15, 2)) = 1.0s
    # T_max = max(4.0, round(5 * 1.05 + 2.5, 2)) = 7.75s
    t_min, t_max = calculate_duration_bounds("吃菜的大狼")
    assert t_min == 1.0
    assert t_max == 7.75

    # Check bounds validator
    ok, msg, _, _ = check_duration_bounds(1.26, "吃菜的大狼")
    assert ok is True

    # Under-duration check
    under_ok, under_msg, _, _ = check_duration_bounds(0.3, "吃菜的大狼")
    assert under_ok is False
    assert "below minimum bound" in under_msg

    # Over-duration check
    over_ok, over_msg, _, _ = check_duration_bounds(15.0, "吃菜的大狼")
    assert over_ok is False
    assert "exceeds maximum bound" in over_msg


def test_audio_qc_clipping_rejected(tmp_path):
    wav_path = str(tmp_path / "clipped.wav")
    make_test_wav(wav_path, duration=1.0, sample_rate=24000, channels=1, sampwidth=2, amplitude=40000.0)

    valid, reason, meta = check_wav_file(wav_path)
    assert valid is False
    assert "clipping" in reason.lower()
    assert meta.get("clipped_samples", 0) > 0


def test_audio_qc_check7_spectral_pure_sine_rejected(tmp_path):
    """Verify Check 7 mathematically rejects pure sine-wave synthetic buzzer tones."""
    wav_path = str(tmp_path / "pure_sine.wav")
    make_test_wav(wav_path, duration=1.0, freq=160.0, include_harmonics=False)
    valid, reason, meta = check_wav_file(wav_path)
    assert valid is False
    assert "spectral" in reason.lower() or "synthetic" in reason.lower()
    assert meta["spectral_energy_2500hz_pct"] < 1.0


def test_audio_qc_check7_spectral_harmonic_speech_passes(tmp_path):
    """Verify Check 7 accepts signals with authentic harmonic distribution."""
    wav_path = str(tmp_path / "harmonic.wav")
    make_test_wav(wav_path, duration=1.0, freq=220.0, include_harmonics=True)
    valid, reason, meta = check_wav_file(wav_path)
    assert valid is True
    assert meta["spectral_energy_2500hz_pct"] >= 1.0
    assert meta["spectral_energy_5000hz_pct"] > 0.0


def test_omni_tts_tempo_scaling(tmp_path):
    out_path = str(tmp_path / "tempo_test.wav")
    generate_pcm_speech_wav("测试语速减半", out_path, tempo=0.5)
    valid, reason, meta = check_wav_file(out_path)
    assert valid is True
    assert meta["duration"] >= 2.0


def test_strip_punctuation():
    text = "深山里住着一只大灰狼，名叫罗罗。"
    stripped = strip_punctuation(text)
    assert "，" not in stripped
    assert "。" not in stripped
    assert stripped == "深山里住着一只大灰狼名叫罗罗"
    assert len(stripped) == 14


# ============================================================================
# 3. OmniVoice TTS Engine Tests
# ============================================================================

def test_omni_tts_reference_voice_spec():
    engine = OmniVoiceEngine()
    assert engine.sample_id in ["", "1DpUPJQx-s41jJ25I0PE8HbfVW_DPXHEX"]
    assert engine.reference_name in ["Vegetarian WolfZ.wav", "Vegetarian Wolf.wav"]


def test_omni_tts_script_texts():
    assert ROW_2_SCRIPT_TEXTS["title"] == "吃菜的大狼"
    assert ROW_2_SCRIPT_TEXTS["scene1"] == "深山里住着一只大灰狼，名叫罗罗。"
    assert ROW_2_SCRIPT_TEXTS["scene2"] == "森林里的小动物们都很怕他，一见到他就跑。"
    assert ROW_2_SCRIPT_TEXTS["scene3"] == "别害怕，我不吃肉，我只喜欢吃胡萝卜和白菜！"
    assert ROW_2_SCRIPT_TEXTS["scene4"] == "小兔子们放心地笑了，大家围着罗罗一起开心地吃蔬菜火锅。"
    assert len(ROW_2_SCRIPT_TEXTS["vocab_items"]) == 5
    assert ROW_2_SCRIPT_TEXTS["vocab_items"][0] == ("vocab_1", "大灰狼")
    assert ROW_2_SCRIPT_TEXTS["vocab_items"][4] == ("vocab_5", "火锅")
    assert ROW_2_SCRIPT_TEXTS["vocab"] == "大灰狼 蔬菜 胡萝卜 白菜 火锅"
    assert ROW_2_SCRIPT_TEXTS["outro_loop"] == OUTRO_LOOP_TEXT_EXACT
    assert OUTRO_LOOP_TEXT_EXACT == "这些生词来自故事……"


def test_omni_tts_synthesize_section(tmp_path):
    out_dir = str(tmp_path / "voice")
    engine = OmniVoiceEngine()

    result = engine.synthesize_section("title", output_dir=out_dir, row_id=2)
    assert os.path.exists(result["output_path"])
    assert os.path.exists(result["manifest"])

    # Verify generated WAV with Audio QC
    valid, reason, meta = check_wav_file(result["output_path"])
    assert valid is True
    assert meta["sample_rate"] == 24000
    assert meta["channels"] == 1
    assert meta["sampwidth"] == 2
    assert meta["rms"] >= 500.0


def test_omni_tts_synthesize_vocab(tmp_path):
    out_dir = str(tmp_path / "voice_vocab")
    engine = OmniVoiceEngine()

    result = engine.synthesize_section("vocab", output_dir=out_dir, row_id=2)
    assert len(result["files"]) == 6  # vocab_1 to vocab_5 + vocab.wav
    assert os.path.exists(result["manifest"])

    for f in result["files"]:
        assert os.path.exists(f)
        v, _, m = check_wav_file(f)
        assert v is True
        assert m["sample_rate"] == 24000


def test_omni_tts_backward_compatibility(tmp_path):
    out_dir = str(tmp_path / "audio_compat")
    tts = OmniTTS()
    script_payload = {
        "batch_id": 2,
        "script": {
            "lines": [
                {"speaker": "Narrator", "zh": "深山里住着一只大灰狼", "vi": "Trong nui sau co mot con soi xam"}
            ]
        }
    }
    manifest = tts.synthesize_script(script_payload, output_dir=out_dir)
    assert len(manifest) == 2  # intro + 1 line
    assert manifest[0]["type"] == "intro_chime"
    assert manifest[1]["line_index"] == 0


def test_omni_tts_prioritizes_cached_reference_voice(tmp_path):
    omni_cache = str(tmp_path / "omnivoice")
    pinned_dir = os.path.join(omni_cache, "voice_samples")
    os.makedirs(pinned_dir, exist_ok=True)
    custom_ref = os.path.join(pinned_dir, "reference.wav")
    make_test_wav(custom_ref, duration=1.2, sample_rate=24000, channels=1, sampwidth=2, amplitude=3500.0)

    engine = OmniVoiceEngine(cache_dir=omni_cache)
    ref_path = engine.ensure_reference_voice()
    assert ref_path == custom_ref
    assert os.path.exists(ref_path)

    # Synthesis should reflect cached reference voice in manifest
    out_dir = str(tmp_path / "synth_out")
    result = engine.synthesize_section("title", output_dir=out_dir)
    with open(result["manifest"], "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["reference_cached"] is True
    assert manifest["reference_sample_path"] == custom_ref


# ============================================================================
# 4. Parallel Orchestrator Tests
# ============================================================================

def test_parallel_orchestrator_subworkflow_mapping():
    assert len(SUB_WORKFLOWS) == 7
    assert SUB_WORKFLOWS["title"] == "wfl1_gen_title.yml"
    assert SUB_WORKFLOWS["scene1"] == "wfl2_gen_scene1.yml"
    assert SUB_WORKFLOWS["scene2"] == "wfl3_gen_scene2.yml"
    assert SUB_WORKFLOWS["scene3"] == "wfl4_gen_scene3.yml"
    assert SUB_WORKFLOWS["scene4"] == "wfl5_gen_scene4.yml"
    assert SUB_WORKFLOWS["vocab"] == "wfl6_gen_vocab.yml"
    assert SUB_WORKFLOWS["outro_loop"] == "wfl7_gen_outro_loop.yml"


def test_parallel_orchestrator_api_dispatch():
    orchestrator = ParallelOrchestrator(token="ghp_test_token_12345", repo="naadld/lelestory.omni")

    mock_resp = MagicMock()
    mock_resp.status_code = 204

    with patch("requests.post", return_value=mock_resp) as mock_post:
        success, msg = orchestrator.dispatch_workflow("wfl1_gen_title.yml", row_id=2)
        assert success is True
        assert "Successfully dispatched wfl1_gen_title.yml" in msg

        # Inspect API call parameters
        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        call_headers = mock_post.call_args[1]["headers"]
        call_json = mock_post.call_args[1]["json"]

        assert "actions/workflows/wfl1_gen_title.yml/dispatches" in call_url
        assert call_headers["Authorization"] == "Bearer ghp_test_token_12345"
        assert call_json["inputs"]["row_id"] == "2"


def test_parallel_orchestrator_dispatch_all():
    orchestrator = ParallelOrchestrator(token="ghp_test_token_12345")

    with patch.object(orchestrator, "dispatch_workflow", return_value=(True, "Dispatched")):
        results = orchestrator.dispatch_all_parallel(row_id=2)
        assert len(results) == 7
        for sec in SUB_WORKFLOWS:
            assert sec in results
            assert results[sec][0] is True


def test_parallel_orchestrator_compile_theme_assets():
    script_payload = {"batch_id": 2, "theme": "HANZIDEGUSHI"}
    orchestrator = ParallelOrchestrator(script_payload)
    manifest = orchestrator.compile_theme_assets("hanzi_story", [{"type": "intro_chime"}])
    assert manifest["batch_id"] == 2
    assert manifest["target_workflow"] == "hanzi_story"
    assert manifest["status"] == "hanzi_story_Compiled"


# ============================================================================
# 5. Gatekeeper 3 Quality Audit & Self-Healing Tests
# ============================================================================

def test_gatekeeper3_audit_success(tmp_path):
    voice_dir = str(tmp_path / "voice_gk3")
    os.makedirs(voice_dir, exist_ok=True)

    # Synthesize all 12 required audio files using acoustic generator
    for section, filename, text in STORY_AUDIO_SPEC:
        p = os.path.join(voice_dir, filename)
        generate_pcm_speech_wav(text, p)

    gk3 = Gatekeeper3()
    passed, reason, report = gk3.audit_row(row_id=2, voice_dir=voice_dir, self_heal=False)

    assert passed is True
    assert "GK3 PASS" in reason
    assert len(report["files_audited"]) == 12
    assert report["failed_sections"] == []


def test_gatekeeper3_self_healing_targeted_redispatch(tmp_path):
    voice_dir = str(tmp_path / "voice_partial")
    os.makedirs(voice_dir, exist_ok=True)

    # Generate 11 files, deliberately omit scene3.wav
    for section, filename, text in STORY_AUDIO_SPEC:
        if filename == "scene3.wav":
            continue
        p = os.path.join(voice_dir, filename)
        generate_pcm_speech_wav(text, p)

    gk3 = Gatekeeper3()

    with patch.object(gk3.orchestrator, "dispatch_workflow", return_value=(True, "Healed")) as mock_dispatch:
        passed, reason, report = gk3.audit_row(
            row_id=2,
            voice_dir=voice_dir,
            self_heal=True
        )

        assert passed is False
        assert "scene3" in report["failed_sections"]
        # Verify self-healing triggered ONLY wfl4_gen_scene3.yml
        mock_dispatch.assert_called_once_with("wfl4_gen_scene3.yml", row_id=2, voice_folder_id="1AgtKSIhgRDW4NfMJi5I1X1l2sXYgFh1e")
        assert "scene3" in report["self_healed"]
        assert report["self_healed"]["scene3"]["workflow"] == "wfl4_gen_scene3.yml"


def test_gatekeeper3_validate_manifest_pass(tmp_path):
    voice_dir = str(tmp_path / "voice_manifest")
    intro_path = os.path.join(voice_dir, "intro.wav")
    make_test_wav(intro_path, duration=0.8, sample_rate=24000, channels=1, sampwidth=2, amplitude=3000.0)

    manifest = {
        "batch_id": 2,
        "audio_manifest": [
            {"type": "intro_chime", "path": intro_path}
        ]
    }
    gk3 = Gatekeeper3()
    passed, reason, updated = gk3.validate(manifest)
    assert passed is True
    assert updated["status"] == "GK3_Passed"


# ============================================================================
# 6. Additional Tests for 0.85x Tempo, Filter Chaining, and Drive Resolution
# ============================================================================

def test_build_atempo_filter_chain_standard_values():
    """Verify filter chain generation for standard and sub-half tempo values."""
    assert build_atempo_filter_chain(1.0) == ""
    assert build_atempo_filter_chain(0.85) == "atempo=0.85"
    assert build_atempo_filter_chain(0.95) == "atempo=0.95"
    assert build_atempo_filter_chain(0.5) == "atempo=0.5"
    assert build_atempo_filter_chain(0.4) == "atempo=0.5,atempo=0.8"
    assert build_atempo_filter_chain(0.25) == "atempo=0.5,atempo=0.5"
    assert build_atempo_filter_chain(2.0) == "atempo=2"
    assert build_atempo_filter_chain(2.5) == "atempo=2,atempo=1.25"


def test_build_atempo_filter_chain_invalid_tempo():
    """Verify non-positive tempos raise ValueError."""
    with pytest.raises(ValueError):
        build_atempo_filter_chain(0.0)
    with pytest.raises(ValueError):
        build_atempo_filter_chain(-0.5)


def test_apply_tempo_scaling_085x(tmp_path):
    """Verify active 0.85x tempo scaling preserves 24kHz mono 16-bit PCM and scales duration."""
    test_wav = str(tmp_path / "test_085.wav")
    make_test_wav(test_wav, duration=2.0, sample_rate=24000, channels=1, sampwidth=2, amplitude=4000.0)

    apply_tempo_scaling(test_wav, tempo=0.85)
    valid, reason, meta = check_wav_file(test_wav)
    assert valid is True
    assert meta["sample_rate"] == 24000
    assert meta["channels"] == 1
    assert meta["sampwidth"] == 2
    assert meta["rms"] >= 500.0
    assert abs(meta["duration"] - (2.0 / 0.85)) < 0.05


def test_apply_tempo_scaling_chained_sub_half(tmp_path):
    """Verify filter chaining executes successfully for tempo < 0.5 (0.4x)."""
    test_wav = str(tmp_path / "test_04.wav")
    make_test_wav(test_wav, duration=2.0, sample_rate=24000, channels=1, sampwidth=2, amplitude=4000.0)

    apply_tempo_scaling(test_wav, tempo=0.4)
    valid, reason, meta = check_wav_file(test_wav)
    assert valid is True
    assert abs(meta["duration"] - 5.0) < 0.1


def test_apply_tempo_scaling_chained_super_double(tmp_path):
    """Verify filter chaining executes successfully for tempo > 2.0 (2.5x)."""
    test_wav = str(tmp_path / "test_25.wav")
    make_test_wav(test_wav, duration=2.0, sample_rate=24000, channels=1, sampwidth=2, amplitude=4000.0)

    apply_tempo_scaling(test_wav, tempo=2.5)
    valid, reason, meta = check_wav_file(test_wav)
    assert valid is True
    assert abs(meta["duration"] - 0.8) < 0.05


def test_apply_tempo_scaling_identity_noop(tmp_path):
    """Verify tempo=1.0 performs zero modification."""
    test_wav = str(tmp_path / "test_10.wav")
    make_test_wav(test_wav, duration=2.0, sample_rate=24000, channels=1, sampwidth=2, amplitude=4000.0)
    orig_size = os.path.getsize(test_wav)

    apply_tempo_scaling(test_wav, tempo=1.0)
    assert os.path.getsize(test_wav) == orig_size


def test_duration_bounds_calibration_085x():
    """Verify duration bounds calibration for 0.85x tempo."""
    t_min, t_max = calculate_duration_bounds("火锅")
    assert t_min == 0.5
    assert t_max == 5.0

    t_min, t_max = calculate_duration_bounds("吃菜的大狼")
    assert t_min == 1.0
    assert t_max == 7.75

    ok, _, _, _ = check_duration_bounds(1.26, "吃菜的大狼")
    assert ok is True


def test_dynamic_folder_resolution_in_orchestrator(monkeypatch):
    """Verify orchestrator passes dynamic voice folder ID and does not hardcode Row #2 ID."""
    from parallel_orchestrator import ParallelOrchestrator
    orch = ParallelOrchestrator(token="dummy_token")

    captured_payload = {}
    def mock_post(url, headers, json, timeout):
        nonlocal captured_payload
        captured_payload = json
        class MockResp:
            status_code = 204
        return MockResp()

    monkeypatch.setattr("requests.post", mock_post)
    orch.dispatch_workflow("wfl1_gen_title.yml", row_id=3, voice_folder_id="custom_folder_123")
    assert captured_payload["inputs"]["row_id"] == "3"
    assert captured_payload["inputs"]["voice_folder_id"] == "custom_folder_123"
