import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest
from colab_orchestrator import build_job_manifest, load_gk2_script_for_row
from drive_resolver import resolve_voice_folder
from audio_qc import calculate_duration_bounds, strip_punctuation


def test_row3_gk2_script_loading():
    """Verify load_gk2_script_for_row loads and adapts Row #3 GK2 script artifact."""
    script_data = load_gk2_script_for_row(3)
    assert script_data is not None, "Failed to load script_gk2_row_3.json"
    assert script_data["title"] == "井底之蛙"
    assert len(script_data["scenes"]) == 10
    assert len(script_data["vocabulary"]) == 5
    assert script_data["include_vocab_recap"] is False

    # Check 1-based indexing on scenes and vocab
    for idx, s in enumerate(script_data["scenes"], 1):
        assert s["scene_num"] == idx
        assert len(s["zh"]) > 0
        assert len(s["pinyin"]) > 0
        assert len(s["en"]) > 0

    for idx, v in enumerate(script_data["vocabulary"], 1):
        assert v["index"] == idx
        assert len(v["word"]) > 0
        assert len(v["pinyin"]) > 0
        assert len(v["en"]) > 0


def test_row3_manifest_exact_17_sections():
    """Verify build_job_manifest for Row #3 generates exactly 17 audio sections with speed 0.85."""
    manifest = build_job_manifest(row_id=3, speed=0.85)

    assert manifest["row_id"] == 3
    assert manifest["title"] == "井底之蛙"
    assert manifest["speed"] == 0.85
    assert manifest["total_audio_sections"] == 17
    assert manifest["scenes_count"] == 10

    script_items = manifest["script_items"]
    assert len(script_items) == 17

    # Mandatory keys
    assert "title" in script_items
    for i in range(1, 11):
        assert f"scene{i}" in script_items
    for i in range(1, 6):
        assert f"vocab_{i}" in script_items
    assert "outro_loop" in script_items

    # Strictly omitted combined recap
    assert "vocab" not in script_items, "Combined vocab recap must be omitted for Row #3"


def test_row2_backward_compatibility():
    """Verify Row #2 retains 18 sections including the vocab recap."""
    manifest = build_job_manifest(row_id=2, speed=0.85)
    assert manifest["row_id"] == 2
    assert manifest["title"] == "吃菜的大狼"
    assert manifest["total_audio_sections"] == 18
    assert "vocab" in manifest["script_items"]


def test_row3_duration_bounds_for_all_17_sections():
    """Verify all 17 sections have mathematically consistent duration bounds."""
    manifest = build_job_manifest(row_id=3, speed=0.70)
    items = manifest["script_items"]

    for sec_key, text in items.items():
        t_min, t_max = calculate_duration_bounds(text)
        stripped = strip_punctuation(text)
        n_chars = len(stripped)

        assert t_min > 0
        assert t_max > t_min
        if n_chars <= 3:
            assert t_min >= 0.25
            assert t_max <= 5.0
        else:
            assert t_min >= 1.0
            assert t_max <= 90.0


def test_row3_voice_folder_resolver():
    """Verify resolve_voice_folder resolves Row #3 to expected Drive folder ID."""
    voice_id, voice_url = resolve_voice_folder(3)
    assert voice_id == "1D8ZZpzMiXTo_S_q56jpj0c5p6YKih72q"
    assert "1D8ZZpzMiXTo_S_q56jpj0c5p6YKih72q" in voice_url
