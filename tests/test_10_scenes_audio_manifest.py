# -*- coding: utf-8 -*-
"""
Tests for 10-Scenes Manifest & Audio Generator in lelestory.omni.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from manifest_generator import StoryManifestGenerator
from audio_generator import StoryAudioGenerator
from audio_qc import check_wav_file

def test_10_scenes_manifest_generation():
    generator = StoryManifestGenerator(row_id=2, tempo=1.0)
    scenes = [
        {"scene_num": i, "speaker": "旁白", "zh": f"这是第{i}个精彩的故事情节句子。", "pinyin": f"zhè shì dì {i} gè gù shì.", "en": f"This is scene {i}."}
        for i in range(1, 11)
    ]
    vocab = [
        {"word": "大灰狼", "pinyin": "dà huī láng", "en": "wolf"},
        {"word": "蔬菜", "pinyin": "shū cài", "en": "vegetables"},
        {"word": "胡萝卜", "pinyin": "hú luó bo", "en": "carrot"},
        {"word": "白菜", "pinyin": "bái cài", "en": "cabbage"},
        {"word": "火锅", "pinyin": "huǒ guō", "en": "hotpot"}
    ]
    outro = {"zh": "这些生词来自故事……", "pinyin": "zhè xiē shēng cí lái zì gù shì ……", "en": "Those vocabulary comes from the story..."}

    manifest = generator.build_10_scenes_manifest(
        title="吃菜的大狼",
        scenes=scenes,
        vocabulary=vocab,
        outro=outro,
        include_vocab_recap=True
    )

    assert manifest["row_id"] == 2
    assert manifest["title"] == "吃菜的大狼"
    assert manifest["scenes_count"] == 10
    assert manifest["vocab_count"] == 5
    assert manifest["sample_rate"] == 24000
    assert manifest["channels"] == 1
    assert manifest["bit_depth"] == 16
    assert len(manifest["scenes"]) == 10
    assert len(manifest["vocabulary"]) == 5
    assert manifest["total_audio_sections"] == 18  # title + 10 scenes + 5 vocab + recap + outro

    for idx, sc in enumerate(manifest["scenes"], 1):
        assert sc["scene_num"] == idx
        assert sc["file"] == f"scene{idx}.wav"
        assert sc["estimated_duration"] >= 1.0
        assert sc["start_time"] >= 0.0

def test_10_scenes_audio_generator_mock():
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_gen = StoryAudioGenerator(row_id=2, tempo=1.0, cache_dir=tmpdir)
        
        # Mock ensure_reference_voice and generate_pcm_speech_wav
        ref_mock_path = os.path.join(tmpdir, "reference.wav")
        with open(ref_mock_path, "wb") as f:
            f.write(b"RIFFmockwav")

        def fake_synth(text, output_path, **kwargs):
            # Create genuine mock WAV
            import wave, struct
            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                samples = bytearray()
                for _ in range(24000):  # 1 second
                    samples.extend(struct.pack("<h", 3000))
                wf.writeframes(samples)
            return output_path

        script_payload = {
            "title": "吃菜的大狼",
            "scenes": [
                {"scene_num": i, "speaker": "旁白", "zh": f"第{i}幕故事", "pinyin": f"dì {i} mù", "en": f"Scene {i}"}
                for i in range(1, 11)
            ],
            "vocabulary": [
                {"word": "词1", "pinyin": "c1", "en": "w1"},
                {"word": "词2", "pinyin": "c2", "en": "w2"},
                {"word": "词3", "pinyin": "c3", "en": "w3"},
                {"word": "词4", "pinyin": "c4", "en": "w4"},
                {"word": "词5", "pinyin": "c5", "en": "w5"},
            ],
            "outro": {"zh": "这些生词来自故事……", "pinyin": "zhè xiē", "en": "Those..."}
        }

        with patch.object(audio_gen.engine, "ensure_reference_voice", return_value=ref_mock_path), \
             patch("audio_generator.generate_pcm_speech_wav", side_effect=fake_synth):
            
            res = audio_gen.generate_all_story_audio(script_payload, tmpdir)
            assert res["status"] == "Audio_Generated"
            assert res["total_files"] == 18  # title + 10 scenes + 5 vocab + recap + outro
            assert os.path.exists(res["manifest_path"])
            for i in range(1, 11):
                assert os.path.exists(os.path.join(tmpdir, f"scene{i}.wav"))
