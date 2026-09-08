import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from cache_manager import ModelCacheManager
from omni_tts import OmniTTS
from parallel_orchestrator import ParallelOrchestrator
from gatekeeper3 import Gatekeeper3
from audio_qc import check_wav_file

def test_model_cache_manager(tmp_path):
    mgr = ModelCacheManager(config_path="models.json")
    assert mgr.ensure_models_cached() == True

def test_omni_tts(tmp_path):
    output_dir = str(tmp_path / "audio_test")
    tts = OmniTTS()
    script_payload = {
        "batch_id": 1,
        "script": {
            "lines": [
                {"speaker": "LeLe", "zh": "你好", "pinyin": "Nǐ hǎo", "vi": "Xin chào"}
            ]
        }
    }
    audio_manifest = tts.synthesize_script(script_payload, output_dir=output_dir)
    assert len(audio_manifest) == 2  # intro + 1 line

def test_gatekeeper3_pass(tmp_path):
    output_dir = str(tmp_path / "audio_gk3")
    tts = OmniTTS()
    script_payload = {
        "batch_id": 1,
        "theme": "HANZIDEGUSHI",
        "script": {
            "lines": [
                {"speaker": "LeLe", "zh": "休", "pinyin": "Xiū", "vi": "Nghỉ ngơi"}
            ]
        }
    }
    audio_manifest = tts.synthesize_script(script_payload, output_dir=output_dir)
    orchestrator = ParallelOrchestrator(script_payload)
    manifest = orchestrator.compile_theme_assets("hanzi_story", audio_manifest)

    gk3 = Gatekeeper3()
    passed, reason, updated = gk3.validate(manifest)
    assert passed == True
    assert updated["status"] == "GK3_Passed"
