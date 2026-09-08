import os
import json
import logging
from typing import Dict, Any, Tuple
from audio_qc import check_wav_file

logger = logging.getLogger("lelestory.omni.gk3")

class Gatekeeper3:
    """Gatekeeper 3: Validates audio synthesis quality, LUFS normalization, intro chime timing, and parallel asset manifest."""

    def validate(self, omni_manifest: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        batch_id = omni_manifest.get("batch_id", 0)
        audio_list = omni_manifest.get("audio_manifest", [])

        if not audio_list:
            msg = f"GK3 FAIL: Missing audio manifest for batch #{batch_id}."
            logger.error(msg)
            omni_manifest["status"] = "GK3_Failed"
            omni_manifest["gk3_error"] = msg
            return False, msg, omni_manifest

        # 1. Verify intro chime
        intro_found = False
        for item in audio_list:
            if item.get("type") == "intro_chime":
                intro_found = True
                path = item.get("path")
                valid, reason = check_wav_file(path, min_duration=0.5)
                if not valid:
                    msg = f"GK3 FAIL: Intro chime audio check failed: {reason}"
                    logger.error(msg)
                    omni_manifest["status"] = "GK3_Failed"
                    omni_manifest["gk3_error"] = msg
                    return False, msg, omni_manifest

        if not intro_found:
            msg = f"GK3 FAIL: Missing intro chime WAV audio for batch #{batch_id}."
            logger.error(msg)
            omni_manifest["status"] = "GK3_Failed"
            omni_manifest["gk3_error"] = msg
            return False, msg, omni_manifest

        # 2. Verify all voice line audio files
        line_items = [item for item in audio_list if "line_index" in item]
        if not line_items:
            msg = f"GK3 FAIL: No line voice items found for batch #{batch_id}."
            logger.error(msg)
            omni_manifest["status"] = "GK3_Failed"
            omni_manifest["gk3_error"] = msg
            return False, msg, omni_manifest

        for item in line_items:
            zh_path = item.get("zh_audio_path")
            vi_path = item.get("vi_audio_path")

            valid_zh, reason_zh = check_wav_file(zh_path, min_duration=1.0)
            if not valid_zh:
                msg = f"GK3 FAIL: Line #{item.get('line_index')} Chinese audio failed: {reason_zh}"
                logger.error(msg)
                omni_manifest["status"] = "GK3_Failed"
                omni_manifest["gk3_error"] = msg
                return False, msg, omni_manifest

            valid_vi, reason_vi = check_wav_file(vi_path, min_duration=1.0)
            if not valid_vi:
                msg = f"GK3 FAIL: Line #{item.get('line_index')} Vietnamese audio failed: {reason_vi}"
                logger.error(msg)
                omni_manifest["status"] = "GK3_Failed"
                omni_manifest["gk3_error"] = msg
                return False, msg, omni_manifest

        omni_manifest["status"] = "GK3_Passed"
        logger.info(f"GK3 PASSED for batch #{batch_id} ({len(line_items)} voice lines validated)")
        return True, "GK3 Passed", omni_manifest

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gatekeeper 3 Validator")
    parser.add_argument("--script-input", type=str, default="artifacts/script_gk2.json")
    parser.add_argument("--output", type=str, default="artifacts/omni_gk3.json")
    args = parser.parse_args()

    from omni_tts import OmniTTS
    from parallel_orchestrator import ParallelOrchestrator

    with open(args.script_input, "r", encoding="utf-8") as f:
        script_data = json.load(f)

    tts = OmniTTS()
    audio_manifest = tts.synthesize_script(script_data)

    orchestrator = ParallelOrchestrator(script_data)
    theme = script_data.get("theme", "HANZIDEGUSHI")
    target_wf = "hanzi_story" if theme == "HANZIDEGUSHI" else "tts_audio"
    compiled_manifest = orchestrator.compile_theme_assets(target_wf, audio_manifest)

    gk3 = Gatekeeper3()
    passed, reason, updated = gk3.validate(compiled_manifest)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    if not passed:
        exit(1)
