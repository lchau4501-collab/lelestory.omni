"""
Audio Generator for LeLe Storybook OmniVoice Engine.
Synthesizes and exports 10 scenes audio + 5 vocabulary audio + outro audio
in authentic 24,000 Hz mono 16-bit PCM WAV format.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from omni_tts import OmniVoiceEngine, generate_pcm_speech_wav, SAMPLE_RATE, CHANNELS
from audio_qc import check_wav_file, check_duration_bounds
from manifest_generator import StoryManifestGenerator

logger = logging.getLogger("lelestory.omni.audio_generator")

class StoryAudioGenerator:
    """
    Coordinates audio synthesis and timing calculation for 10 scenes,
    5 vocabulary flashcards, and outro loop.
    """

    def __init__(self, row_id: int = 2, tempo: float = 1.0, cache_dir: Optional[str] = None):
        self.row_id = row_id
        self.tempo = tempo
        self.engine = OmniVoiceEngine(cache_dir=cache_dir)
        self.manifest_gen = StoryManifestGenerator(row_id=row_id, tempo=tempo)

    def generate_all_story_audio(
        self,
        script_payload: Dict[str, Any],
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Synthesizes 10 scenes audio + 5 vocab audio + outro audio WAV 24kHz mono PCM 16-bit.
        Returns generated file manifest with QC metrics.
        """
        os.makedirs(output_dir, exist_ok=True)
        title = script_payload.get("title", "吃菜的大狼")
        raw_scenes = script_payload.get("scenes") or script_payload.get("lines") or []
        vocabulary = script_payload.get("vocabulary", [])
        outro = script_payload.get("outro", {})

        manifest = self.manifest_gen.build_10_scenes_manifest(
            title=title,
            scenes=raw_scenes,
            vocabulary=vocabulary,
            outro=outro,
            include_vocab_recap=True
        )

        ref_voice = self.engine.ensure_reference_voice()
        generated_files = []
        qc_results = {}

        # 1. Title
        title_wav = os.path.join(output_dir, "title.wav")
        generate_pcm_speech_wav(title, title_wav, tempo=self.tempo, reference_wav_path=ref_voice)
        valid, reason, meta = check_wav_file(title_wav)
        generated_files.append(title_wav)
        qc_results["title.wav"] = {"valid": valid, "duration": meta.get("duration", 0), "reason": reason}

        # 2. 10 Scenes
        for sc in manifest["scenes"]:
            s_num = sc["scene_num"]
            zh_text = sc["zh"]
            sc_wav = os.path.join(output_dir, f"scene{s_num}.wav")
            generate_pcm_speech_wav(zh_text, sc_wav, tempo=self.tempo, reference_wav_path=ref_voice)
            valid, reason, meta = check_wav_file(sc_wav)
            generated_files.append(sc_wav)
            qc_results[f"scene{s_num}.wav"] = {"valid": valid, "duration": meta.get("duration", 0), "reason": reason}

        # 3. 5 Vocabulary Items
        for v in manifest["vocabulary"]:
            v_idx = v["index"]
            w_text = v["word"]
            v_wav = os.path.join(output_dir, f"vocab_{v_idx}.wav")
            generate_pcm_speech_wav(w_text, v_wav, tempo=self.tempo, reference_wav_path=ref_voice)
            valid, reason, meta = check_wav_file(v_wav)
            generated_files.append(v_wav)
            qc_results[f"vocab_{v_idx}.wav"] = {"valid": valid, "duration": meta.get("duration", 0), "reason": reason}

        # 4. Vocab Recap
        if "vocab" in manifest.get("script_items", {}):
            recap_text = manifest["script_items"]["vocab"]
            recap_wav = os.path.join(output_dir, "vocab.wav")
            generate_pcm_speech_wav(recap_text, recap_wav, tempo=self.tempo, reference_wav_path=ref_voice)
            valid, reason, meta = check_wav_file(recap_wav)
            generated_files.append(recap_wav)
            qc_results["vocab.wav"] = {"valid": valid, "duration": meta.get("duration", 0), "reason": reason}

        # 5. Outro Loop
        outro_text = manifest.get("script_items", {}).get("outro_loop", "这些生词来自故事……")
        outro_wav = os.path.join(output_dir, "outro_loop.wav")
        generate_pcm_speech_wav(outro_text, outro_wav, tempo=self.tempo, reference_wav_path=ref_voice)
        valid, reason, meta = check_wav_file(outro_wav)
        generated_files.append(outro_wav)
        qc_results["outro_loop.wav"] = {"valid": valid, "duration": meta.get("duration", 0), "reason": reason}

        # Save manifest
        manifest_path = os.path.join(output_dir, "job_manifest.json")
        manifest["qc_results"] = qc_results
        manifest["generated_files"] = [os.path.basename(p) for p in generated_files]
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        return {
            "status": "Audio_Generated",
            "row_id": self.row_id,
            "title": title,
            "manifest_path": manifest_path,
            "output_dir": output_dir,
            "total_files": len(generated_files),
            "qc_results": qc_results
        }
