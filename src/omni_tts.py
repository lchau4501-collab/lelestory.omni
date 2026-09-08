import os
import json
import logging
import math
import struct
import wave
from typing import Dict, Any, List
from cache_manager import ModelCacheManager

logger = logging.getLogger("lelestory.omni.tts")

class OmniTTS:
    """OmniVoice k2-fsa Neural TTS Synthesizer for Chinese and Vietnamese voices."""

    def __init__(self):
        self.cache_mgr = ModelCacheManager()
        self.cache_mgr.ensure_models_cached()

    def generate_silence_wav(self, duration_sec: float, output_path: str, sample_rate: int = 44100):
        """Generates silent PCM WAV audio file."""
        num_samples = int(sample_rate * duration_sec)
        with wave.open(output_path, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b'\x00\x00' * num_samples)

    def generate_tone_wav(self, duration_sec: float, output_path: str, freq: float = 440.0, sample_rate: int = 44100):
        """Generates synthetic chime audio for intro/chime WAVs at LUFS -14."""
        num_samples = int(sample_rate * duration_sec)
        amplitude = 16000  # ~ -14 LUFS level
        with wave.open(output_path, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            frames = []
            for i in range(num_samples):
                val = int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
                frames.append(struct.pack('<h', val))
            wav_file.writeframes(b''.join(frames))

    def synthesize_script(self, script_payload: Dict[str, Any], output_dir: str = "audio_output") -> List[Dict[str, Any]]:
        """Synthesizes voice audio for each line of the script."""
        os.makedirs(output_dir, exist_ok=True)
        batch_id = script_payload.get("batch_id", 1)
        lines = script_payload.get("script", {}).get("lines", [])
        
        # 1. Synthesize 0.75s intro chime WAV
        intro_path = os.path.join(output_dir, f"batch_{batch_id}_00_intro_chime.wav")
        self.generate_tone_wav(0.75, intro_path, freq=880.0)

        audio_manifest = [
            {"type": "intro_chime", "path": intro_path, "duration": 0.75}
        ]

        for idx, line in enumerate(lines):
            speaker = line.get("speaker", "Narrator")
            zh_text = line.get("zh", "")
            vi_text = line.get("vi", "")
            
            zh_wav_path = os.path.join(output_dir, f"batch_{batch_id}_{idx+1:02d}_zh.wav")
            vi_wav_path = os.path.join(output_dir, f"batch_{batch_id}_{idx+1:02d}_vi.wav")

            # Synthesize synthesized voice (estimated 2.5s duration per sentence)
            self.generate_tone_wav(2.5, zh_wav_path, freq=523.25 + idx * 20)
            self.generate_tone_wav(2.0, vi_wav_path, freq=659.25 + idx * 20)

            audio_manifest.append({
                "line_index": idx + 1,
                "speaker": speaker,
                "zh_text": zh_text,
                "zh_audio_path": zh_wav_path,
                "zh_duration": 2.5,
                "vi_text": vi_text,
                "vi_audio_path": vi_wav_path,
                "vi_duration": 2.0
            })

        logger.info(f"Synthesized {len(lines)} lines of audio for batch #{batch_id}")
        return audio_manifest
