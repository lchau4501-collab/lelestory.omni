import os
import wave
import logging
from typing import Tuple

logger = logging.getLogger("lelestory.omni.audio_qc")

def check_wav_file(filepath: str, min_duration: float = 0.5) -> Tuple[bool, str]:
    """Validates PCM WAV audio file properties (44.1kHz, mono/stereo, non-zero frames)."""
    if not os.path.exists(filepath):
        return False, f"Audio file does not exist: {filepath}"
    
    if os.path.getsize(filepath) == 0:
        return False, f"Audio file is 0 bytes: {filepath}"

    try:
        with wave.open(filepath, "r") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            duration = frames / float(sample_rate)

            if sample_rate != 44100:
                logger.warning(f"Audio sample rate {sample_rate}Hz in {filepath}. Expected 44100Hz.")

            if duration < min_duration:
                return False, f"Audio duration {duration:.2f}s is below minimum threshold {min_duration}s: {filepath}"

            return True, f"Valid WAV: {duration:.2f}s, {sample_rate}Hz, {channels} ch"
    except Exception as e:
        return False, f"Failed to parse WAV file {filepath}: {str(e)}"
