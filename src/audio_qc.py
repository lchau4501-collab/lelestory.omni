"""
Audio Quality Control (QC) and Acoustic Integrity Verification Module.
Validates 24,000 Hz mono 16-bit PCM WAV audio characteristics,
minimum RMS amplitude (silence rejection), dynamic speech duration bounding,
and digital peak clipping detection.
Calibrated for 0.85x speech tempo (~1.18x duration vs baseline).
"""

import os
import re
import wave
import struct
import math
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("lelestory.omni.qc")

DEFAULT_SAMPLE_RATE = 24000  # Strictly 24 kHz
DEFAULT_CHANNELS = 1         # Strictly Mono
DEFAULT_SAMPWIDTH = 2        # 16-bit PCM (2 bytes)
DEFAULT_BIT_DEPTH = 16
DEFAULT_MIN_RMS = 500.0       # Minimum RMS amplitude to reject silence
DEFAULT_MIN_DURATION = 0.5    # Minimum absolute seconds
DEFAULT_MAX_CLIPPING_RATIO = 0.01  # Maximum 1% digital clipping threshold

# Punctuation to strip when calculating character count N
PUNCTUATION_REGEX = re.compile(r"[\s\.,\/#!$%\^&\*;:{}=\-_`~()，。！？……、《》“”‘’；：—·]+")


def strip_punctuation(text: str) -> str:
    """Removes whitespace and Chinese/English punctuation marks."""
    if not text:
        return ""
    return PUNCTUATION_REGEX.sub("", text)


def calculate_duration_bounds(text: str) -> Tuple[float, float]:
    """
    Calculates dynamic Chinese speech duration bounds [T_min, T_max] based on character count N,
    calibrated for 0.85x speech tempo (~1.18x duration vs baseline, pitch-preserved).
    
    Formula for 0.85x tempo:
      N = non-punctuation character count
      If N <= 3: T in [0.5s, 5.0s]
      If N > 3:
        T_min = max(1.0, round(N * 0.15, 2))
        T_max = max(4.0, round(N * 1.05 + 2.5, 2))
    """
    stripped = strip_punctuation(text)
    n = len(stripped)

    if n <= 3:
        t_min = 0.5
        t_max = 5.0
    else:
        t_min = max(1.0, round(n * 0.15, 2))
        t_max = max(4.0, round(n * 1.05 + 2.5, 2))

    return round(t_min, 2), round(t_max, 2)


def check_duration_bounds(duration: float, text: str) -> Tuple[bool, str, float, float]:
    """
    Verifies whether the audio duration falls within [T_min, T_max] for the given text.
    """
    t_min, t_max = calculate_duration_bounds(text)
    if duration < t_min:
        return False, f"Audio duration {duration:.2f}s is below minimum bound {t_min:.2f}s (char count={len(strip_punctuation(text))})", t_min, t_max
    if duration > t_max:
        return False, f"Audio duration {duration:.2f}s exceeds maximum bound {t_max:.2f}s (char count={len(strip_punctuation(text))})", t_min, t_max
    return True, f"Duration {duration:.2f}s within bounds [{t_min:.2f}s, {t_max:.2f}s]", t_min, t_max


def calculate_rms(raw_frames: bytes, sampwidth: int) -> float:
    """Calculates the Root Mean Square (RMS) amplitude of raw PCM audio frames."""
    if not raw_frames:
        return 0.0

    if sampwidth == 2:
        # 16-bit signed PCM (<h)
        num_samples = len(raw_frames) // 2
        if num_samples == 0:
            return 0.0
        samples = struct.unpack(f"<{num_samples}h", raw_frames)
        sum_sq = sum(float(s) * float(s) for s in samples)
        return math.sqrt(sum_sq / float(num_samples))
    elif sampwidth == 1:
        # 8-bit unsigned PCM
        num_samples = len(raw_frames)
        if num_samples == 0:
            return 0.0
        samples = [float(b - 128) * 256.0 for b in raw_frames]
        sum_sq = sum(s * s for s in samples)
        return math.sqrt(sum_sq / float(num_samples))
    else:
        # 24-bit or 32-bit fallback
        return 0.0


def check_wav_file(
    filepath: str,
    min_duration: float = DEFAULT_MIN_DURATION,
    expected_sample_rate: int = DEFAULT_SAMPLE_RATE,
    expected_channels: int = DEFAULT_CHANNELS,
    min_rms: float = DEFAULT_MIN_RMS,
    max_clipping_ratio: float = DEFAULT_MAX_CLIPPING_RATIO
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates PCM WAV audio file properties against strict acoustic quality criteria.
    - sample_rate == 24000 Hz
    - channels == 1 (mono)
    - sampwidth == 2 (16-bit PCM)
    - file size > 0
    - duration >= min_duration
    - RMS amplitude >= min_rms (silence threshold >= 500)
    - digital clipping <= max_clipping_ratio (peak digital clipping check)
    
    Returns:
      (is_valid: bool, reason: str, metadata: dict)
    """
    info: Dict[str, Any] = {
        "filepath": filepath,
        "exists": False,
        "file_size": 0,
        "sample_rate": 0,
        "channels": 0,
        "sampwidth": 0,
        "bit_depth": 0,
        "duration": 0.0,
        "rms": 0.0,
        "peak": 0,
        "clipped_samples": 0,
        "clipping_ratio": 0.0,
        "is_valid": False,
    }

    if not filepath or not os.path.exists(filepath):
        msg = f"Audio file does not exist: {filepath}"
        logger.error(msg)
        return False, msg, info

    file_size = os.path.getsize(filepath)
    info["exists"] = True
    info["file_size"] = file_size

    if file_size == 0:
        msg = f"Audio file is 0 bytes: {filepath}"
        logger.error(msg)
        return False, msg, info

    try:
        with wave.open(filepath, "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            frames = wf.getnframes()
            duration = frames / float(sample_rate) if sample_rate > 0 else 0.0

            raw_frames = wf.readframes(frames)
            rms = calculate_rms(raw_frames, sampwidth)

            info.update({
                "sample_rate": sample_rate,
                "channels": channels,
                "sampwidth": sampwidth,
                "bit_depth": sampwidth * 8,
                "duration": round(duration, 3),
                "frames": frames,
                "rms": round(rms, 2),
            })

            # Check 1: Sample Rate (strictly 24000 Hz)
            if sample_rate != expected_sample_rate:
                msg = f"Invalid sample rate {sample_rate}Hz in {filepath}. Expected {expected_sample_rate}Hz."
                logger.error(msg)
                return False, msg, info

            # Check 2: Channels (strictly mono = 1)
            if channels != expected_channels:
                msg = f"Invalid channels {channels} in {filepath}. Expected {expected_channels} (mono)."
                logger.error(msg)
                return False, msg, info

            # Check 3: Sample Width (strictly 16-bit PCM = 2 bytes)
            if sampwidth != 2:
                msg = f"Invalid sample width {sampwidth * 8}-bit in {filepath}. Expected 16-bit PCM."
                logger.error(msg)
                return False, msg, info

            # Check 4: Minimum Duration
            if duration < min_duration:
                msg = f"Audio duration {duration:.2f}s is below minimum threshold {min_duration}s: {filepath}"
                logger.error(msg)
                return False, msg, info

            # Check 5: Silence / RMS Amplitude (RMS >= 500)
            if rms < min_rms:
                msg = f"Audio RMS amplitude {rms:.1f} is below silence threshold {min_rms}: {filepath}"
                logger.error(msg)
                return False, msg, info

            # Check 6: Peak Digital Clipping Detection (clipping_ratio <= max_clipping_ratio)
            if sampwidth == 2 and frames > 0:
                samples = struct.unpack(f"<{frames * channels}h", raw_frames)
                peak = max(abs(s) for s in samples) if samples else 0
                clipped_count = sum(1 for s in samples if abs(s) >= 32767)
                clipping_ratio = clipped_count / float(len(samples)) if samples else 0.0
                info["peak"] = peak
                info["clipped_samples"] = clipped_count
                info["clipping_ratio"] = round(clipping_ratio, 4)

                if clipping_ratio > max_clipping_ratio:
                    msg = f"Severe digital clipping detected ({clipped_count} samples, {clipping_ratio*100:.2f}% > {max_clipping_ratio*100:.1f}%): {filepath}"
                    logger.error(msg)
                    return False, msg, info

            info["is_valid"] = True
            msg = f"Valid WAV: {duration:.2f}s, {sample_rate}Hz, {channels}ch, 16-bit, RMS={rms:.1f}"
            logger.info(f"{filepath} -> {msg}")
            return True, msg, info

    except Exception as e:
        msg = f"Failed to parse WAV file {filepath}: {str(e)}"
        logger.error(msg)
        return False, msg, info


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    target = sys.argv[1] if len(sys.argv) > 1 else "artifacts/voice_row_2/title.wav"
    valid, reason, meta = check_wav_file(target)
    print(f"File: {target}\nValid: {valid}\nReason: {reason}\nMetadata: {meta}")
