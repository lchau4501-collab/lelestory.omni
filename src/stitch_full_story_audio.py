#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Story Audio Stitching Engine for LeLeStory Pipeline.
Requirement R1 & R3: OmniVoice 24kHz Audio & Timeline Contracts (2026-09-15 Locked Specification).
Implements exact timeline silences:
- LEAD_IN_SILENCE = 2.0s (before Scene 1)
- INTER_SCENE_SILENCE = 2.0s (between scenes 1..10)
- SCENE_TO_VOCAB_SILENCE = 4.0s (Scene 10 -> Slide 11)
- INTER_VOCAB_SILENCE = 1.0s
- Slide 11 silent vocabulary progression: 2.0s silent reading time per vocab item (5 rows = 10.0s total)
- Audio active ONLY at outro loop sentence ("这些生词来自故事……")
- Instant video/audio termination upon completion of outro loop (0.0s padding).
"""

import os
import sys
import wave
import struct
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict, Any

logger = logging.getLogger("lelestory.omni.stitch")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPWIDTH = 2  # 16-bit PCM

# Canonical timeline silence intervals conforming to 2026-09-15 locked specification
LEAD_IN_SILENCE = 2.0                    # 2.0s silence before Scene 1 audio begins
INTER_SCENE_SILENCE = 2.0                # 2.0s silence between scene narrations
SCENE_TO_VOCAB_SILENCE = 4.0             # 4.0s silence transition from Scene 10 to Slide 11
INTER_VOCAB_SILENCE = 1.0                # 1.0s silence interval constant
VOCAB_SILENT_PROGRESSION_DURATION = 2.0  # 2.0s silent reading progression per vocab item on Slide 11


def create_silence_bytes(duration_sec: float) -> bytes:
    """Generates raw PCM bytes of silence at 24kHz mono 16-bit."""
    n_samples = max(0, int(SAMPLE_RATE * duration_sec))
    return b"\x00" * (n_samples * SAMPWIDTH * CHANNELS)


def create_silence_wav(
    output_path: Union[str, Path],
    duration_seconds: float,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    sampwidth: int = SAMPWIDTH
) -> str:
    """Creates a standalone silence WAV file with exact duration at 24kHz mono 16-bit PCM."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n_samples = max(0, int(sample_rate * duration_seconds))
    raw_data = b"\x00" * (n_samples * sampwidth * channels)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_data)
    return str(p)


def read_wav_frames(path: Path) -> bytes:
    """
    Reads audio frames from WAV file. Resamples via ffmpeg if sample rate,
    channel count, or bit depth do not match the 24kHz mono 16-bit specification.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    with wave.open(str(path), "rb") as wf:
        n_ch = wf.getnchannels()
        s_width = wf.getsampwidth()
        s_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if s_rate != SAMPLE_RATE or n_ch != CHANNELS or s_width != SAMPWIDTH:
        tmp_conv = str(path) + ".resampled.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(path),
            "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-c:a", "pcm_s16le",
            tmp_conv
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with wave.open(tmp_conv, "rb") as wf2:
            frames = wf2.readframes(wf2.getnframes())
        if os.path.exists(tmp_conv):
            os.remove(tmp_conv)

    return frames


def build_story_audio_sequence(
    voice_dir: Path,
    num_scenes: int = 10,
    num_vocab: int = 5
) -> List[Tuple[str, Union[float, Path]]]:
    """
    Constructs the canonical audio/silence timeline sequence:
    1. Title audio
    2. 2.0s lead-in silence (LEAD_IN_SILENCE)
    3. Scenes 1..N with 2.0s inter-scene silences (INTER_SCENE_SILENCE)
    4. 4.0s scene-to-vocab transition silence (SCENE_TO_VOCAB_SILENCE)
    5. Slide 11: 5-vocab silent reading progression (2.0s per row, audio only at outro loop)
    6. Outro loop audio (outro_loop.wav)
    7. Instant termination (0.0s padding after outro loop)
    """
    sequence: List[Tuple[str, Union[float, Path]]] = []

    # Title
    title_wav = voice_dir / "title.wav"
    if title_wav.exists():
        sequence.append(("TITLE", title_wav))

    # Lead-in silence before Scene 1
    sequence.append(("SILENCE_PRE_SCENE1", LEAD_IN_SILENCE))

    # Scenes 1..num_scenes
    for s in range(1, num_scenes + 1):
        sc_wav = voice_dir / f"scene{s}.wav"
        sequence.append((f"SCENE_{s}", sc_wav))
        if s < num_scenes:
            sequence.append((f"SILENCE_INTER_SCENE_{s}_{s+1}", INTER_SCENE_SILENCE))
        else:
            sequence.append(("SILENCE_SCENE_TO_VOCAB", SCENE_TO_VOCAB_SILENCE))

    # Slide 11: 5-vocab silent reading progression (audio active ONLY at outro loop!)
    for v in range(1, num_vocab + 1):
        sequence.append((f"SILENCE_VOCAB_{v}", VOCAB_SILENT_PROGRESSION_DURATION))

    # Outro loop audio
    outro_wav = voice_dir / "outro_loop.wav"
    if outro_wav.exists():
        sequence.append(("OUTRO_LOOP", outro_wav))

    return sequence


def stitch_story_audio(
    voice_dir: Union[str, Path],
    output_wav: Union[str, Path],
    output_mp3: Optional[Union[str, Path]] = None,
    num_scenes: int = 10,
    num_vocab: int = 5
) -> Tuple[str, float]:
    """
    Stitches story audio clips and timeline silences into a single continuous master WAV file.
    Conforms strictly to:
    - 24,000 Hz mono 16-bit PCM WAV
    - Slide 11 silent vocabulary progression (audio only at outro loop)
    - Instant termination upon completion of outro loop.
    Returns (output_wav_path, total_duration_seconds).
    """
    v_dir = Path(voice_dir)
    out_wav = Path(output_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    sequence = build_story_audio_sequence(v_dir, num_scenes=num_scenes, num_vocab=num_vocab)

    all_frames: List[bytes] = []
    current_time = 0.0
    logger.info(f"Stitching story audio from {v_dir} -> {out_wav}")

    for item_type, val in sequence:
        if isinstance(val, (int, float)):
            dur = float(val)
            all_frames.append(create_silence_bytes(dur))
            current_time += dur
            logger.debug(f"  [+{dur:4.1f}s SILENCE] -> {current_time:6.2f}s | {item_type}")
        elif isinstance(val, Path):
            if not val.exists():
                logger.warning(f"Audio file {val.name} not found, substituting 2.0s silence.")
                all_frames.append(create_silence_bytes(2.0))
                current_time += 2.0
                continue
            frames = read_wav_frames(val)
            dur = len(frames) / (SAMPLE_RATE * SAMPWIDTH * CHANNELS)
            all_frames.append(frames)
            current_time += dur
            logger.debug(f"  [AUDIO {dur:5.2f}s] -> {current_time:6.2f}s | {item_type} ({val.name})")

    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPWIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(all_frames))

    logger.info(f"✅ Master WAV generated: {out_wav} (Total duration: {current_time:.2f}s)")

    # Optional MP3 conversion
    if output_mp3 is not None:
        out_mp3 = Path(output_mp3)
        out_mp3.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(out_wav),
                "-codec:a", "libmp3lame", "-b:a", "320k",
                str(out_mp3)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"✅ Master MP3 generated: {out_mp3}")
        except Exception as e:
            logger.warning(f"Could not convert to MP3 via ffmpeg: {e}")

    return str(out_wav), current_time


def build_full_story(
    voice_dir: Optional[str] = None,
    output_wav: Optional[str] = None,
    output_mp3: Optional[str] = None
) -> Tuple[str, float]:
    """Default runner for CLI or programmatic execution."""
    v_dir = Path(voice_dir) if voice_dir else Path("/media/vpsg24gb/DATA/lelehoctiengtrung/lelestory/artifacts/#4 - 小马过河/voice")
    o_wav = Path(output_wav) if output_wav else Path("/media/vpsg24gb/DATA/lelehoctiengtrung/lelestory/artifacts/#4 - 小马过河/story_4_full_master_voice_0.70x.wav")
    o_mp3 = Path(output_mp3) if output_mp3 else Path("/media/vpsg24gb/DATA/lelehoctiengtrung/lelestory/artifacts/#4 - 小马过河/story_4_full_master_voice_0.70x.mp3")

    return stitch_story_audio(v_dir, o_wav, o_mp3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch full story audio with exact timeline silences")
    parser.add_argument("--voice-dir", type=str, default=None, help="Directory containing audio clips")
    parser.add_argument("--output-wav", type=str, default=None, help="Output master WAV path")
    parser.add_argument("--output-mp3", type=str, default=None, help="Output master MP3 path")
    parser.add_argument("--num-scenes", type=int, default=10, help="Number of scene clips (default: 10)")
    args = parser.parse_args()

    v_path = Path(args.voice_dir) if args.voice_dir else None
    w_path = Path(args.output_wav) if args.output_wav else None
    m_path = Path(args.output_mp3) if args.output_mp3 else None

    build_full_story(v_path, w_path, m_path)
