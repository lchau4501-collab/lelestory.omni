"""
Alignment Generator for LeLe Storybook OmniVoice Pipeline.
Extracts character-by-character start and end timestamps [{"char": c, "pinyin": py, "start": t0, "end": t1}]
from Omni Voice 24kHz WAV audio using sherpa-onnx CTC OfflineRecognizer,
with deterministic linear syllable interpolation fallback for offline testing.
"""

import os
import re
import json
import logging
import wave
import struct
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("lelestory.omni.alignment")

# Punctuation set for pause boundary detection
PUNCTUATION_PAUSE_REGEX = re.compile(r"[\s\.,\/#!$%\^&\*;:{}=\-_`~()，。！？……、《》“”‘’；：—·]+")
CHINESE_CHAR_REGEX = re.compile(r"[\u4e00-\u9fff]")
PUNCTUATION_CHARS = set("，。！？；：“”‘’（）《》【】……——·、.,!?;:\"'()[]{}<>-`~")

DEFAULT_MODEL_DIR = os.path.expanduser("~/.cache/sherpa-onnx/ctc_aligner")


def get_audio_duration(wav_path: str) -> float:
    """Extracts duration in seconds from standard PCM WAV file."""
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
    with wave.open(wav_path, "rb") as wf:
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        if framerate <= 0:
            raise ValueError(f"Invalid framerate {framerate} in {wav_path}")
        return round(float(nframes) / float(framerate), 3)


def linear_syllable_interpolation(text: str, duration: float) -> List[Dict[str, Any]]:
    """
    Fast, deterministic linear syllable interpolation fallback for Chinese character alignment.
    Models natural speech lead-in/lead-out silences and pause boundaries at punctuation.
    Extracts contextual Pinyin tone marks via pypinyin.
    Execution time: < 0.2ms per sentence. Zero GPU/VPS burden.
    """
    try:
        import pypinyin
    except ImportError:
        pypinyin = None

    # Strip whitespace & split by pause punctuation
    segments = re.split(r'([，,、；;])', text)
    clauses = []
    comma_count = 0
    for part in segments:
        if part in '，,、；;':
            comma_count += 1
        else:
            clean = [c for c in part if CHINESE_CHAR_REGEX.match(c) or c.isalnum()]
            if clean:
                clauses.append(clean)

    total_chars = sum(len(c) for c in clauses)
    if total_chars == 0:
        return []

    dur_bound = round(max(0.0, float(duration)), 3)
    if dur_bound <= 0.0:
        pys = []
        if pypinyin:
            try:
                pys = pypinyin.pinyin(text, style=pypinyin.Style.TONE, errors=lambda c: [char for char in c])
            except Exception as e:
                logger.warning(f"pypinyin parsing failed: {e}")
                pys = []
        characters = []
        char_idx = 0
        text_chars = list(text)
        for clause in clauses:
            for c in clause:
                py = ""
                if pys:
                    while char_idx < len(text_chars) and text_chars[char_idx] != c:
                        char_idx += 1
                    if char_idx < len(pys) and pys[char_idx]:
                        py = pys[char_idx][0]
                    char_idx += 1
                if (not py or any(p in PUNCTUATION_CHARS for p in py)) and pypinyin and CHINESE_CHAR_REGEX.match(c):
                    single_py = pypinyin.pinyin(c, style=pypinyin.Style.TONE)
                    py = single_py[0][0] if (single_py and single_py[0] and single_py[0][0]) else ""
                elif not py and not CHINESE_CHAR_REGEX.match(c):
                    py = c
                characters.append({"char": c, "pinyin": py, "start": 0.0, "end": 0.0})
        return characters

    default_lead_in = min(0.18, max(0.08, dur_bound * 0.05))
    default_lead_out = min(0.18, max(0.08, dur_bound * 0.05))
    avail_duration = max(0.0, dur_bound - default_lead_in - default_lead_out)

    # Allow 0.20s pause per comma if enough time is available
    pause_dur = 0.20 if comma_count > 0 and avail_duration > (comma_count * 0.20 + total_chars * 0.15) else 0.0
    total_pause = comma_count * pause_dur
    speech_avail = avail_duration - total_pause
    default_needed = total_chars * 0.08

    if speech_avail >= default_needed:
        lead_in = default_lead_in
        char_dur = speech_avail / total_chars
    else:
        # Duration is smaller than default syllable minimums.
        # Scale speech window proportionally within [0.0, dur_bound].
        pause_dur = 0.0
        lead_in = min(0.08, dur_bound * 0.05)
        lead_out = min(0.08, dur_bound * 0.05)
        avail_speech = max(0.0, dur_bound - lead_in - lead_out)
        if avail_speech <= 0.0:
            lead_in = 0.0
            lead_out = 0.0
            avail_speech = dur_bound
        char_dur = avail_speech / total_chars

    # Contextual sentence-level Pinyin extraction (handles polyphonic words like 一只 zhī)
    pys = []
    if pypinyin:
        try:
            # Using errors=lambda c: [char for char in c] forces 1:1 character-to-pinyin
            # mapping, preventing multi-character punctuation clusters (：“, ！”) or
            # numbers/English words from compressing into single tokens and desyncing char_idx.
            pys = pypinyin.pinyin(text, style=pypinyin.Style.TONE, errors=lambda c: [char for char in c])
        except Exception as e:
            logger.warning(f"pypinyin parsing failed: {e}")
            pys = []

    characters = []
    curr_time = lead_in
    char_idx = 0
    text_chars = list(text)

    for clause_idx, clause in enumerate(clauses):
        for c in clause:
            # Find matching pinyin
            py = ""
            if pys:
                while char_idx < len(text_chars) and text_chars[char_idx] != c:
                    char_idx += 1
                if char_idx < len(pys) and pys[char_idx]:
                    py = pys[char_idx][0]
                char_idx += 1

            if (not py or any(p in PUNCTUATION_CHARS for p in py)) and pypinyin and CHINESE_CHAR_REGEX.match(c):
                single_py = pypinyin.pinyin(c, style=pypinyin.Style.TONE)
                if single_py and single_py[0] and single_py[0][0]:
                    py = single_py[0][0]
                else:
                    py = ""
            elif not py and not CHINESE_CHAR_REGEX.match(c):
                py = c

            c_start = round(curr_time, 3)
            c_end = round(curr_time + char_dur, 3)
            # Strictly bound within [0.0, dur_bound]
            c_start = max(0.0, min(dur_bound, c_start))
            c_end = max(c_start, min(dur_bound, c_end))

            characters.append({
                "char": c,
                "pinyin": py,
                "start": c_start,
                "end": c_end
            })
            curr_time += char_dur

        if clause_idx < len(clauses) - 1 and pause_dur > 0:
            curr_time += pause_dur

    return characters


class SherpaCTCAligner:
    """
    Wrapper around sherpa-onnx OfflineRecognizer for acoustic forced alignment.
    Accepts 24,000 Hz WAV audio and outputs character-by-character timestamps.
    Falls back gracefully to linear syllable interpolation when models are absent.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.recognizer = None
        self._init_recognizer()

    def _init_recognizer(self):
        try:
            import sherpa_onnx
        except ImportError:
            logger.info("sherpa-onnx not installed; aligner will use linear syllable fallback.")
            return

        # Candidate model files in model_dir or ~/.cache/sherpa-onnx/
        candidates = [
            (os.path.join(self.model_dir, "model.int8.onnx"), os.path.join(self.model_dir, "tokens.txt")),
            (os.path.join(self.model_dir, "model.onnx"), os.path.join(self.model_dir, "tokens.txt")),
            (os.path.expanduser("~/.cache/sherpa-onnx/model.int8.onnx"), os.path.expanduser("~/.cache/sherpa-onnx/tokens.txt")),
        ]
        for model_path, tokens_path in candidates:
            if os.path.isfile(model_path) and os.path.isfile(tokens_path):
                try:
                    logger.info(f"Loading sherpa-onnx CTC model from {model_path}...")
                    self.recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
                        model=model_path,
                        tokens=tokens_path,
                        sample_rate=16000,
                        provider="cpu"
                    )
                    break
                except Exception as e:
                    logger.warning(f"Failed to load CTC model from {model_path}: {e}")

    def is_available(self) -> bool:
        return self.recognizer is not None

    def align(self, wav_path: str, text: str, duration: float) -> Tuple[List[Dict[str, Any]], str]:
        """
        Performs alignment on 24kHz WAV audio.
        Returns (characters_list, method_name).
        """
        if not self.is_available():
            return linear_syllable_interpolation(text, duration), "linear_interpolation_fallback"

        try:
            import soundfile as sf
            samples, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)

            stream = self.recognizer.create_stream()
            stream.accept_waveform(sr, samples)
            self.recognizer.decode_stream(stream)
            res = stream.result

            tokens = res.tokens
            timestamps = res.timestamps
            durations = res.durations if hasattr(res, "durations") else []

            if not tokens or len(tokens) == 0:
                return linear_syllable_interpolation(text, duration), "linear_interpolation_fallback"

            # Contextual Pinyin via 1:1 token alignment (avoid dict key overwrites)
            import pypinyin
            clean_chars = []
            clean_pinyins = []
            try:
                pys = pypinyin.pinyin(text, style=pypinyin.Style.TONE, errors=lambda c: [char for char in c])
                for ch, py_item in zip(text, pys):
                    if CHINESE_CHAR_REGEX.match(ch) or ch.isalnum():
                        clean_chars.append(ch)
                        clean_pinyins.append(py_item[0] if (py_item and py_item[0]) else "")
            except Exception as exc:
                logger.warning(f"Contextual pypinyin failed in CTC aligner: {exc}")
                for ch in text:
                    if CHINESE_CHAR_REGEX.match(ch) or ch.isalnum():
                        clean_chars.append(ch)
                        clean_pinyins.append("")

            aligned_chars = []
            token_idx = 0
            prev_end = 0.0
            for char_pos, ch in enumerate(clean_chars):
                if token_idx < len(tokens):
                    t0 = round(float(timestamps[token_idx]), 3)
                    t_dur = round(float(durations[token_idx]), 3) if (durations and token_idx < len(durations)) else 0.20
                    t1 = round(t0 + max(0.05, t_dur), 3)
                    token_idx += 1
                else:
                    t0 = prev_end
                    t1 = min(round(duration, 3), round(t0 + 0.20, 3))

                if t0 < prev_end:
                    t0 = prev_end
                if t1 <= t0:
                    t1 = round(t0 + 0.05, 3)
                t1 = min(round(duration + 0.05, 3), t1)

                py = clean_pinyins[char_pos] if char_pos < len(clean_pinyins) else ""
                if not py or any(p in PUNCTUATION_CHARS for p in py):
                    if pypinyin and CHINESE_CHAR_REGEX.match(ch):
                        single_py = pypinyin.pinyin(ch, style=pypinyin.Style.TONE)
                        if single_py and single_py[0] and single_py[0][0]:
                            py = single_py[0][0]
                        else:
                            py = ""
                    else:
                        py = ch if not CHINESE_CHAR_REGEX.match(ch) else ""

                aligned_chars.append({
                    "char": ch,
                    "pinyin": py,
                    "start": t0,
                    "end": t1
                })
                prev_end = t0

            return aligned_chars, "sherpa_onnx_ctc"

        except Exception as exc:
            logger.warning(f"sherpa-onnx alignment failed with exception: {exc}. Falling back to linear interpolation.")
            return linear_syllable_interpolation(text, duration), "linear_interpolation_fallback"


# Singleton aligner instance for reuse
_global_aligner: Optional[SherpaCTCAligner] = None


def get_global_aligner(model_dir: Optional[str] = None) -> SherpaCTCAligner:
    global _global_aligner
    if _global_aligner is None or (model_dir and _global_aligner.model_dir != model_dir):
        _global_aligner = SherpaCTCAligner(model_dir=model_dir)
    return _global_aligner


def generate_ctc_alignment(
    wav_path: str,
    text: str,
    model_dir: Optional[str] = None,
    force_fallback: bool = False
) -> Dict[str, Any]:
    """
    Core Contract API for Milestone 1 Audio Alignment (PROJECT.md Interface Contract).
    Returns {"text": text, "duration": float, "alignment_method": str, "characters": [{"char": ..., "pinyin": ..., "start": ..., "end": ...}]}.
    """
    duration = get_audio_duration(wav_path)

    if force_fallback:
        chars = linear_syllable_interpolation(text, duration)
        method = "linear_interpolation_fallback"
    else:
        aligner = get_global_aligner(model_dir=model_dir)
        chars, method = aligner.align(wav_path, text, duration)

    return {
        "text": text,
        "duration": duration,
        "alignment_method": method,
        "characters": chars
    }


def get_active_character_at_time(
    alignment_data: Optional[Dict[str, Any]],
    current_time: float
) -> Optional[Dict[str, Any]]:
    """
    Helper for M3 Real-Time Karaoke Highlight Glow (#FFE066 / #FFD700).
    Returns the character entry active at current_time (in seconds), or None.
    Defensively handles None alignment_data, {"characters": None}, and non-finite timestamps.
    """
    if not alignment_data or not isinstance(alignment_data, dict):
        return None

    if current_time is None or math.isnan(current_time) or math.isinf(current_time):
        return None

    chars = alignment_data.get("characters") or []
    for idx, item in enumerate(chars):
        if not isinstance(item, dict):
            continue
        t_start = item.get("start")
        t_end = item.get("end")
        if t_start is None or t_end is None:
            continue
        if t_start <= current_time < t_end:
            return {
                "index": idx,
                "char": item.get("char", ""),
                "pinyin": item.get("pinyin", ""),
                "start": t_start,
                "end": t_end
            }
    return None


class StoryAlignmentGenerator:
    """
    Batch Story Alignment Generator.
    Processes all 17-18 audio sections for a story row, generates individual
    alignment_{section}.json files and aggregated alignment_manifest.json.
    """

    def __init__(self, model_dir: Optional[str] = None, force_fallback: bool = False):
        self.model_dir = model_dir
        self.force_fallback = force_fallback

    def generate_story_alignment(
        self,
        voice_dir: str,
        script_payload: Dict[str, Any],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Iterates over all audio sections in voice_dir matching script_payload.
        Outputs alignment_manifest.json and alignment_{section}.json files.
        """
        output_dir = output_dir or voice_dir
        os.makedirs(output_dir, exist_ok=True)

        title = script_payload.get("title", "")
        row_id = script_payload.get("row_id", 2)
        raw_scenes = script_payload.get("scenes") or script_payload.get("lines") or []
        vocabulary = script_payload.get("vocabulary", [])
        outro = script_payload.get("outro", {})

        sections_to_align = []
        if "audio_manifest" in script_payload and isinstance(script_payload["audio_manifest"], list):
            for item in script_payload["audio_manifest"]:
                sec = item.get("section", "")
                text = item.get("text", "")
                wav_path = item.get("path", "")
                wav_name = os.path.basename(wav_path) if wav_path else f"{sec}.wav"
                if sec and text:
                    sections_to_align.append((sec, wav_name, text))
        else:
            if title:
                sections_to_align.append(("title", "title.wav", title))

            for idx, sc in enumerate(raw_scenes, start=1):
                s_num = sc.get("scene_num", idx)
                zh = sc.get("zh", "").strip()
                if zh:
                    sections_to_align.append((f"scene{s_num}", f"scene{s_num}.wav", zh))

            vocab_words = []
            for v_idx, v in enumerate(vocabulary[:5], start=1):
                w = v.get("word", "").strip()
                if w:
                    vocab_words.append(w)
                    sections_to_align.append((f"vocab_{v_idx}", f"vocab_{v_idx}.wav", w))

            if (script_payload.get("include_vocab_recap", False) or os.path.isfile(os.path.join(voice_dir, "vocab.wav"))) and vocab_words:
                sections_to_align.append(("vocab", "vocab.wav", " ".join(vocab_words)))

            outro_text = outro.get("zh", "这些生词来自故事……") if isinstance(outro, dict) else str(outro)
            if outro_text:
                sections_to_align.append(("outro_loop", "outro_loop.wav", outro_text))

        total_duration = 0.0
        sections_manifest = {}
        fallback_count = 0

        for sec_name, wav_name, text in sections_to_align:
            wav_path = os.path.join(voice_dir, wav_name)
            if not os.path.isfile(wav_path):
                logger.warning(f"Audio file {wav_path} not found. Skipping section {sec_name}.")
                continue

            align_res = generate_ctc_alignment(
                wav_path=wav_path,
                text=text,
                model_dir=self.model_dir,
                force_fallback=self.force_fallback
            )
            align_res["section"] = sec_name
            align_res["file"] = wav_name
            total_duration += align_res["duration"]

            if align_res["alignment_method"] == "linear_interpolation_fallback":
                fallback_count += 1

            # Save individual section alignment
            sec_path = os.path.join(output_dir, f"alignment_{sec_name}.json")
            with open(sec_path, "w", encoding="utf-8") as f:
                json.dump(align_res, f, ensure_ascii=False, indent=2)

            sections_manifest[sec_name] = align_res

        overall_manifest = {
            "row_id": row_id,
            "title": title,
            "engine": "sherpa-onnx-ctc",
            "fallback_used": fallback_count > 0,
            "fallback_sections_count": fallback_count,
            "total_sections": len(sections_manifest),
            "total_duration": round(total_duration, 3),
            "sections": sections_manifest
        }

        manifest_path = os.path.join(output_dir, "alignment_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(overall_manifest, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ Generated alignment manifest for {len(sections_manifest)} sections in {manifest_path}")
        return overall_manifest
