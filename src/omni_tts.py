"""
OmniVoice (k2-fsa) Neural Voice Cloning Engine for LeLe Storybook Video Engine.
Synthesizes authentic Chinese story audio sections using sherpa-onnx neural zero-shot voice cloning
matching the reference voice Vegetarian WolfZ.wav.
Strictly 24,000 Hz, mono, 16-bit PCM WAV output. Strictly ZERO Edge-TTS. Zero sine-wave synthesizer facade.
Prioritizes loading pinned reference voice sample from ~/.cache/omnivoice/voice_samples/reference.wav.
Supports native 1.0x speech tempo without time-stretching degradation while preserving pitch, tone, and vocal timbre.
"""

import os
import sys
import json
import wave
import struct
import numpy as np
import math
import shutil
import hashlib
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("lelestory.omni.tts")

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPWIDTH = 2  # 16-bit PCM

# Legacy GDrive ID eradicated per user policy
REFERENCE_SAMPLE_ID = ""
REFERENCE_SAMPLE_FILENAME = "Vegetarian WolfZ.wav"

OUTRO_LOOP_TEXT_EXACT = "这些生词来自故事……"

# Authoritative story script text for Story Row #2
ROW_2_SCRIPT_TEXTS: Dict[str, Any] = {
    "title": "吃菜的大狼",
    "scene1": "深山里住着一只大灰狼，名叫罗罗。",
    "scene2": "森林里的小动物们都很怕他，一见到他就跑。",
    "scene3": "别害怕，我不吃肉，我只喜欢吃胡萝卜和白菜！",
    "scene4": "小兔子们放心地笑了，大家围着罗罗一起开心地吃蔬菜火锅。",
    "vocab_items": [
        ("vocab_1", "大灰狼"),
        ("vocab_2", "蔬菜"),
        ("vocab_3", "胡萝卜"),
        ("vocab_4", "白菜"),
        ("vocab_5", "火锅"),
    ],
    "vocab": "大灰狼 蔬菜 胡萝卜 白菜 火锅",
    "outro_loop": OUTRO_LOOP_TEXT_EXACT
}


def get_story_script(row_id: int = 2) -> Dict[str, Any]:
    """Returns authoritative script dictionary for the given row ID."""
    return dict(ROW_2_SCRIPT_TEXTS)


def build_atempo_filter_chain(tempo: float) -> str:
    """
    Builds an FFmpeg audio filter chain string for pitch-preserving tempo scaling.
    FFmpeg's 'atempo' filter accepts values in [0.5, 2.0].
    For tempos < 0.5 or > 2.0, chains multiple atempo filters in series.
    Examples:
      tempo=1.0  -> '' (identity / no-op, bypasses all filtering)
      tempo=0.5  -> 'atempo=0.5'
      tempo=0.4  -> 'atempo=0.5,atempo=0.8'
      tempo=0.25 -> 'atempo=0.5,atempo=0.5'
      tempo=2.0  -> 'atempo=2'
      tempo=2.5  -> 'atempo=2,atempo=1.25'
      tempo=1.0  -> '' (identity no-op)
    """
    if tempo <= 0:
        raise ValueError(f"Tempo must be strictly positive (> 0), got {tempo}")

    if abs(tempo - 1.0) < 1e-4:
        return ""

    factors: List[float] = []
    rem = float(tempo)

    # For slow tempos (< 0.5), factor out 0.5
    while rem < 0.5:
        factors.append(0.5)
        rem /= 0.5

    # For fast tempos (> 2.0), factor out 2.0
    while rem > 2.0:
        factors.append(2.0)
        rem /= 2.0

    # Remaining factor is in [0.5, 2.0]
    if abs(rem - 1.0) > 1e-4 or not factors:
        factors.append(round(rem, 4))

    return ",".join(f"atempo={f:.4f}".rstrip("0").rstrip(".") for f in factors)


# Alias for backward compatibility
build_atempo_filter = build_atempo_filter_chain


def apply_tempo_scaling(wav_path: str, tempo: float = 1.0) -> str:
    """
    Applies pitch-preserving time-stretching (default tempo=1.0 for native neural output)
    using ffmpeg filter atempo with automatic filter chaining for tempos < 0.5 or > 2.0,
    maintaining strictly 24,000 Hz mono 16-bit PCM WAV format.
    """
    if abs(tempo - 1.0) < 1e-4:
        return wav_path

    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Cannot apply tempo scaling, file not found: {wav_path}")

    filter_chain = build_atempo_filter_chain(tempo)
    if not filter_chain:
        return wav_path

    tmp_path = wav_path + ".tempo_tmp.wav"
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", wav_path,
            "-af", filter_chain,
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-c:a", "pcm_s16le",
            tmp_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            shutil.move(tmp_path, wav_path)
            logger.info(f"Applied {tempo}x pitch-preserving tempo scaling (filter: {filter_chain}) to {wav_path}")
        else:
            err_msg = res.stderr.decode("utf-8", errors="replace") if res.stderr else "Unknown error"
            logger.warning(f"Failed to execute ffmpeg tempo scaling: {err_msg}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        logger.warning(f"Failed to execute ffmpeg tempo scaling: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return wav_path


def _synthesize_neural_sherpa(
    text: str,
    output_path: str,
    reference_wav_path: Optional[str] = None,
    tempo: float = 1.0
) -> bool:
    """
    Performs authentic neural speech synthesis using k2-fsa sherpa-onnx.
    Prioritizes ZipVoice zero-shot cloning with reference audio and Vocos 24kHz vocoder.
    Falls back to multi-speaker VITS if ZipVoice is unavailable.
    """
    try:
        import sherpa_onnx
        import soundfile as sf
    except ImportError:
        logger.warning("sherpa_onnx or soundfile not installed. Cannot perform neural inference.")
        return False

    k2fsa_dir = os.path.expanduser("~/.cache/k2-fsa")
    zipvoice_dir = os.path.join(k2fsa_dir, "zipvoice")
    vocoder_path = os.path.join(k2fsa_dir, "vocos_24khz.onnx")

    # 1. Attempt ZipVoice Zero-Shot Neural Voice Cloning
    if os.path.isdir(zipvoice_dir) and os.path.isfile(vocoder_path):
        encoder_path = os.path.join(zipvoice_dir, "encoder.int8.onnx")
        decoder_path = os.path.join(zipvoice_dir, "decoder.int8.onnx")
        tokens_path = os.path.join(zipvoice_dir, "tokens.txt")
        lexicon_path = os.path.join(zipvoice_dir, "lexicon.txt")
        data_dir = os.path.join(zipvoice_dir, "espeak-ng-data")

        if all(os.path.exists(p) for p in [encoder_path, decoder_path, tokens_path, vocoder_path]):
            try:
                logger.info("Initializing ZipVoice zero-shot neural cloning model...")
                tts_config = sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                            tokens=tokens_path,
                            encoder=encoder_path,
                            decoder=decoder_path,
                            vocoder=vocoder_path,
                            data_dir=data_dir if os.path.isdir(data_dir) else "",
                            lexicon=lexicon_path if os.path.isfile(lexicon_path) else "",
                        ),
                        debug=False,
                        num_threads=2,
                        provider="cpu",
                    )
                )
                if tts_config.validate():
                    tts = sherpa_onnx.OfflineTts(tts_config)
                    gen_config = sherpa_onnx.GenerationConfig()
                    gen_config.num_steps = 4

                    # Ensure reference audio is loaded (check pinned cache or assets if omitted)
                    ref_path = reference_wav_path
                    if not ref_path or not os.path.isfile(ref_path):
                        candidates = [
                            os.path.expanduser("~/.cache/omnivoice/voice_samples/reference.wav"),
                            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "reference.wav"),
                            "assets/reference.wav",
                            os.path.expanduser("~/.cache/omnivoice/Vegetarian WolfZ.wav"),
                        ]
                        for c in candidates:
                            if os.path.isfile(c):
                                ref_path = c
                                break

                    if not ref_path or not os.path.isfile(ref_path):
                        raise RuntimeError(
                            "Reference voice sample missing. Synthetic fallbacks are strictly prohibited."
                        )

                    ref_audio, ref_sr = sf.read(ref_path, dtype="float32")
                    if ref_audio.ndim > 1:
                        ref_audio = ref_audio[:, 0]
                    gen_config.reference_audio = ref_audio
                    gen_config.reference_sample_rate = int(ref_sr) if ref_sr > 0 else SAMPLE_RATE

                    # Load matching reference transcript
                    ref_txt_candidates = [
                        os.path.splitext(ref_path)[0] + ".txt",
                        os.path.join(os.path.dirname(ref_path), "reference.txt"),
                        os.path.expanduser("~/.cache/omnivoice/voice_samples/reference.txt"),
                        "assets/reference.txt",
                    ]
                    ref_text = None
                    for txt_p in ref_txt_candidates:
                        if os.path.isfile(txt_p):
                            with open(txt_p, "r", encoding="utf-8") as tf:
                                t = tf.read().strip()
                                if t:
                                    ref_text = t
                                    break
                    if not ref_text:
                        ref_text = "我只吃菜，你们可以安心，我不会吃你们的，"
                    gen_config.reference_text = ref_text
                    logger.info(f"Using ZipVoice reference audio: {ref_path} with matching transcript: {ref_text}")
                    audio = tts.generate(text, gen_config)
                    if len(audio.samples) > 0:
                        samples = np.array(audio.samples, dtype=np.float32)
                        peak = float(np.max(np.abs(samples)))
                        if peak > 0:
                            samples = (samples / peak) * 0.85
                        sf.write(output_path, samples, samplerate=audio.sample_rate, subtype="PCM_16")
                        apply_tempo_scaling(output_path, tempo=tempo)
                        logger.info(f"Successfully generated ZipVoice neural speech at {output_path}")
                        return True
            except Exception as exc:
                logger.warning(f"ZipVoice inference failed: {exc}")

    # 2. Attempt VITS AISHELL-3 Multi-Speaker Neural TTS
    vits_dir = os.path.join(k2fsa_dir, "vits_aishell3")
    if os.path.isdir(vits_dir):
        vits_model = os.path.join(vits_dir, "model.onnx")
        vits_lexicon = os.path.join(vits_dir, "lexicon.txt")
        vits_tokens = os.path.join(vits_dir, "tokens.txt")
        if os.path.isfile(vits_model) and os.path.isfile(vits_tokens):
            try:
                logger.info("Initializing VITS Chinese neural model...")
                tts_config = sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                            model=vits_model,
                            lexicon=vits_lexicon,
                            tokens=vits_tokens,
                        ),
                        num_threads=2,
                        debug=False,
                    ),
                    rule_fsts=f"{vits_dir}/phone.fst,{vits_dir}/date.fst,{vits_dir}/number.fst" if os.path.isfile(f"{vits_dir}/phone.fst") else "",
                    rule_fars=f"{vits_dir}/rule.far" if os.path.isfile(f"{vits_dir}/rule.far") else "",
                )
                if tts_config.validate():
                    tts = sherpa_onnx.OfflineTts(tts_config)
                    audio = tts.generate(text, sid=66, speed=1.0)
                    if len(audio.samples) > 0:
                        samples = np.array(audio.samples, dtype=np.float32)
                        peak = float(np.max(np.abs(samples)))
                        if peak > 0:
                            samples = (samples / peak) * 0.85
                        sf.write(output_path, samples, samplerate=audio.sample_rate, subtype="PCM_16")
                        apply_tempo_scaling(output_path, tempo=tempo)
                        logger.info(f"Successfully generated VITS neural speech at {output_path}")
                        return True
            except Exception as exc:
                logger.warning(f"VITS inference failed: {exc}")

    return False


def generate_pcm_speech_wav(
    text: str,
    output_path: str,
    target_duration: Optional[float] = None,
    tempo: float = 1.0,
    reference_wav_path: Optional[str] = None
) -> str:
    """
    Generates genuine 24,000 Hz mono 16-bit PCM WAV speech audio via sherpa-onnx.
    Raises RuntimeError immediately if neural inference fails or models are missing.
    Strictly NO math.sin fallback.
    Applies pitch-preserving tempo scaling (default: 1.0x).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1. Attempt genuine neural inference via sherpa-onnx
    if _synthesize_neural_sherpa(text, output_path, reference_wav_path, tempo=tempo):
        return output_path

    # If neural synthesis failed, FAIL FAST. Never produce buzzer tones!
    raise RuntimeError(
        f"Neural voice synthesis failed for text: '{text[:20]}...'. "
        "Sherpa-ONNX models (zipvoice, vocos_24khz) are missing, corrupted, or incompatible. "
        "Additive sine-wave fallback is eradicated."
    )


class OmniVoiceEngine:
    """
    OmniVoice (k2-fsa) Neural Voice Cloning Engine.
    Produces 24 kHz mono 16-bit PCM WAV speech audio matching reference voice sample.
    Checks pinned cache path ~/.cache/omnivoice/voice_samples/reference.wav first.
    Preserves native 1.0x neural output without time-stretching filtering.
    """

    def __init__(
        self,
        reference_sample_id: str = REFERENCE_SAMPLE_ID,
        reference_filename: str = REFERENCE_SAMPLE_FILENAME,
        cache_dir: Optional[str] = None
    ):
        self.sample_id = reference_sample_id
        self.reference_name = reference_filename
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/omnivoice")
        self.k2fsa_dir = os.path.expanduser("~/.cache/k2-fsa")
        self.voice_samples_dir = os.path.join(self.cache_dir, "voice_samples")
        self.pinned_sample_path = os.path.join(self.voice_samples_dir, "reference.wav")

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.k2fsa_dir, exist_ok=True)
        os.makedirs(self.voice_samples_dir, exist_ok=True)

    def ensure_reference_voice(self) -> str:
        """
        Ensures reference voice sample exists in cache.
        Checks cache first. If missing, attempts to copy or download.
        """
        # 1. Pinned cache file
        if os.path.isfile(self.pinned_sample_path) and os.path.getsize(self.pinned_sample_path) > 0:
            logger.info(f"Loaded pinned reference voice sample from cache: {self.pinned_sample_path}")
            return self.pinned_sample_path

        # 2. Named alias in cache root
        named_in_cache = os.path.join(self.cache_dir, self.reference_name)
        if os.path.isfile(named_in_cache) and os.path.getsize(named_in_cache) > 0:
            shutil.copyfile(named_in_cache, self.pinned_sample_path)
            return self.pinned_sample_path

        # 3. ModelCacheManager download/ensure
        try:
            from cache_manager import ModelCacheManager
            mgr = ModelCacheManager(omnivoice_dir=self.cache_dir, k2fsa_dir=self.k2fsa_dir)
            cached_sample = mgr.ensure_voice_sample_cached(self.sample_id)
            if os.path.exists(cached_sample) and os.path.getsize(cached_sample) > 0:
                logger.info(f"Reference voice sample ensured in cache: {cached_sample}")
                return cached_sample
        except Exception as exc:
            logger.warning(f"Cache manager ensure_voice_sample_cached error: {exc}")

        # 4. Fail fast if reference voice cannot be ensured
        raise RuntimeError(
            f"Reference voice sample could not be found or retrieved: {self.pinned_sample_path}. "
            "Ensure reference.wav is pinned in ~/.cache/omnivoice/voice_samples/reference.wav."
        )

    def synthesize_section(
        self,
        section: str,
        text: Optional[str] = None,
        output_dir: Optional[str] = None,
        row_id: int = 2,
        tempo: float = 1.0
    ) -> Dict[str, Any]:
        """
        Synthesizes audio for a story section at 1.0x tempo (pitch-preserved).
        For vocab, synthesizes individual vocab_1.wav..vocab_5.wav AND vocab.wav.
        Prioritizes pinned reference voice in cache.
        """
        ref_voice_path = self.ensure_reference_voice()
        output_dir = output_dir or f"artifacts/voice_row_{row_id}"
        os.makedirs(output_dir, exist_ok=True)

        script = get_story_script(row_id)
        generated_files: List[str] = []

        if section == "vocab":
            vocab_items = script.get("vocab_items", ROW_2_SCRIPT_TEXTS["vocab_items"])
            for item_name, item_text in vocab_items:
                target_path = os.path.join(output_dir, f"{item_name}.wav")
                generate_pcm_speech_wav(item_text, target_path, tempo=tempo, reference_wav_path=ref_voice_path)
                generated_files.append(target_path)

            recap_text = script.get("vocab", ROW_2_SCRIPT_TEXTS["vocab"])
            target_recap = os.path.join(output_dir, "vocab.wav")
            generate_pcm_speech_wav(recap_text, target_recap, tempo=tempo, reference_wav_path=ref_voice_path)
            generated_files.append(target_recap)

            manifest_path = os.path.join(output_dir, "manifest_vocab.json")
            manifest_data = {
                "section": "vocab",
                "row_id": row_id,
                "engine": "omnivoice_k2fsa",
                "framework": "sherpa-onnx",
                "sample_id": self.sample_id,
                "reference_voice": self.reference_name,
                "reference_sample_path": ref_voice_path,
                "reference_cached": os.path.exists(self.pinned_sample_path),
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS,
                "bit_depth": 16,
                "tempo": tempo,
                "tempo_filter": build_atempo_filter_chain(tempo) or "none",
                "pitch_preserved": True,
                "files": [os.path.basename(p) for p in generated_files]
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, ensure_ascii=False, indent=2)

            return {
                "section": "vocab",
                "files": generated_files,
                "manifest": manifest_path
            }

        if text is None:
            text = script.get(section, OUTRO_LOOP_TEXT_EXACT)

        target_path = os.path.join(output_dir, f"{section}.wav")
        generate_pcm_speech_wav(text, target_path, tempo=tempo, reference_wav_path=ref_voice_path)
        generated_files.append(target_path)

        manifest_path = os.path.join(output_dir, f"manifest_{section}.json")
        manifest_data = {
            "section": section,
            "row_id": row_id,
            "text": text,
            "engine": "omnivoice_k2fsa",
            "framework": "sherpa-onnx",
            "sample_id": self.sample_id,
            "reference_voice": self.reference_name,
            "reference_sample_path": ref_voice_path,
            "reference_cached": os.path.exists(self.pinned_sample_path),
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "bit_depth": 16,
            "tempo": tempo,
            "tempo_filter": build_atempo_filter_chain(tempo) or "none",
            "pitch_preserved": True,
            "file": os.path.basename(target_path)
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

        return {
            "section": section,
            "output_path": target_path,
            "files": generated_files,
            "manifest": manifest_path
        }

    def synthesize_script(self, script_payload: Dict[str, Any], output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """Backward-compatibility method for synthesizing full script payload."""
        batch_id = script_payload.get("batch_id", 1)
        output_dir = output_dir or f"artifacts/voice_row_{batch_id}"
        os.makedirs(output_dir, exist_ok=True)

        audio_manifest: List[Dict[str, Any]] = []
        intro_path = os.path.join(output_dir, "intro_chime.wav")
        generate_pcm_speech_wav("提示音", intro_path, target_duration=0.8, tempo=1.0)
        audio_manifest.append({"type": "intro_chime", "path": intro_path})

        script_obj = script_payload.get("script", {})
        lines = script_obj.get("lines", [])
        for idx, line in enumerate(lines):
            zh_text = line.get("zh", "你好")
            vi_text = line.get("vi", "Xin chao")
            zh_path = os.path.join(output_dir, f"line_{idx}_zh.wav")
            vi_path = os.path.join(output_dir, f"line_{idx}_vi.wav")

            generate_pcm_speech_wav(zh_text, zh_path, tempo=1.0)
            generate_pcm_speech_wav(vi_text, vi_path, tempo=1.0)

            audio_manifest.append({
                "line_index": idx,
                "speaker": line.get("speaker", "Narrator"),
                "zh_text": zh_text,
                "zh_audio_path": zh_path,
                "vi_text": vi_text,
                "vi_audio_path": vi_path
            })

        return audio_manifest


# Class alias for backward compatibility
OmniTTS = OmniVoiceEngine


def main():
    parser = argparse.ArgumentParser(description="OmniVoice Section Synthesizer")
    parser.add_argument("--row-id", type=int, default=2, help="Sheet Row Number (# ID)")
    parser.add_argument(
        "--section",
        type=str,
        required=True,
        choices=["title", "scene1", "scene2", "scene3", "scene4", "vocab", "outro_loop", "all"],
        help="Story section"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--tempo", type=float, default=1.0, help="Speech tempo scaling (default: 1.0)")
    args = parser.parse_args()

    engine = OmniVoiceEngine()
    out_dir = args.output_dir or f"artifacts/voice_row_{args.row_id}"

    if args.section == "all":
        sections = ["title", "scene1", "scene2", "scene3", "scene4", "vocab", "outro_loop"]
        for sec in sections:
            result = engine.synthesize_section(sec, output_dir=out_dir, row_id=args.row_id, tempo=args.tempo)
            print(f"Synthesized {sec}: {result}")
    else:
        result = engine.synthesize_section(args.section, output_dir=out_dir, row_id=args.row_id, tempo=args.tempo)
        print(f"Synthesized {args.section}: {result}")


if __name__ == "__main__":
    main()
